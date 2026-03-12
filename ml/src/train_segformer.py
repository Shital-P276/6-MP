# src/train_segformer.py
#
# Fine-tunes SegFormer-B2 on floor plan segmentation (4 classes).
#
# Model: nvidia/segformer-b2-finetuned-ade-512-512  (pretrained on ADE20K)
# Task:  Semantic segmentation → background / wall / door / window
#
# Features:
#   - Mixed precision training (AMP) — halves VRAM, ~2x faster on Ampere GPUs
#   - Class-weighted cross-entropy — counters severe background imbalance
#   - Linear warmup LR scheduler
#   - Saves best checkpoint (by val mIoU) to checkpoints/segformer/best/
#   - TensorBoard logging (run: tensorboard --logdir runs/)
#   - Resumes from checkpoint if RESUME_FROM is set
#
# Hardware requirements:
#   - GPU (8GB+ VRAM): batch_size=8, ~6hr for 50 epochs on CubiCasa5k
#   - Google Colab A100: batch_size=16, ~2hr
#   - Colab T4 (free):   batch_size=4,  ~12hr — works but slow
#   - CPU only:          batch_size=2,  not recommended for full training
#
# Usage:
#   python src/train_segformer.py
#   python src/train_segformer.py --epochs 20 --batch-size 4   # quick test
#
# Output:
#   checkpoints/segformer/best/   — HuggingFace save_pretrained format
#   runs/segformer/               — TensorBoard logs

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    SegformerForSemanticSegmentation,
    get_linear_schedule_with_warmup,
)

# Add parent directory so imports work when running from project root
sys.path.insert(0, str(Path(__file__).parent))
from datasets import FloorPlanDataset
from augmentations import get_train_transforms, get_val_transforms


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

CONFIG = {
    # Model
    'model_name':    'nvidia/segformer-b2-finetuned-ade-512-512',
    'num_classes':   4,
    'image_size':    512,

    # Training
    'batch_size':    8,          # reduce to 4 if CUDA OOM
    'lr':            6e-5,       # peak learning rate
    'weight_decay':  0.01,
    'epochs':        50,
    'warmup_steps':  500,        # linear warmup steps

    # Class weights — counters background pixel dominance
    # Pixel distribution in CubiCasa5k: ~85% bg, ~12% wall, ~2% door, ~1% window
    # Formula: weight[c] = 1 / freq[c], scaled so wall = 3.0
    'class_weights': [0.5, 3.0, 5.0, 5.0],  # bg, wall, door, window

    # Data
    'data_root':     './data',
    'num_workers':   4,          # set to 0 on Windows if DataLoader errors

    # Checkpoints / logging
    'save_dir':      './checkpoints/segformer',
    'log_dir':       './runs/segformer',
    'resume_from':   None,       # set to checkpoint path to resume

    # Evaluation
    'val_every':     1,          # validate every N epochs
    'save_best_only': True,
}


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_iou_per_class(
    preds: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int = 4,
) -> list:
    """
    Compute per-class IoU (Intersection over Union).

    Args:
        preds:   (H, W) int64 — predicted class indices
        labels:  (H, W) int64 — ground truth class indices

    Returns:
        List of float IoU per class. NaN if class not present in gt or pred.
    """
    ious = []
    for cls in range(num_classes):
        pred_cls = (preds == cls)
        gt_cls   = (labels == cls)
        tp       = (pred_cls & gt_cls).sum().float()
        union    = (pred_cls | gt_cls).sum().float()
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append((tp / union).item())
    return ious


def format_metrics(mean_ious: np.ndarray) -> str:
    """Format per-class IoU into a readable string."""
    names = ['bg', 'wall', 'door', 'win']
    parts = [f"{n}={v:.3f}" for n, v in zip(names, mean_ious)]
    miou  = float(np.nanmean(mean_ious))
    return f"mIoU={miou:.4f}  |  " + "  ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg: dict):
    # ── Device ────────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"  Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU:    {torch.cuda.get_device_name(0)}")
        print(f"  VRAM:   {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    print(f"  Model:  {cfg['model_name']}")
    print(f"  Epochs: {cfg['epochs']}  |  Batch: {cfg['batch_size']}")
    print(f"{'='*60}\n")

    # ── Datasets & dataloaders ────────────────────────────────────────────
    train_ds = FloorPlanDataset(
        cfg['data_root'], 'train',
        get_train_transforms(cfg['image_size']),
    )
    val_ds = FloorPlanDataset(
        cfg['data_root'], 'val',
        get_val_transforms(cfg['image_size']),
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=cfg['num_workers'],
        pin_memory=(device.type == 'cuda'),
        drop_last=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=max(1, cfg['batch_size'] // 2),
        shuffle=False,
        num_workers=cfg['num_workers'],
    )

    # ── Model ─────────────────────────────────────────────────────────────
    print(f"Loading {cfg['model_name']} ...")
    model = SegformerForSemanticSegmentation.from_pretrained(
        cfg['model_name'],
        num_labels=cfg['num_classes'],
        id2label={0: 'background', 1: 'wall', 2: 'door', 3: 'window'},
        label2id={'background': 0, 'wall': 1, 'door': 2, 'window': 3},
        ignore_mismatched_sizes=True,   # classifier head is re-initialized
    ).to(device)

    if cfg.get('resume_from'):
        print(f"Resuming from {cfg['resume_from']}")
        state = torch.load(cfg['resume_from'], map_location=device)
        model.load_state_dict(state)

    # ── Optimizer & scheduler ─────────────────────────────────────────────
    # Separate LR for backbone vs decode head (backbone needs lower LR since
    # it's pretrained; decode head is randomly initialized)
    backbone_params = [
        p for n, p in model.named_parameters()
        if 'segformer.encoder' in n
    ]
    head_params = [
        p for n, p in model.named_parameters()
        if 'segformer.encoder' not in n
    ]
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': cfg['lr'] * 0.1},  # 10x lower
        {'params': head_params,     'lr': cfg['lr']},
    ], weight_decay=cfg['weight_decay'])

    total_steps = len(train_dl) * cfg['epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg['warmup_steps'],
        num_training_steps=total_steps,
    )

    # ── AMP scaler (only on CUDA) ─────────────────────────────────────────
    scaler      = GradScaler(enabled=(device.type == 'cuda'))
    class_wts   = torch.tensor(cfg['class_weights'], device=device)
    writer      = SummaryWriter(cfg['log_dir'])
    best_miou   = 0.0
    save_dir    = Path(cfg['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting training. Logs → {cfg['log_dir']}")
    print(f"Run: tensorboard --logdir {cfg['log_dir']}\n")

    # ── Training loop ─────────────────────────────────────────────────────
    for epoch in range(1, cfg['epochs'] + 1):

        # ──── TRAIN ────────────────────────────────────────────────────
        model.train()
        epoch_loss  = 0.0
        num_batches = 0

        for batch in train_dl:
            pixel_values = batch['pixel_values'].to(device)  # (B, 3, H, W)
            labels       = batch['labels'].to(device)        # (B, H, W)

            optimizer.zero_grad()

            with autocast(enabled=(device.type == 'cuda')):
                outputs = model(pixel_values=pixel_values)

                # SegFormer outputs logits at 1/4 resolution → upsample
                logits = F.interpolate(
                    outputs.logits,
                    size=(cfg['image_size'], cfg['image_size']),
                    mode='bilinear',
                    align_corners=False,
                )
                loss = F.cross_entropy(logits, labels, weight=class_wts)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss  += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        writer.add_scalar('train/loss', avg_loss, epoch)
        writer.add_scalar('train/lr_head',
                          optimizer.param_groups[1]['lr'], epoch)

        # ──── VALIDATE ────────────────────────────────────────────────
        if epoch % cfg['val_every'] == 0:
            model.eval()
            all_ious = []

            with torch.no_grad():
                for batch in val_dl:
                    pixel_values = batch['pixel_values'].to(device)
                    labels       = batch['labels'].to(device)

                    with autocast(enabled=(device.type == 'cuda')):
                        outputs = model(pixel_values=pixel_values)
                        logits  = F.interpolate(
                            outputs.logits,
                            size=(cfg['image_size'], cfg['image_size']),
                            mode='bilinear',
                            align_corners=False,
                        )

                    preds = logits.argmax(dim=1)  # (B, H, W)
                    for p, l in zip(preds.cpu(), labels.cpu()):
                        all_ious.append(
                            compute_iou_per_class(p, l, cfg['num_classes'])
                        )

            mean_ious = np.nanmean(all_ious, axis=0)
            miou      = float(np.nanmean(mean_ious))

            # Log to TensorBoard
            writer.add_scalar('val/mIoU',       miou,          epoch)
            writer.add_scalar('val/wall_IoU',   mean_ious[1],  epoch)
            writer.add_scalar('val/door_IoU',   mean_ious[2],  epoch)
            writer.add_scalar('val/window_IoU', mean_ious[3],  epoch)

            flag = ''
            if miou > best_miou:
                best_miou = miou
                model.save_pretrained(str(save_dir / 'best'))
                flag = '  ✓ SAVED'

            print(
                f"Epoch {epoch:3d}/{cfg['epochs']}  "
                f"loss={avg_loss:.4f}  "
                f"{format_metrics(mean_ious)}{flag}"
            )

            # Early stopping hint
            if mean_ious[1] < 0.30 and epoch >= 10:
                print("\n⚠️  Wall IoU < 30% at epoch 10. Check:")
                print("   1. Mask values are 0-3 (not 0/255)")
                print("   2. class_weights[1] (wall) is ≥ 3.0")
                print("   3. Data split JSON paths are correct\n")
        else:
            print(f"Epoch {epoch:3d}/{cfg['epochs']}  loss={avg_loss:.4f}")

    writer.close()
    print(f"\nTraining complete. Best mIoU: {best_miou:.4f}")
    print(f"Checkpoint saved to: {save_dir / 'best'}")

    # Save final config alongside checkpoint
    with open(save_dir / 'best' / 'train_config.json', 'w') as f:
        json.dump(cfg, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Train SegFormer for floor plan segmentation')
    p.add_argument('--epochs',      type=int,   default=None,
                   help=f"Override epochs (default: {CONFIG['epochs']})")
    p.add_argument('--batch-size',  type=int,   default=None,
                   help=f"Override batch size (default: {CONFIG['batch_size']})")
    p.add_argument('--lr',          type=float, default=None)
    p.add_argument('--data-root',   type=str,   default=None)
    p.add_argument('--save-dir',    type=str,   default=None)
    p.add_argument('--resume-from', type=str,   default=None)
    p.add_argument('--model',       type=str,   default=None,
                   help='HuggingFace model name (e.g. nvidia/segformer-b0-finetuned-ade-512-512)')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    cfg  = CONFIG.copy()

    # Apply CLI overrides
    if args.epochs:      cfg['epochs']      = args.epochs
    if args.batch_size:  cfg['batch_size']  = args.batch_size
    if args.lr:          cfg['lr']          = args.lr
    if args.data_root:   cfg['data_root']   = args.data_root
    if args.save_dir:    cfg['save_dir']    = args.save_dir
    if args.resume_from: cfg['resume_from'] = args.resume_from
    if args.model:       cfg['model_name']  = args.model

    train(cfg)
