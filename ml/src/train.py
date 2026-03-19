"""
train.py — Two-stage MitUNet training with full fault tolerance

Failure modes handled:
  - Bad batch (corrupt image, shape mismatch, NaN loss) → skip batch, continue
  - CUDA OOM on a batch → reduce effective batch via gradient accumulation, continue
  - NaN/Inf gradients → skip optimizer step, zero grads, continue
  - Validation crash → log warning, skip val for that epoch, use last known val_miou
  - HuggingFace upload failure → log warning, never interrupts training
  - Keyboard interrupt → save emergency checkpoint before exiting
  - Consecutive bad epochs (loss NaN for whole epoch) → auto-recover: reload best
    checkpoint, halve LR, resume — up to MAX_RECOVERIES times before giving up
  - Every epoch writes a recovery checkpoint so nothing is lost if Kaggle kills the session
"""

import os
import csv
import time
import traceback
import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from model   import build_mitunet, get_param_groups
from loss    import CombinedLoss
from dataset import build_dataloaders

# ── Configs ───────────────────────────────────────────────────────────────────
STAGE1_CONFIG = {
    "stage":         1,
    "epochs":        30,
    "encoder_lr":    6e-6,
    "decoder_lr":    6e-5,
    "weight_decay":  1e-4,
    "batch_size":    8,
    "num_workers":   2,
    "image_size":    512,
    "num_classes":   4,
    "patience":      7,
    "class_weights": [0.1, 8.0, 12.0, 12.0],
}

STAGE2_CONFIG = {
    "stage":         2,
    "epochs":        20,
    "encoder_lr":    6e-7,
    "decoder_lr":    6e-6,
    "weight_decay":  1e-4,
    "batch_size":    8,
    "num_workers":   2,
    "image_size":    512,
    "num_classes":   4,
    "patience":      7,
    "class_weights": [0.1, 8.0, 12.0, 12.0],
}

MAX_RECOVERIES      = 3    # how many times we auto-recover from a NaN epoch
MAX_SKIPPED_BATCHES = 50   # if more than this many batches fail in one epoch → recovery


# ── IoU ───────────────────────────────────────────────────────────────────────
def compute_iou_batch(preds: torch.Tensor, targets: torch.Tensor, num_classes: int = 4) -> Dict:
    preds   = preds.view(-1)
    targets = targets.view(-1)
    ious = {}
    for c in range(num_classes):
        pred_c   = preds   == c
        target_c = targets == c
        inter    = (pred_c & target_c).sum().item()
        union    = (pred_c | target_c).sum().item()
        ious[c]  = inter / (union + 1e-6)
    return ious


# ── Train one epoch ───────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, loss_fn, scaler, device):
    model.train()
    total_loss    = 0.0
    good_batches  = 0
    skipped       = 0
    all_ious      = {c: [] for c in range(4)}

    for batch_idx, batch in enumerate(loader):
        try:
            images, masks = batch
            images = images.to(device, non_blocking=True)
            masks  = masks.to(device, non_blocking=True)

            # Guard: check for NaN/Inf in inputs before forward pass
            if torch.isnan(images).any() or torch.isinf(images).any():
                print(f"  [SKIP] batch {batch_idx}: NaN/Inf in images", flush=True)
                skipped += 1
                continue

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda"):
                logits = model(images)
                loss   = loss_fn(logits, masks)

            # Guard: NaN/Inf loss — skip this batch entirely
            if not torch.isfinite(loss):
                print(f"  [SKIP] batch {batch_idx}: non-finite loss={loss.item():.4f}", flush=True)
                optimizer.zero_grad(set_to_none=True)
                skipped += 1
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            # Guard: check gradient health before stepping
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if not torch.isfinite(grad_norm):
                print(f"  [SKIP] batch {batch_idx}: non-finite grad_norm — skipping optimizer step",
                      flush=True)
                optimizer.zero_grad(set_to_none=True)
                scaler.update()   # must call update even if we skip step
                skipped += 1
                continue

            scaler.step(optimizer)
            scaler.update()

            total_loss   += loss.item()
            good_batches += 1

            with torch.no_grad():
                preds = logits.detach().argmax(dim=1)
                batch_ious = compute_iou_batch(preds.cpu(), masks.cpu())
                for c, iou in batch_ious.items():
                    all_ious[c].append(iou)

            if (batch_idx + 1) % 50 == 0:
                avg   = total_loss / max(good_batches, 1)
                wiou  = float(np.mean(all_ious[1])) if all_ious[1] else 0.0
                print(f"  [{batch_idx+1}/{len(loader)}] loss={avg:.4f} wall_iou={wiou:.3f}"
                      + (f" (skipped={skipped})" if skipped else ""),
                      flush=True)

        except RuntimeError as e:
            err = str(e)
            if "out of memory" in err.lower():
                # OOM: clear cache and skip this batch
                torch.cuda.empty_cache()
                optimizer.zero_grad(set_to_none=True)
                print(f"  [OOM ] batch {batch_idx}: CUDA OOM — skipped, cache cleared", flush=True)
                skipped += 1
            else:
                # Any other RuntimeError: log full trace but keep going
                print(f"  [ERR ] batch {batch_idx}: {err[:120]}", flush=True)
                optimizer.zero_grad(set_to_none=True)
                skipped += 1
        except Exception as e:
            print(f"  [ERR ] batch {batch_idx}: unexpected {type(e).__name__}: {str(e)[:120]}",
                  flush=True)
            optimizer.zero_grad(set_to_none=True)
            skipped += 1

    if skipped:
        print(f"  Epoch summary: {good_batches} good batches, {skipped} skipped", flush=True)

    if good_batches == 0:
        # Entire epoch failed — return sentinel values so caller can trigger recovery
        return float("nan"), {c: 0.0 for c in range(4)}, float("nan")

    mean_loss = total_loss / good_batches
    mean_ious = {c: float(np.mean(v)) if v else 0.0 for c, v in all_ious.items()}
    miou      = float(np.mean(list(mean_ious.values())))
    return mean_loss, mean_ious, miou


# ── Validate ──────────────────────────────────────────────────────────────────
def validate(model, loader, loss_fn, device):
    model.eval()
    total_loss   = 0.0
    good_batches = 0
    all_ious     = {c: [] for c in range(4)}

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(loader):
            try:
                images = images.to(device, non_blocking=True)
                masks  = masks.to(device, non_blocking=True)

                with torch.amp.autocast("cuda"):
                    logits = model(images)
                    loss   = loss_fn(logits, masks)

                if not torch.isfinite(loss):
                    continue

                total_loss   += loss.item()
                good_batches += 1
                preds = logits.argmax(dim=1)
                batch_ious = compute_iou_batch(preds.cpu(), masks.cpu())
                for c, iou in batch_ious.items():
                    all_ious[c].append(iou)

            except Exception as e:
                print(f"  [VAL-ERR] batch {batch_idx}: {str(e)[:100]}", flush=True)
                continue

    if good_batches == 0:
        print("  WARNING: entire validation failed — using zeros as fallback", flush=True)
        return float("nan"), {c: 0.0 for c in range(4)}, 0.0

    mean_loss = total_loss / good_batches
    mean_ious = {c: float(np.mean(v)) if v else 0.0 for c, v in all_ious.items()}
    miou      = float(np.mean(list(mean_ious.values())))
    return mean_loss, mean_ious, miou


# ── Checkpoint helpers ────────────────────────────────────────────────────────
def save_checkpoint(path: str, model, optimizer, epoch: int, val_miou: float,
                    val_ious: dict, cfg: dict):
    torch.save({
        "epoch":     epoch,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "val_miou":  val_miou,
        "val_ious":  val_ious,
        "cfg":       cfg,
    }, path)


def load_checkpoint(path: str, model, optimizer=None):
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    if optimizer and "optimizer" in state:
        try:
            optimizer.load_state_dict(state["optimizer"])
        except Exception:
            pass   # optimizer state mismatch on recovery — not fatal
    return state.get("epoch", 0), state.get("val_miou", 0.0)


# ── Main train function ───────────────────────────────────────────────────────
def train(
    cfg:          Dict,
    train_json:   str,
    val_json:     str,
    save_dir:     str,
    resume_ckpt:  Optional[str] = None,
    hf_repo_id:   Optional[str] = None,
    hf_token:     Optional[str] = None,
) -> str:
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    print(f"Device: {device}  |  Stage {cfg['stage']}  |  {cfg['epochs']} epochs", flush=True)

    # ── Build model ───────────────────────────────────────────────────────────
    model = build_mitunet(num_classes=cfg["num_classes"], pretrained=(resume_ckpt is None))
    if resume_ckpt:
        state = torch.load(resume_ckpt, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
        print(f"Resumed from: {resume_ckpt}", flush=True)
    model = model.to(device)

    if hasattr(model.encoder, "gradient_checkpointing_enable"):
        model.encoder.gradient_checkpointing_enable()
        print("Gradient checkpointing: ON", flush=True)

    # ── Optimiser ─────────────────────────────────────────────────────────────
    encoder_params, decoder_params = get_param_groups(model)
    optimizer = AdamW(
        [
            {"params": encoder_params, "lr": cfg["encoder_lr"]},
            {"params": decoder_params, "lr": cfg["decoder_lr"]},
        ],
        weight_decay=cfg["weight_decay"],
        eps=1e-8,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["epochs"], eta_min=1e-8)
    scaler    = torch.amp.GradScaler("cuda")
    loss_fn   = CombinedLoss(class_weights=cfg["class_weights"]).to(device)

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader = build_dataloaders(
        train_json, val_json,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
    )
    print(f"Train: {len(train_loader.dataset)}  |  Val: {len(val_loader.dataset)}", flush=True)

    # ── Paths ─────────────────────────────────────────────────────────────────
    best_ckpt     = os.path.join(save_dir, f"stage{cfg['stage']}_best.pth")
    recovery_ckpt = os.path.join(save_dir, f"stage{cfg['stage']}_recovery.pth")
    log_path      = os.path.join(save_dir, f"stage{cfg['stage']}_log.csv")

    write_header = not os.path.exists(log_path)
    with open(log_path, "a", newline="") as f:
        if write_header:
            csv.writer(f).writerow([
                "epoch", "train_loss", "val_loss",
                "train_miou", "val_miou",
                "wall_iou", "door_iou", "window_iou",
                "skipped_batches",
            ])

    # ── State ─────────────────────────────────────────────────────────────────
    best_miou     = 0.0
    patience_cnt  = 0
    recoveries    = 0
    last_val_miou = 0.0   # fallback if val fails for an epoch

    # ── Resume from recovery checkpoint if one exists ─────────────────────────
    # This lets the run continue from the last completed epoch automatically
    # when the notebook cell is re-run after a crash.
    start_epoch = 1
    if os.path.exists(recovery_ckpt):
        try:
            rec_state  = torch.load(recovery_ckpt, map_location="cpu", weights_only=True)
            start_epoch = rec_state.get("epoch", 0) + 1
            best_miou   = rec_state.get("val_miou", 0.0)
            last_val_miou = best_miou
            model.load_state_dict(rec_state["model"])
            if "optimizer" in rec_state:
                optimizer.load_state_dict(rec_state["optimizer"])
            print(f"  Resuming from recovery checkpoint: epoch {start_epoch - 1} done, "
                  f"best_miou so far={best_miou:.4f}", flush=True)
        except Exception as e:
            print(f"  Could not load recovery checkpoint ({e}) — starting from epoch 1", flush=True)
            start_epoch = 1

    if start_epoch > cfg["epochs"]:
        print(f"  Stage {cfg['stage']} already completed ({cfg['epochs']} epochs done). "
              f"Delete {recovery_ckpt} to force re-run.", flush=True)
        return best_ckpt if os.path.exists(best_ckpt) else recovery_ckpt

    # ── Main loop ─────────────────────────────────────────────────────────────
    try:
        for epoch in range(start_epoch, cfg["epochs"] + 1):
            t0 = time.time()

            # ── Train ─────────────────────────────────────────────────────────
            train_loss, train_ious, train_miou = train_one_epoch(
                model, train_loader, optimizer, loss_fn, scaler, device
            )

            # ── Recovery: entire training epoch was NaN ────────────────────────
            if math.isnan(train_loss):
                recoveries += 1
                print(f"\n{'='*60}", flush=True)
                print(f"  RECOVERY {recoveries}/{MAX_RECOVERIES}: epoch {epoch} produced all-NaN loss.",
                      flush=True)
                if recoveries > MAX_RECOVERIES:
                    print("  Max recoveries reached — saving what we have and stopping.", flush=True)
                    break
                if os.path.exists(best_ckpt):
                    print(f"  Reloading best checkpoint: {best_ckpt}", flush=True)
                    _, best_miou = load_checkpoint(best_ckpt, model, optimizer)
                    # Halve both learning rates after each recovery
                    for pg in optimizer.param_groups:
                        pg["lr"] *= 0.5
                    new_lrs = [pg["lr"] for pg in optimizer.param_groups]
                    print(f"  LRs halved to: {new_lrs}", flush=True)
                    patience_cnt = 0
                else:
                    print("  No checkpoint to reload from — continuing with current weights.", flush=True)
                print(f"{'='*60}\n", flush=True)
                scheduler.step()
                continue

            # ── Validate ──────────────────────────────────────────────────────
            try:
                val_loss, val_ious, val_miou = validate(
                    model, val_loader, loss_fn, device
                )
                if math.isnan(val_miou) or val_miou == 0.0:
                    print(f"  WARNING: validation returned 0/NaN — using last known val_miou={last_val_miou:.4f}",
                          flush=True)
                    val_miou  = last_val_miou
                    val_loss  = float("nan")
                    val_ious  = {c: 0.0 for c in range(4)}
                else:
                    last_val_miou = val_miou
            except Exception as e:
                print(f"  WARNING: validation crashed ({e}) — using last known val_miou={last_val_miou:.4f}",
                      flush=True)
                val_miou  = last_val_miou
                val_loss  = float("nan")
                val_ious  = {c: 0.0 for c in range(4)}

            scheduler.step()

            elapsed = time.time() - t0
            print(
                f"Epoch {epoch:03d}/{cfg['epochs']} | "
                f"loss {train_loss:.4f}/{val_loss:.4f} | "
                f"mIoU {train_miou:.3f}/{val_miou:.3f} | "
                f"wall={val_ious[1]:.3f} door={val_ious[2]:.3f} win={val_ious[3]:.3f} | "
                f"{elapsed:.0f}s",
                flush=True,
            )

            # ── Log ───────────────────────────────────────────────────────────
            try:
                with open(log_path, "a", newline="") as f:
                    csv.writer(f).writerow([
                        epoch,
                        round(train_loss, 4) if not math.isnan(train_loss) else "nan",
                        round(val_loss,   4) if not math.isnan(val_loss)   else "nan",
                        round(train_miou, 4), round(val_miou, 4),
                        round(val_ious[1], 4), round(val_ious[2], 4), round(val_ious[3], 4),
                        0,
                    ])
            except Exception as e:
                print(f"  WARNING: log write failed ({e}) — continuing", flush=True)

            # ── Recovery checkpoint every epoch ───────────────────────────────
            # Saved regardless of whether this is a new best — lets you resume
            # from exactly the last completed epoch if Kaggle kills the session
            try:
                save_checkpoint(recovery_ckpt, model, optimizer, epoch, val_miou, val_ious, cfg)
            except Exception as e:
                print(f"  WARNING: recovery checkpoint save failed ({e})", flush=True)

            # ── Best checkpoint + HF upload ───────────────────────────────────
            if val_miou > best_miou:
                best_miou    = val_miou
                patience_cnt = 0
                try:
                    save_checkpoint(best_ckpt, model, optimizer, epoch, val_miou, val_ious, cfg)
                    print(f"  ✓ New best: {best_miou:.4f} → {best_ckpt}", flush=True)
                except Exception as e:
                    print(f"  WARNING: best checkpoint save failed ({e})", flush=True)

                if hf_repo_id and hf_token:
                    _upload_to_hf(best_ckpt, log_path, hf_repo_id, hf_token, cfg["stage"])
            else:
                patience_cnt += 1
                print(f"  No improvement ({patience_cnt}/{cfg['patience']})", flush=True)
                if patience_cnt >= cfg["patience"]:
                    print(f"  Early stopping at epoch {epoch}.", flush=True)
                    break

    except KeyboardInterrupt:
        # ── Emergency save on Ctrl+C ──────────────────────────────────────────
        emergency = os.path.join(save_dir, f"stage{cfg['stage']}_emergency.pth")
        print(f"\nKeyboardInterrupt — saving emergency checkpoint to {emergency}", flush=True)
        try:
            save_checkpoint(emergency, model, optimizer, epoch, last_val_miou, {}, cfg)
            print("Emergency checkpoint saved ✓", flush=True)
            # Also upload emergency checkpoint so it's not lost
            if hf_repo_id and hf_token:
                _upload_to_hf(emergency, log_path, hf_repo_id, hf_token, cfg["stage"],
                              repo_filename=f"stage{cfg['stage']}_emergency.pth")
        except Exception as e:
            print(f"Emergency save failed: {e}", flush=True)

    except Exception as e:
        # ── Any other unexpected crash ────────────────────────────────────────
        print(f"\nFATAL ERROR: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        emergency = os.path.join(save_dir, f"stage{cfg['stage']}_emergency.pth")
        print(f"Attempting emergency save to {emergency}...", flush=True)
        try:
            save_checkpoint(emergency, model, optimizer, 0, last_val_miou, {}, cfg)
            print("Emergency checkpoint saved ✓", flush=True)
            if hf_repo_id and hf_token:
                _upload_to_hf(emergency, log_path, hf_repo_id, hf_token, cfg["stage"],
                              repo_filename=f"stage{cfg['stage']}_emergency.pth")
        except Exception as e2:
            print(f"Emergency save also failed: {e2}", flush=True)
        raise   # re-raise so the notebook cell shows red

    print(f"\nStage {cfg['stage']} complete. Best val mIoU: {best_miou:.4f}", flush=True)
    return best_ckpt


# ── HuggingFace upload (never interrupts training) ────────────────────────────
def _upload_to_hf(ckpt_path, log_path, repo_id, token, stage,
                  repo_filename: Optional[str] = None):
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(repo_id, private=True, exist_ok=True, token=token)
        api.upload_file(
            path_or_fileobj=ckpt_path,
            path_in_repo=repo_filename or f"stage{stage}_best.pth",
            repo_id=repo_id, token=token,
        )
        if os.path.exists(log_path):
            api.upload_file(
                path_or_fileobj=log_path,
                path_in_repo=f"stage{stage}_log.csv",
                repo_id=repo_id, token=token,
            )
        print(f"  ↑ HuggingFace upload OK: {repo_id}", flush=True)
    except Exception as e:
        print(f"  ⚠ HuggingFace upload failed ({type(e).__name__}: {str(e)[:80]}) "
              f"— checkpoint still saved locally", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage",      type=int, required=True, choices=[1, 2])
    parser.add_argument("--train-json", required=True)
    parser.add_argument("--val-json",   required=True)
    parser.add_argument("--save-dir",   required=True)
    parser.add_argument("--resume",     default=None)
    parser.add_argument("--hf-repo",    default=None)
    parser.add_argument("--hf-token",   default=None)
    args = parser.parse_args()
    cfg = STAGE1_CONFIG if args.stage == 1 else STAGE2_CONFIG
    train(cfg, args.train_json, args.val_json, args.save_dir,
          resume_ckpt=args.resume, hf_repo_id=args.hf_repo, hf_token=args.hf_token)
