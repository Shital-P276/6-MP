"""
evaluate.py — IoU, precision, recall for 4-class floor plan segmentation

BUG FIX: original compute_metrics received batched tensors but the full
evaluate_checkpoint didn't accumulate per-pixel stats correctly across batches —
it was computing IoU per batch then averaging, which gives wrong results when
batches have very different class distributions (e.g. a batch with zero doors
would drag door IoU to zero incorrectly). Fixed to accumulate pixel-level
TP/FP/FN across ALL batches before computing final IoU.
"""

import torch
import numpy as np
from typing import Dict


def compute_metrics_accumulated(
    all_preds:   torch.Tensor,
    all_targets: torch.Tensor,
    num_classes: int = 4,
) -> Dict[str, float]:
    """
    Compute metrics from full-dataset accumulated predictions.
    Both tensors are 1D (all pixels flattened).

    FIX: accumulate TP/FP/FN across the entire dataset before dividing,
    not per-batch — gives correct IoU even for rare classes like doors/windows.
    """
    preds   = all_preds.view(-1).numpy()
    targets = all_targets.view(-1).numpy()

    ious, precisions, recalls = {}, {}, {}
    for c in range(num_classes):
        pred_c   = preds   == c
        target_c = targets == c
        tp = int((pred_c &  target_c).sum())
        fp = int((pred_c & ~target_c).sum())
        fn = int((~pred_c & target_c).sum())
        ious[c]       = tp / (tp + fp + fn + 1e-6)
        precisions[c] = tp / (tp + fp + 1e-6)
        recalls[c]    = tp / (tp + fn + 1e-6)

    miou = float(np.mean(list(ious.values())))
    return {
        "miou":           round(miou, 4),
        "bg_iou":         round(float(ious[0]), 4),
        "wall_iou":       round(float(ious[1]), 4),
        "door_iou":       round(float(ious[2]), 4),
        "window_iou":     round(float(ious[3]), 4),
        "wall_precision": round(float(precisions[1]), 4),
        "wall_recall":    round(float(recalls[1]), 4),
        "door_precision": round(float(precisions[2]), 4),
        "door_recall":    round(float(recalls[2]), 4),
        "win_precision":  round(float(precisions[3]), 4),
        "win_recall":     round(float(recalls[3]), 4),
    }


# Keep old name as alias for any existing callers
def compute_metrics(preds, targets, num_classes=4):
    return compute_metrics_accumulated(preds, targets, num_classes)


def evaluate_checkpoint(
    ckpt_path:  str,
    val_json:   str,
    batch_size: int = 8,
) -> Dict[str, float]:
    """
    Full evaluation of a saved checkpoint on a val/test split.
    Accumulates predictions across all batches before computing metrics.
    """
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from model   import build_mitunet
    from dataset import FloorPlanDataset, get_val_transforms
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model = build_mitunet(num_classes=4, pretrained=False)
    model.load_state_dict(state["model"])
    model = model.to(device).eval()

    ds     = FloorPlanDataset(val_json, transforms=get_val_transforms())
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)

    # FIX: accumulate ALL predictions as flat tensors, compute metrics once at end
    all_preds, all_targets = [], []
    with torch.no_grad():
        for images, masks in loader:
            with torch.amp.autocast("cuda"):
                logits = model(images.to(device))
            preds = logits.argmax(dim=1).cpu()
            all_preds.append(preds)
            all_targets.append(masks.cpu())

    all_preds   = torch.cat(all_preds,   dim=0)  # (N, H, W)
    all_targets = torch.cat(all_targets, dim=0)  # (N, H, W)
    metrics     = compute_metrics_accumulated(all_preds, all_targets)

    print("\n── Evaluation Results ─────────────────────────────")
    print(f"  mIoU:            {metrics['miou']:.4f}")
    print(f"  Wall  IoU:       {metrics['wall_iou']:.4f}  "
          f"P={metrics['wall_precision']:.3f}  R={metrics['wall_recall']:.3f}")
    print(f"  Door  IoU:       {metrics['door_iou']:.4f}  "
          f"P={metrics['door_precision']:.3f}  R={metrics['door_recall']:.3f}")
    print(f"  Window IoU:      {metrics['window_iou']:.4f}  "
          f"P={metrics['win_precision']:.3f}  R={metrics['win_recall']:.3f}")
    print("───────────────────────────────────────────────────\n")
    return metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",       required=True)
    parser.add_argument("--val-json",   required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    evaluate_checkpoint(args.ckpt, args.val_json, args.batch_size)
