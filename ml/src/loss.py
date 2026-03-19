"""
loss.py — Combined CrossEntropy + per-class Tversky loss

BUG FIXES:
1. class_weights tensor was created once in __init__ and stored as a plain tensor,
   not as a registered buffer. When model moves to GPU, CrossEntropyLoss.weight
   stays on CPU → device mismatch crash. Fixed: register as buffer.
2. Tversky was called with logits[:, c:c+1] (raw logits) — sigmoid inside
   TverskyLoss is correct, but we must NOT also apply softmax before this.
   The current approach is correct, keeping note explicit.
3. tv_loss was divided by (logits.shape[1] - 1) using shape at call time,
   which works, but made it fragile if called with wrong tensor. Fixed to
   use explicit num_classes from init.
"""

import torch
import torch.nn as nn


class TverskyLoss(nn.Module):
    """
    Tversky loss for a single binary channel.
    alpha=0.6: penalise FP heavily → clean, sharp boundaries
    beta=0.4:  penalise FN less   → don't miss thin walls
    """

    def __init__(self, alpha: float = 0.6, beta: float = 0.4, smooth: float = 1e-6):
        super().__init__()
        self.alpha  = alpha
        self.beta   = beta
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, 1, H, W) — raw logit for one class
            targets: (B, H, W)    — binary float mask for that class (0.0 or 1.0)
        """
        probs = torch.sigmoid(logits.squeeze(1))   # (B, H, W)
        TP = (probs * targets).sum()
        FP = (probs * (1.0 - targets)).sum()
        FN = ((1.0 - probs) * targets).sum()
        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        return 1.0 - tversky


class CombinedLoss(nn.Module):
    """
    CrossEntropy (global class balance) + per-class Tversky (boundary sharpness).

    Class weights [0.1, 8.0, 12.0, 12.0]:
      - Background  0.1  — majority class, suppress its gradient
      - Wall        8.0  — primary target
      - Door       12.0  — ~3% of pixels, needs heavy upweighting
      - Window     12.0  — ~2% of pixels, same
    """

    def __init__(
        self,
        class_weights: list = None,
        num_classes:   int   = 4,
        ce_weight:     float = 1.0,
        tv_weight:     float = 1.0,
        tversky_alpha: float = 0.6,
        tversky_beta:  float = 0.4,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.ce_weight   = ce_weight
        self.tv_weight   = tv_weight

        weights = torch.tensor(class_weights or [0.1, 8.0, 12.0, 12.0], dtype=torch.float32)
        # FIX: register as buffer so it moves to GPU with model.to(device)
        self.register_buffer("ce_weights", weights)

        self.tv = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, 4, H, W) — raw class logits (no softmax/sigmoid applied)
            targets: (B, H, W)    — integer class mask [0,1,2,3]  dtype=long
        Returns:
            scalar combined loss
        """
        # CrossEntropy — uses registered buffer (auto on correct device)
        ce_loss = nn.functional.cross_entropy(
            logits, targets.long(), weight=self.ce_weights
        )

        # Per-class Tversky on foreground classes only (skip class 0 = background)
        tv_loss = torch.tensor(0.0, device=logits.device)
        for c in range(1, self.num_classes):
            binary_target = (targets == c).float()
            tv_loss = tv_loss + self.tv(logits[:, c:c+1], binary_target)
        tv_loss = tv_loss / (self.num_classes - 1)

        return self.ce_weight * ce_loss + self.tv_weight * tv_loss


if __name__ == "__main__":
    loss_fn = CombinedLoss()
    # Simulate device move (catches the buffer bug if not registered correctly)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = loss_fn.to(device)
    logits  = torch.randn(2, 4, 512, 512, device=device)
    targets = torch.randint(0, 4, (2, 512, 512), device=device)
    loss    = loss_fn(logits, targets)
    print(f"Loss: {loss.item():.4f}  (device={loss.device})  ✓")
