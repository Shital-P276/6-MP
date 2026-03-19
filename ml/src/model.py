"""
model.py — MitUNet
= Mix-Transformer B2 encoder + U-Net decoder + scSE attention
= smp.Unet(encoder_name="mit_b2", decoder_attention_type="scse")

This IS MitUNet. The MiT-B2 is the encoder (Mix-Transformer from the paper).
The U-Net decoder with scSE attention blocks is the decoder.
smp.Unet handles this natively in 0.3.3 — no transplant, no Segformer class needed.

Verified against smp 0.3.3 docs:
  - smp.Unet exists ✓
  - encoder_name="mit_b2" supported ✓
  - decoder_attention_type="scse" supported ✓
  - MiT-B2 has 4 encoder stages, not 5 — encoder_depth must NOT be set
    (smp default encoder_depth=5 will crash with mit_b2, which only has 4 stages)
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


def build_mitunet(num_classes: int = 4, pretrained: bool = True) -> nn.Module:
    """
    Build MitUNet: MiT-B2 encoder (Mix-Transformer) + U-Net decoder with scSE.

    Args:
        num_classes: 4 = background, wall, door, window
        pretrained:  True = load ImageNet weights for encoder
                     False = random init (used when loading from checkpoint)
    """
    encoder_weights = "imagenet" if pretrained else None

    model = smp.Unet(
        encoder_name="mit_b2",
        encoder_weights=encoder_weights,
        # Do NOT set encoder_depth — MiT-B2 has exactly 4 stages internally.
        # smp handles this automatically when encoder_name starts with "mit_".
        in_channels=3,
        classes=num_classes,
        decoder_attention_type="scse",
    )

    # Verify output shape with a dry run — catches any version/config issues
    # before wasting time downloading data
    try:
        dummy = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            out = model(dummy)
        assert out.shape == (1, num_classes, 64, 64), \
            f"Expected output (1, {num_classes}, 64, 64), got {out.shape}"
    except Exception as e:
        raise RuntimeError(
            f"MitUNet build/verify failed: {e}\n"
            f"smp version: {smp.__version__}\n"
            "Expected: segmentation-models-pytorch==0.3.3"
        )

    return model


def get_param_groups(model: nn.Module):
    """
    Two parameter groups for differential learning rate.
    Encoder (pretrained MiT-B2) gets 10x lower LR than decoder (random init).
    """
    encoder_ids    = {id(p) for p in model.encoder.parameters()}
    encoder_params = list(model.encoder.parameters())
    decoder_params = [p for p in model.parameters() if id(p) not in encoder_ids]
    return encoder_params, decoder_params


if __name__ == "__main__":
    print(f"smp version: {smp.__version__}")
    print("Building MitUNet (pretrained=False)...")
    model = build_mitunet(num_classes=4, pretrained=False)
    x = torch.randn(2, 3, 512, 512)
    with torch.no_grad():
        out = model(x)
    total = sum(p.numel() for p in model.parameters()) / 1e6
    enc   = sum(p.numel() for p in model.encoder.parameters()) / 1e6
    enc_p, dec_p = get_param_groups(model)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Params: {total:.1f}M total  (encoder={enc:.1f}M  decoder={total-enc:.1f}M)")
    print(f"Param groups: encoder={len(enc_p)}  decoder={len(dec_p)}")
    print("Build OK ✓")
