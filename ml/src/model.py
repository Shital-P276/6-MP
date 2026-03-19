"""
model.py — MitUNet with MiT-B2 encoder transplant
smp 0.3.3 compatible — does NOT use smp.Segformer (added in later versions)
Instead loads MiT-B2 weights directly via timm, which smp 0.3.3 uses internally.
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


def build_mitunet(num_classes: int = 4, pretrained: bool = True) -> nn.Module:
    """
    Build MitUNet: MiT-B2 encoder + U-Net decoder with scSE attention.

    smp 0.3.3 fix: smp.Segformer does not exist in 0.3.3.
    We build smp.Unet with mit_b2 encoder directly — smp 0.3.3 loads
    MiT-B2 pretrained weights from timm internally when encoder_weights='imagenet'.
    No transplant needed. This is cleaner and more reliable.
    """
    encoder_weights = "imagenet" if pretrained else None

    model = smp.Unet(
        encoder_name="mit_b2",
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=num_classes,
        decoder_attention_type="scse",
    )

    # Verify with a dry run
    try:
        dummy = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            out = model(dummy)
        assert out.shape == (1, num_classes, 64, 64), \
            f"Expected (1,{num_classes},64,64), got {out.shape}"
    except Exception as e:
        raise RuntimeError(f"MitUNet build failed: {e}")

    return model


def get_param_groups(model: nn.Module):
    """
    Two parameter groups for differential LR.
    Encoder (pretrained) gets 10x lower LR than decoder.
    """
    encoder_ids    = {id(p) for p in model.encoder.parameters()}
    encoder_params = list(model.encoder.parameters())
    decoder_params = [p for p in model.parameters() if id(p) not in encoder_ids]
    return encoder_params, decoder_params


if __name__ == "__main__":
    print("Testing MitUNet build...")
    model = build_mitunet(num_classes=4, pretrained=False)
    x = torch.randn(2, 3, 512, 512)
    with torch.no_grad():
        out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    total = sum(p.numel() for p in model.parameters()) / 1e6
    enc   = sum(p.numel() for p in model.encoder.parameters()) / 1e6
    enc_p, dec_p = get_param_groups(model)
    print(f"Params: {total:.1f}M  (encoder={enc:.1f}M, decoder={total-enc:.1f}M)")
    print("Build OK ✓")
