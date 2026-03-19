"""
model.py — MitUNet with MiT-B2 encoder transplant
Architecture: SegFormer MiT-B2 encoder → U-Net decoder with scSE attention
Paper: Parashchuk et al. (2024) — 87.84% mIoU on CubiCasa5k
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


def build_mitunet(num_classes: int = 4, pretrained: bool = True) -> nn.Module:
    """
    Build MitUNet by transplanting MiT-B2 encoder from SegFormer into U-Net body.

    BUG FIX: smp.Segformer uses a different internal encoder wrapper than smp.Unet.
    We must verify the transplanted encoder produces the correct output channels
    [64, 128, 320, 512] before returning.

    Args:
        num_classes: 4 = background, wall, door, window
        pretrained:  load ImageNet weights for encoder (True for training from scratch)
    """
    encoder_weights = "imagenet" if pretrained else None

    # Step 1 — build SegFormer to extract its pretrained MiT-B2 encoder
    aux_segformer = smp.Segformer(
        encoder_name="mit_b2",
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=num_classes,
    )

    # Step 2 — build U-Net shell (encoder_weights=None, we transplant below)
    model = smp.Unet(
        encoder_name="mit_b2",
        encoder_weights=None,
        in_channels=3,
        classes=num_classes,
        decoder_attention_type="scse",
    )

    # Step 3 — transplant the pretrained encoder
    model.encoder = aux_segformer.encoder

    # Step 4 — verify transplant with a dry run (catches smp version mismatches early)
    try:
        dummy = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            out = model(dummy)
        assert out.shape == (1, num_classes, 64, 64), \
            f"Expected output (1,{num_classes},64,64), got {out.shape}"
    except Exception as e:
        raise RuntimeError(
            f"MitUNet transplant verification failed: {e}\n"
            "Pin version: pip install segmentation-models-pytorch==0.3.3"
        )

    return model


def get_param_groups(model: nn.Module):
    """
    Return two parameter groups for differential learning rate.
    Encoder (pretrained) gets 10x lower LR than decoder (random init).

    BUG FIX: include segmentation_head in decoder group — previous version
    only captured model.decoder, missing the final conv head parameters.
    """
    encoder_ids    = {id(p) for p in model.encoder.parameters()}
    encoder_params = list(model.encoder.parameters())
    # Everything NOT in encoder: decoder + segmentation_head + any classification_head
    decoder_params = [p for p in model.parameters() if id(p) not in encoder_ids]
    return encoder_params, decoder_params


if __name__ == "__main__":
    print("Testing MitUNet build...")
    model = build_mitunet(num_classes=4, pretrained=False)
    x = torch.randn(2, 3, 512, 512)
    with torch.no_grad():
        out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")   # (2, 4, 512, 512)
    total = sum(p.numel() for p in model.parameters()) / 1e6
    enc   = sum(p.numel() for p in model.encoder.parameters()) / 1e6
    enc_p, dec_p = get_param_groups(model)
    print(f"Params: {total:.1f}M total  (encoder={enc:.1f}M, decoder={total-enc:.1f}M)")
    print(f"Param groups: encoder={len(enc_p)}, decoder={len(dec_p)}")
    print("Build OK ✓")
