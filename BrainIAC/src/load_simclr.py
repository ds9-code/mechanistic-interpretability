"""
Load SimCLR backbone for use with NEW SAE.
The SimCLR checkpoint was trained with an older MONAI ViT. Current MONAI ViT has cross_attn
layers that the checkpoint doesn't have. We load with strict=False to allow those keys to remain init.
"""

import torch
import torch.nn as nn
from monai.networks.nets import ViT


def load_simclr_backbone(checkpoint_path, device="cuda"):
    """
    Load SimCLR ViT backbone from checkpoint.
    Uses strict=False because MONAI ViT now has cross_attn layers not in the checkpoint.
    Returns backbone that outputs (hidden_states, ...) with CLS token at [0][:, 0, :].
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)

    # Extract backbone weights (keys like "backbone.xxx" -> "xxx")
    backbone_state = {}
    for k, v in state_dict.items():
        if k.startswith("backbone."):
            backbone_state[k[9:]] = v

    backbone = ViT(
        in_channels=1,
        img_size=(96, 96, 96),
        patch_size=(16, 16, 16),
        hidden_size=768,
        mlp_dim=3072,
        num_layers=12,
        num_heads=12,
        save_attn=True,
    )
    missing, unexpected = backbone.load_state_dict(backbone_state, strict=False)
    if missing:
        print(f"  (SimCLR backbone: {len(missing)} keys not in checkpoint, e.g. cross_attn - left as init)")
    if unexpected:
        print(f"  (Checkpoint: {len(unexpected)} keys not loaded)")
    backbone = backbone.to(device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False
    return backbone


def simclr_forward(backbone, x):
    """Get 768-d CLS token from backbone. x: [B, 1, 96, 96, 96]."""
    with torch.no_grad():
        features = backbone(x)
        cls_token = features[0][:, 0, :]  # [B, 768]
    return cls_token
