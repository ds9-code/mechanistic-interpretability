import os
from argparse import Namespace

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import ViT

from simclr.model import SimCLRModel


def build_vit_backbone(cfg):
    return ViT(
        in_channels=cfg["backbone"]["in_channels"],
        img_size=tuple(cfg["backbone"]["img_size"]),
        patch_size=tuple(cfg["backbone"]["patch_size"]),
        hidden_size=cfg["backbone"]["hidden_size"],
        mlp_dim=cfg["backbone"]["mlp_dim"],
        num_layers=cfg["backbone"]["num_layers"],
        num_heads=cfg["backbone"]["num_heads"],
        save_attn=True,
    )


def load_backbone(cfg):
    checkpoint_path = cfg["backbone"].get("checkpoint")
    if checkpoint_path and os.path.exists(checkpoint_path):
        hparams = Namespace(
            max_epochs=1,
            lr=cfg["optim"]["lr"],
            weight_decay=cfg["optim"]["weight_decay"],
            momentum=0.9,
            temperature=0.1,
        )
        simclr_model = SimCLRModel.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            hparams=hparams,
        )
        backbone = simclr_model.backbone
    else:
        backbone = build_vit_backbone(cfg)

    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False
    return backbone


class SAEAdapterMLPGrounded(pl.LightningModule):
    """
    Simple sparse autoencoder:
      - Encoder: embedding_dim -> latent_dim
      - Decoder: latent_dim -> embedding_dim
      - Sparsity: top-k or L1 on latent z
      - Objective: reconstruction + sparsity (no extra adapter/grounding terms)
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.use_embeddings = bool(cfg.get("data", {}).get("use_embeddings", False))
        self.backbone = None if self.use_embeddings else load_backbone(cfg)
        self.embedding_dim = cfg["backbone"]["hidden_size"]
        self.latent_dim = cfg["model"]["latent_dim"]
        self.sparsity_type = cfg["model"]["sparsity_type"]
        self.topk = cfg["model"]["topk"]
        self.l1_weight = cfg["model"]["l1_weight"]

        self.encoder = nn.Linear(self.embedding_dim, self.latent_dim)
        self.decoder = nn.Linear(self.latent_dim, self.embedding_dim)

    def compute_embedding(self, images):
        if self.backbone is None:
            raise RuntimeError("Backbone is disabled. Provide embeddings in the batch.")
        with torch.no_grad():
            features = self.backbone(images)
            embedding = features[0][:, 0, :]
        return embedding

    def apply_sparsity(self, z_raw: torch.Tensor):
        if self.sparsity_type == "topk":
            k = min(self.topk, z_raw.size(1))
            idx = z_raw.abs().topk(k=k, dim=1, largest=True, sorted=False).indices
            mask = torch.zeros_like(z_raw, dtype=torch.bool)
            mask.scatter_(1, idx, True)
            z = z_raw.masked_fill(~mask, 0.0)
            l1_loss = z_raw.new_zeros(())
        elif self.sparsity_type == "l1":
            z = z_raw
            l1_loss = z_raw.abs().sum(dim=1).mean()
        return z, l1_loss

    def training_step(self, batch, batch_idx):
        if "embedding" in batch:
            x = batch["embedding"]
        else:
            images = batch["image"]
            x = self.compute_embedding(images)

        z_raw = self.encoder(x)
        z, l1_loss = self.apply_sparsity(z_raw)
        x_recon = self.decoder(z)

        rec_loss = F.mse_loss(x_recon, x)
        total_loss = rec_loss
        if self.sparsity_type == "l1":
            total_loss = total_loss + self.l1_weight * l1_loss

        # Diagnostics similar to the OpenMidnight SAE:
        # - fraction of non-zero activations
        # - mean absolute value of active units
        # - fraction of latents used in this batch (top-k case)
        with torch.no_grad():
            nnz_frac = (z != 0).float().mean()
            active_counts = (z != 0).sum(dim=1).clamp_min(1)
            z_active_abs_mean = z.abs().sum(dim=1).div(active_counts).mean()

            if self.sparsity_type == "topk":
                k = min(self.topk, z_raw.size(1))
                idx = z_raw.abs().topk(k=k, dim=1, largest=True, sorted=False).indices
                unique_used = torch.unique(idx).numel()
                latents_used_frac = unique_used / float(self.latent_dim)
            else:
                latents_used_frac = torch.tensor(0.0, device=self.device)

        self.log("train_loss", total_loss, prog_bar=True)
        self.log("rec_loss", rec_loss, prog_bar=True)
        self.log("z_nnz_frac", nnz_frac)
        self.log("z_active_abs_mean", z_active_abs_mean)
        self.log("latents_used_frac_batch", latents_used_frac)
        if self.sparsity_type == "l1":
            self.log("l1_loss", l1_loss)
        return total_loss

    def configure_optimizers(self):
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.cfg["optim"]["lr"],
            weight_decay=self.cfg["optim"]["weight_decay"],
        )
        return optimizer
