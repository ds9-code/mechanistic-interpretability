import os
import yaml
import numpy as np
import torch
import pytorch_lightning as pl
import wandb
import sys
from monai.transforms import (
    Compose,
    LoadImaged,
    NormalizeIntensityd,
    RandAffined,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandAdjustContrastd,
    Resized,
    ToTensord,
)
from monai.utils import set_determinism
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from simclr.random_resized_crop import RandomResizedCrop3D
from dataset import NiftiSingleViewDataset, EmbeddingDataset
from model_mlp_grounded import SAEAdapterMLPGrounded


## no augmentation loading 
def build_train_transform(cfg):
    keys = ["image"]
    return Compose(
        [
            LoadImaged(keys=keys, ensure_channel_first=True),
            Resized(keys=keys, spatial_size=tuple(cfg["data"]["size"])),
            NormalizeIntensityd(keys=keys, nonzero=True, channel_wise=True),
            ToTensord(keys=keys),
        ]
    )


## in case we need augmentations 
"""def build_train_transform(cfg):
    keys = ["image"]
    randomresizecrop = RandomResizedCrop3D(keys=keys)
    return Compose(
        [
            LoadImaged(keys=keys, ensure_channel_first=True),
            Resized(keys=keys, spatial_size=tuple(cfg["data"]["size"])),
            NormalizeIntensityd(keys=keys, nonzero=True, channel_wise=True),
            RandAffined(
                keys=keys,
                rotate_range=(np.pi / 12, np.pi / 12, 0),
                translate_range=((-10, 10), (-10, 10), (-5, 5)),
                scale_range=((0.85, 1.15), (0.85, 1.15), (0.9, 1.1)),
                padding_mode="border",
                prob=0.7,
            ),
            RandFlipd(keys=keys, spatial_axis=[0], prob=0.5),
            randomresizecrop,
            RandGaussianSmoothd(keys=keys, prob=0.3, sigma_x=(0.5, 1.5), sigma_y=(0.5, 1.5), sigma_z=(0.5, 1.5)),
            RandGaussianNoised(keys=keys, prob=0.3, std=0.05),
            RandAdjustContrastd(keys=keys, prob=0.3, gamma=(0.7, 1.3)),
            ToTensord(keys=keys),
        ]
    )"""


if __name__ == "__main__":
    with open("config.yml", "r") as file:
        cfg = yaml.safe_load(file)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["trainer"]["devices"][0])
    wandb.init(project=cfg["logger"]["project"], name=cfg["logger"]["run_name"])
    wandb.config.update(cfg)

    # Ensure checkpoint directory exists and is writable before training
    save_dir = cfg["logger"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    if not os.access(save_dir, os.W_OK):
        raise RuntimeError(f"Checkpoint directory {save_dir} is not writable")

    wandb_logger = WandbLogger(log_model=True)
    set_determinism(seed=0)
    torch.manual_seed(0)
    np.random.seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    torch.set_float32_matmul_precision("medium")

    if cfg["data"].get("use_embeddings", False):
        train_dataset = EmbeddingDataset(
            embedding_csv=cfg["data"].get("embedding_csv"),
            embedding_npy=cfg["data"].get("embedding_npy"),
        )
    else:
        train_transform = build_train_transform(cfg)
        train_dataset = NiftiSingleViewDataset(
            csv_file=cfg["data"]["csv_file"],
            root_dir=cfg["data"]["root_dir"],
            transform=train_transform,
        )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )

    model_type = cfg["model"].get("adapter_type", "linear")
    if model_type == "mlp":
        model = SAEAdapterMLP(cfg)
    elif model_type == "mlp_grounded":
        model = SAEAdapterMLPGrounded(cfg)
    else:
        model = SAEAdapter(cfg)

    checkpoint_callback = ModelCheckpoint(
        monitor="train_loss",
        mode="min",
        save_top_k=5,
        save_last=True,
        dirpath=cfg["logger"]["save_dir"],
        filename=cfg["logger"]["save_name"],
    )

    trainer = pl.Trainer(
        max_epochs=cfg["trainer"]["max_epochs"],
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        devices=cfg["trainer"]["devices"],
        precision=cfg["trainer"]["precision"],
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        gradient_clip_val=1.0,
    )

    trainer.fit(model, train_loader)
