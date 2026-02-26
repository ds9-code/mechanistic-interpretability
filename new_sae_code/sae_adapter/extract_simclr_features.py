import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch
import yaml
from argparse import Namespace
from monai.transforms import Compose, LoadImaged, NormalizeIntensityd, Resized, ToTensord
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from dataset import NiftiSingleViewDataset
from simclr.model import SimCLRModel


def build_eval_transform(cfg):
    keys = ["image"]
    return Compose(
        [
            LoadImaged(keys=keys, ensure_channel_first=True),
            Resized(keys=keys, spatial_size=tuple(cfg["data"]["size"])),
            NormalizeIntensityd(keys=keys, nonzero=True, channel_wise=True),
            ToTensord(keys=keys),
        ]
    )


def load_simclr_backbone(checkpoint_path, cfg, device):
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
    backbone = simclr_model.backbone.to(device)
    backbone.eval()
    return backbone


def main():
    parser = argparse.ArgumentParser(description="Extract SimCLR CLS features.")
    parser.add_argument("--config", default="config.yml", help="Path to sae_adapter config.")
    parser.add_argument("--csv", default=None, help="CSV file to extract features from.")
    parser.add_argument("--simclr_ckpt", default=None, help="SimCLR checkpoint path.")
    parser.add_argument("--out_dir", default=None, help="Output directory for features.")
    args = parser.parse_args()

    with open(args.config, "r") as file:
        cfg = yaml.safe_load(file)

    csv_file = args.csv or cfg["inference"]["val_csv"] or cfg["data"]["csv_file"]
    simclr_ckpt = args.simclr_ckpt or cfg["inference"]["simclr_checkpoint"] or cfg["backbone"]["checkpoint"]
    out_dir = args.out_dir or cfg["inference"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = build_eval_transform(cfg)
    dataset = NiftiSingleViewDataset(
        csv_file=csv_file,
        root_dir=cfg["data"]["root_dir"],
        transform=transform,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )

    backbone = load_simclr_backbone(simclr_ckpt, cfg, device)

    features = []
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            outputs = backbone(images)
            x = outputs[0][:, 0, :]
            features.append(x.cpu())

    features = torch.cat(features, dim=0).numpy()
    feature_cols = [f"f{i}" for i in range(features.shape[1])]
    df = pd.DataFrame(features, columns=feature_cols)
    df.to_csv(os.path.join(out_dir, cfg["inference"]["simclr_file"]), index=False)
    with open(os.path.join(out_dir, "source_csv.txt"), "w") as f:
        f.write(csv_file)


if __name__ == "__main__":
    main()
