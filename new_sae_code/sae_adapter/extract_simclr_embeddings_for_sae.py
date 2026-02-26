import argparse
import os
import sys

import pandas as pd
import torch
import yaml
from tqdm import tqdm
from argparse import Namespace
from monai.transforms import Compose, LoadImaged, NormalizeIntensityd, Resized, ToTensord

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset import NiftiSingleViewDataset
from simclr.model import SimCLRModel
from model import SAEAdapter
from model_mlp import SAEAdapterMLP
from monai.data.meta_tensor import MetaTensor
from numpy.core.multiarray import _reconstruct
from numpy import ndarray, dtype


class MCIStrokeDataset(torch.utils.data.Dataset):
    """Dataset class for MCI and Stroke tasks."""

    def __init__(self, csv_path, root_dir, transform=None):
        self.dataframe = pd.read_csv(csv_path, dtype={"pat_id": str, "dataset": str})
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        pat_id = str(self.dataframe.loc[idx, "pat_id"])
        dataset = str(self.dataframe.loc[idx, "dataset"])
        if not pat_id.endswith(".nii.gz"):
            pat_id = f"{pat_id}.nii.gz"
        img_path = os.path.join(self.root_dir, f"{dataset}/data", pat_id)
        sample = {"image": img_path}
        if self.transform:
            sample = self.transform(sample)
        return sample


class IDHFlairDataset(torch.utils.data.Dataset):
    """Dataset class for IDH with FLAIR modality only."""

    def __init__(self, csv_path, root_dir, transform=None):
        self.dataframe = pd.read_csv(csv_path, dtype={"pat_id": str, "dataset": str})
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        pat_id = str(self.dataframe.loc[idx, "pat_id"])
        dataset = str(self.dataframe.loc[idx, "dataset"])
        img_path = os.path.join(self.root_dir, dataset, "FLAIR", f"{pat_id}_FLAIR.nii.gz")
        sample = {"image": img_path}
        if self.transform:
            sample = self.transform(sample)
        return sample


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


def load_sae_model(cfg, device):
    sae_ckpt = cfg["inference"]["sae_checkpoint"]
    torch.serialization.add_safe_globals([MetaTensor, _reconstruct, ndarray, dtype])
    with torch.serialization.safe_globals([MetaTensor, _reconstruct, ndarray, dtype]):
        checkpoint = torch.load(sae_ckpt, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

    model_type = cfg["model"].get("adapter_type", "linear")
    if model_type == "mlp":
        model = SAEAdapterMLP(cfg)
    else:
        model = SAEAdapter(cfg)

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Extract SimCLR embeddings for SAE training.")
    parser.add_argument("--config", default="config.yml", help="Path to sae_adapter config.")
    args = parser.parse_args()

    with open(args.config, "r") as file:
        cfg = yaml.safe_load(file)

    csv_file = cfg["data"].get("embedding_source_csv") or cfg["data"]["csv_file"]
    embedding_model = cfg["data"].get("embedding_model", "simclr")
    simclr_ckpt = cfg["backbone"]["checkpoint"]
    out_csv = cfg["data"]["embedding_csv"]
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    device = torch.device(cfg["data"].get("embedding_device", "cuda:0"))
    transform = build_eval_transform(cfg)
    dataset_type = cfg["data"].get("dataset_type", "").lower()
    if dataset_type == "stroke":
        dataset = MCIStrokeDataset(
            csv_path=csv_file,
            root_dir=cfg["data"]["root_dir"],
            transform=transform,
        )
    elif dataset_type == "idh":
        dataset = IDHFlairDataset(
            csv_path=csv_file,
            root_dir=cfg["data"]["root_dir"],
            transform=transform,
        )
    else:
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

    if embedding_model == "sae":
        model = load_sae_model(cfg, device)
    else:
        backbone = load_simclr_backbone(simclr_ckpt, cfg, device)

    features = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting SimCLR embeddings"):
            images = batch["image"].to(device)
            if embedding_model == "sae":
                x = model.compute_embedding(images)
                z_raw = model.encoder(x)
                z, _ = model.apply_sparsity(z_raw)
                wz = model.adapter(z)
                y = x + wz
                features.append(y.cpu())
            else:
                outputs = backbone(images)
                x = outputs[0][:, 0, :]
                features.append(x.cpu())

    features = torch.cat(features, dim=0).numpy()
    feature_cols = [f"f{i}" for i in range(features.shape[1])]
    df = pd.DataFrame(features, columns=feature_cols)
    src_df = pd.read_csv(csv_file)
    id_cols = src_df.select_dtypes(exclude=["number"]).columns.tolist()
    label_col = cfg["data"].get("embedding_label_col", "label")
    keep_cols = []
    if id_cols:
        keep_cols.extend(id_cols)
    if label_col in src_df.columns:
        keep_cols.append(label_col)
    if keep_cols:
        df = pd.concat([src_df[keep_cols].reset_index(drop=True), df], axis=1)
    df.to_csv(out_csv, index=False)


if __name__ == "__main__":
    main()
