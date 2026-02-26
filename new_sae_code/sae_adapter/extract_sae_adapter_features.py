import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm
from monai.data.meta_tensor import MetaTensor
from numpy.core.multiarray import _reconstruct
from numpy import ndarray, dtype
from monai.transforms import Compose, LoadImaged, NormalizeIntensityd, Resized, ToTensord
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from dataset import NiftiSingleViewDataset, EmbeddingDataset
from model import SAEAdapter
from model_mlp import SAEAdapterMLP
from model_mlp_grounded import SAEAdapterMLPGrounded


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


def main():
    parser = argparse.ArgumentParser(description="Extract SAE-adapter features (y = x + Wz).")
    parser.add_argument("--config", default="config.yml", help="Path to sae_adapter config.")
    parser.add_argument("--csv", default=None, help="CSV file to extract features from.")
    parser.add_argument("--sae_ckpt", default=None, help="SAE adapter checkpoint path.")
    parser.add_argument("--out_dir", default=None, help="Output directory for features.")
    parser.add_argument("--embedding_csv", default=None, help="Optional embedding CSV to use instead of images.")
    args = parser.parse_args()

    with open(args.config, "r") as file:
        cfg = yaml.safe_load(file)

    csv_file = args.csv or cfg["inference"]["val_csv"] or cfg["data"]["csv_file"]
    sae_ckpt = args.sae_ckpt or cfg["inference"]["sae_checkpoint"]
    out_dir = args.out_dir or cfg["inference"]["out_dir"]
    embedding_csv = args.embedding_csv or cfg["inference"].get("embedding_csv")
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cpu")
    if embedding_csv:
        cfg = dict(cfg)
        cfg["data"] = dict(cfg["data"])
        cfg["data"]["use_embeddings"] = True
        dataset = EmbeddingDataset(embedding_csv=embedding_csv)
    else:
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

    torch.serialization.add_safe_globals([MetaTensor, _reconstruct, ndarray, dtype])
    with torch.serialization.safe_globals([MetaTensor, _reconstruct, ndarray, dtype]):
        checkpoint = torch.load(sae_ckpt, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    model_type = cfg["model"].get("adapter_type", "linear")
    if model_type == "mlp":
        model = SAEAdapterMLP(cfg)
    elif model_type == "mlp_grounded":
        model = SAEAdapterMLPGrounded(cfg)
    else:
        model = SAEAdapter(cfg)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    features = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting SAE features"):
            if "embedding" in batch:
                x = batch["embedding"].to(device)
            else:
                images = batch["image"].to(device)
                x = model.compute_embedding(images)
            z_raw = model.encoder(x)
            z, _ = model.apply_sparsity(z_raw)
            if model_type == "mlp_grounded":
                x_recon = model.decoder(z)
                features.append(x_recon.cpu())
            else:
                wz = model.adapter(z)
                features.append(wz.cpu())

    features = torch.cat(features, dim=0).numpy()
    feature_cols = [f"f{i}" for i in range(features.shape[1])]
    df = pd.DataFrame(features, columns=feature_cols)
    df.to_csv(os.path.join(out_dir, cfg["inference"]["sae_file"]), index=False)
    with open(os.path.join(out_dir, "source_csv.txt"), "w") as f:
        f.write(csv_file)


if __name__ == "__main__":
    main()
