import argparse
import os
import sys

import pandas as pd
import torch
import yaml
from monai.data.meta_tensor import MetaTensor
from numpy.core.multiarray import _reconstruct
from numpy import ndarray, dtype

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset import EmbeddingDataset
from model import SAEAdapter
from model_mlp import SAEAdapterMLP
from model_mlp_grounded import SAEAdapterMLPGrounded


def main():
    parser = argparse.ArgumentParser(description="Extract SAE-adapter embeddings y = x + Wz from precomputed x.")
    parser.add_argument("--config", default="config.yml", help="Path to sae_adapter config.")
    args = parser.parse_args()

    with open(args.config, "r") as file:
        cfg = yaml.safe_load(file)

    embedding_csv = cfg["inference"]["embedding_csv"]
    sae_ckpt = cfg["inference"]["sae_checkpoint"]
    out_dir = cfg["inference"]["out_dir"]
    out_file = cfg["inference"]["sae_file"]
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cpu")
    dataset = EmbeddingDataset(embedding_csv=embedding_csv)
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
        for batch in dataloader:
            x = batch["embedding"].to(device)
            z_raw = model.encoder(x)
            z, _ = model.apply_sparsity(z_raw)
            if model_type == "mlp_grounded":
                y = model.decoder(z)
            else:
                wz = model.adapter(z)
                y = x + wz
            features.append(y.cpu())

    features = torch.cat(features, dim=0).numpy()
    feature_cols = [f"f{i}" for i in range(features.shape[1])]
    df = pd.DataFrame(features, columns=feature_cols)
    src_df = pd.read_csv(embedding_csv)
    meta_cols = [c for c in src_df.columns if not c.startswith("f")]
    if meta_cols:
        df = pd.concat([src_df[meta_cols].reset_index(drop=True), df], axis=1)
    df.to_csv(os.path.join(out_dir, out_file), index=False)


if __name__ == "__main__":
    main()
