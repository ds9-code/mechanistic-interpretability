"""
Activation-level intervention analysis for top NEW SAE features.
For each feature (e.g. 857, 17827, 1796), set that feature to different levels
(negative, zero, median, max, above_max, baseline) for all test samples, decode,
run probe, and plot predicted brain age vs activation level (scatter + box).
Outputs: activation_levels table (CSV), scatter/box figure (PNG).
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
from monai.data.meta_tensor import MetaTensor
from numpy.core.multiarray import _reconstruct
from numpy import ndarray, dtype

MECHINTERP = Path(__file__).resolve().parents[2]
BRAINIAC_SRC = Path(__file__).resolve().parent
NEW_SAE_ADAPTER = MECHINTERP / "new_sae_code" / "sae_adapter"
SOPHONT = Path("/media/data/divyanshu/sophont")
for p in [SOPHONT, str(NEW_SAE_ADAPTER)]:
    if p not in sys.path:
        sys.path.insert(0, p)
sys.path.insert(0, str(BRAINIAC_SRC))

from dataset import BrainAgeDataset, get_validation_transform
from load_simclr import load_simclr_backbone, simclr_forward
from model_mlp_grounded import SAEAdapterMLPGrounded


def load_new_sae(config_path, checkpoint_path, device, use_embeddings=True):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    if "data" not in cfg:
        cfg["data"] = {}
    cfg["data"]["use_embeddings"] = use_embeddings
    torch.serialization.add_safe_globals([MetaTensor, _reconstruct, ndarray, dtype])
    with torch.serialization.safe_globals([MetaTensor, _reconstruct, ndarray, dtype]):
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    model = SAEAdapterMLPGrounded(cfg)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def load_probe(path, device, input_dim=768):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    new_state = {}
    for k, v in state.items():
        if k.startswith("model."):
            k = k[6:]
        if k.startswith("linear_probe."):
            k = k[13:]
        new_state[k] = v
    if "linear.weight" in new_state and "weight" not in new_state:
        new_state["weight"] = new_state.pop("linear.weight")
    if "linear.bias" in new_state and "bias" not in new_state:
        new_state["bias"] = new_state.pop("linear.bias")
    probe = nn.Linear(input_dim, 1).to(device)
    probe.load_state_dict(new_state, strict=False)
    probe.eval()
    return probe


@torch.no_grad()
def extract_activations_and_labels(backbone, new_sae, loader, device):
    all_z, all_label = [], []
    for batch in tqdm(loader, desc="Extracting activations"):
        x = batch["image"].to(device)
        emb = simclr_forward(backbone, x)
        z_raw = new_sae.encoder(emb)
        z, _ = new_sae.apply_sparsity(z_raw)
        all_z.append(z.cpu())
        all_label.append(batch["label"])
    return torch.cat(all_z), torch.cat(all_label).float()


@torch.no_grad()
def predict_ages_for_activations(new_sae, probe, activations, device, no_normalize=True):
    """activations [N, L]. Decode, optionally normalize, probe -> [N] predicted ages."""
    act = activations.to(device)
    decoded = new_sae.decoder(act)
    if no_normalize:
        pred = probe(decoded).squeeze(-1).cpu().numpy()
    else:
        mean = torch.zeros(1, decoded.size(1), device=device)
        std = torch.ones(1, decoded.size(1), device=device)
        decoded_norm = (decoded - mean) / (std + 1e-8)
        pred = probe(decoded_norm).squeeze(-1).cpu().numpy()
    return pred


# Activation level names and colors (match template)
LEVELS = ["negative", "zero", "median", "max", "above_max", "baseline"]
# Custom levels for feature 1796: negative -> median -> zero -> max -> above_max -> baseline (at right edge)
LEVELS_CUSTOM_1796 = ["negative", "median", "zero", "max", "above_max", "baseline"]
COLORS = {
    "negative": "#C45B2C",      # orange/brown
    "zero": "#6BA3D8",         # light blue
    "median": "#5A9E4C",       # green
    "max": "#CC3311",          # red/dark orange
    "above_max": "#9966CC",    # purple
    "baseline": "#999999",     # light grey
    "one": "#E6B800",          # gold (for custom "1" level)
}


def main():
    parser = argparse.ArgumentParser(description="Activation-level intervention + box/scatter plots")
    parser.add_argument("--simclr_checkpoint", type=str,
                        default="/media/data/divyanshu/sophont/checkpoints/brainiac_traincsv_simclr_norm_vit_cls_vitb_tejas_lr0005_best-model-epoch=11-train_loss=0.00.ckpt")
    parser.add_argument("--new_sae_config", type=str, default=None)
    parser.add_argument("--new_sae_checkpoint", type=str, required=True)
    parser.add_argument("--linear_probe", type=str, required=True)
    parser.add_argument("--test_csv", type=str, default="data/csvs/development_test_set.csv")
    parser.add_argument("--root_dir", type=str, default="data/images/data")
    parser.add_argument("--feature_indices", type=str, default="857,17827,1796",
                        help="Comma-separated feature indices")
    parser.add_argument("--output_dir", type=str, default="linear_probe_results_new_sae_age")
    parser.add_argument("--output_prefix", type=str, default="activation_levels")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no_normalize", action="store_true", default=True)
    parser.add_argument("--isolated", action="store_true",
                        help="Zero out all other features so only the intervened feature varies. Gives clearly separated bands (like the previous experiment).")
    parser.add_argument("--custom_levels_feature", type=int, default=None,
                        help="If set, run only this feature with custom levels: negative (-2), baseline (actual), zero (0), one (1), above_max. E.g. 1796.")
    args = parser.parse_args()

    if args.simclr_checkpoint.endswith('"'):
        args.simclr_checkpoint = args.simclr_checkpoint.rstrip('"')

    project = MECHINTERP
    config_path = Path(args.new_sae_config) if args.new_sae_config else (project / "sae_training_new_sae" / "config_brainage.yml")
    if not config_path.exists():
        config_path = project / "new_sae_code" / "sae_adapter" / "config.yml"
    ckpt_path = project / args.new_sae_checkpoint if not str(args.new_sae_checkpoint).startswith("/") else Path(args.new_sae_checkpoint)
    probe_path = project / args.linear_probe if not str(args.linear_probe).startswith("/") else Path(args.linear_probe)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.custom_levels_feature is not None:
        feature_indices = [args.custom_levels_feature]
    else:
        feature_indices = [int(x.strip()) for x in args.feature_indices.split(",")]
    # Feature 1796 always uses custom levels (negative, baseline, zero, one, above_max); others use default levels.
    FEATURE_CUSTOM_LEVELS = 1796
    out_dir = project / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading SimCLR backbone...")
    backbone = load_simclr_backbone(args.simclr_checkpoint, device)
    print("Loading NEW SAE...")
    new_sae = load_new_sae(str(config_path), str(ckpt_path), device, use_embeddings=True)
    print("Loading linear probe...")
    probe = load_probe(str(probe_path), device)

    dataset = BrainAgeDataset(
        csv_path=str(project / args.test_csv),
        root_dir=str(project / args.root_dir),
        transform=get_validation_transform((96, 96, 96)),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=(device.type == "cuda"))

    print("Extracting test activations...")
    z, labels = extract_activations_and_labels(backbone, new_sae, loader, device)
    z = z.numpy()
    labels = labels.numpy()
    n_samples = z.shape[0]

    # Same adjustment levels for all features: negative=-2, zero=0; per-feature: median, max, above_max
    NEGATIVE_VAL = -2.0
    ZERO_VAL = 0.0

    level_table_rows = []
    results = {}  # feature_idx -> { level_name -> pred_ages (array) }
    levels_per_feature = {}  # feature_idx -> list of level names (for plotting)

    for feat_idx in feature_indices:
        col = z[:, feat_idx]
        median_val = float(np.median(col))
        max_val = float(np.max(col))
        above_max_val = max_val + (max_val - median_val) / 2.0 if (max_val - median_val) != 0 else max_val * 1.5

        use_custom_1796 = feat_idx == FEATURE_CUSTOM_LEVELS
        levels_this_feature = LEVELS_CUSTOM_1796 if use_custom_1796 else LEVELS
        levels_per_feature[feat_idx] = levels_this_feature

        if use_custom_1796:
            level_table_rows.append({
                "Feature": feat_idx,
                "negative": NEGATIVE_VAL,
                "median": round(median_val, 6),
                "zero": ZERO_VAL,
                "max": round(max_val, 6),
                "above_max [(max-median)/2]": round(above_max_val, 6),
                "baseline": "actual",
            })
            level_values = {
                "negative": NEGATIVE_VAL,
                "median": median_val,
                "zero": ZERO_VAL,
                "max": max_val,
                "above_max": above_max_val,
            }
        else:
            level_table_rows.append({
                "Feature": feat_idx,
                "negative": NEGATIVE_VAL,
                "zero": ZERO_VAL,
                "median": round(median_val, 6),
                "max": round(max_val, 6),
                "above_max [(max-median)/2]": round(above_max_val, 6),
            })
            level_values = {
                "negative": NEGATIVE_VAL,
                "zero": ZERO_VAL,
                "median": median_val,
                "max": max_val,
                "above_max": above_max_val,
            }
        results[feat_idx] = {}

        for level_name in levels_this_feature:
            if level_name == "baseline":
                z_use = z.copy()
            else:
                if args.isolated:
                    # Zero out all features, then set only this feature to the level.
                    # Decoder output then depends only on this feature -> clear separation between levels.
                    z_use = np.zeros_like(z)
                    z_use[:, feat_idx] = level_values[level_name]
                else:
                    z_use = z.copy()
                    z_use[:, feat_idx] = level_values[level_name]
            z_t = torch.from_numpy(z_use).float()
            pred = predict_ages_for_activations(new_sae, probe, z_t, device, no_normalize=args.no_normalize)
            results[feat_idx][level_name] = pred

    # Save activation levels table
    df_table = pd.DataFrame(level_table_rows)
    table_path = out_dir / f"{args.output_prefix}_table.csv"
    df_table.to_csv(table_path, index=False)
    print(f"Saved {table_path}")
    print(df_table.to_string(index=False))

    # Scatter + box plots (one subplot per feature: top row scatter, bottom row box)
    n_feat = len(feature_indices)
    fig, axes = plt.subplots(2, n_feat, figsize=(5 * n_feat, 10))
    if n_feat == 1:
        axes = axes.reshape(2, 1)

    for i, feat_idx in enumerate(feature_indices):
        levs = levels_per_feature[feat_idx]
        # Scatter: x = image index, y = predicted brain age, color = level
        ax_scatter = axes[0, i]
        for level_name in levs:
            if level_name == "baseline":
                continue  # hide baseline dots in scatter
            pred = results[feat_idx][level_name]
            ax_scatter.scatter(np.arange(n_samples), pred, c=COLORS[level_name], label=level_name, alpha=0.5, s=8)
        ax_scatter.set_xlabel("Image index")
        ax_scatter.set_ylabel("Predicted brain age (months)")
        ax_scatter.set_title(f"Feature {feat_idx}")
        ax_scatter.legend(loc="upper right", fontsize=8)
        ax_scatter.set_ylim(0, max(500, pred.max() * 1.05))

        # Box: x = level, y = predicted brain age
        ax_box = axes[1, i]
        data_for_box = [results[feat_idx][lev] for lev in levs]
        bp = ax_box.boxplot(data_for_box, tick_labels=levs, patch_artist=True, showfliers=False)
        for j, lev in enumerate(levs):
            bp["boxes"][j].set_facecolor(COLORS[lev])
        ax_box.set_xlabel("Activation level")
        ax_box.set_ylabel("Predicted brain age (months)")
        ax_box.set_title(f"Feature {feat_idx}")
        ax_box.set_ylim(0, max(500, np.concatenate(data_for_box).max() * 1.05))

    plt.tight_layout()
    fig_path = out_dir / f"{args.output_prefix}_scatter_box.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {fig_path}")

    print("Done.")


if __name__ == "__main__":
    main()
