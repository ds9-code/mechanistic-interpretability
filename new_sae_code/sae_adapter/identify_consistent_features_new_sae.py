"""
Extract activations from NEW SAE (SimCLR -> encoder -> top-k -> z) and identify
consistent / highly activated features. Saves consistent_features.json for use with
mask_and_mae_change_new_sae.py.

Criteria:
- Consistent: activation_rate >= min_activation_rate (feature is non-zero in that fraction of samples).
- With top-k=64 and 49152 latents, random rate ~0.13%. We use default min_activation_rate=0.01 (1%).
- Optionally also save top N by mean_activation when active (highly activated).
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
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


@torch.no_grad()
def extract_activations(backbone, new_sae, loader, device):
    all_z = []
    all_pat_id = []
    for batch in tqdm(loader, desc="Extracting activations"):
        x = batch["image"].to(device)
        emb = simclr_forward(backbone, x)
        z_raw = new_sae.encoder(emb)
        z, _ = new_sae.apply_sparsity(z_raw)
        all_z.append(z.cpu())
        pat_ids = batch.get("pat_id", [])
        if isinstance(pat_ids, torch.Tensor):
            pat_ids = pat_ids.cpu().tolist()
        all_pat_id.extend(pat_ids if isinstance(pat_ids, list) else [str(p) for p in pat_ids])
    z = torch.cat(all_z, dim=0).numpy()
    return z, all_pat_id


def compute_feature_stats(activations):
    """activations: [N, latent_dim]. Top-k so most entries 0."""
    n_samples, n_features = activations.shape
    stats = []
    for j in tqdm(range(n_features), desc="Feature stats"):
        col = activations[:, j]
        active = col != 0
        rate = float(np.mean(active))
        when_active = col[active]
        mean_active = float(np.mean(when_active)) if when_active.size else 0.0
        max_active = float(np.max(when_active)) if when_active.size else 0.0
        std_active = float(np.std(when_active)) if when_active.size else 0.0
        n_active = int(np.sum(active))
        stats.append({
            "feature_idx": j,
            "activation_rate": rate,
            "num_activated_samples": n_active,
            "mean_activation_when_active": mean_active,
            "max_activation": max_active,
            "std_activation_when_active": std_active,
        })
    return stats


def identify_consistent(stats, min_activation_rate=0.01, exclude_always_on=True, always_on_threshold=0.99):
    """
    Features that fire in at least min_activation_rate fraction of samples.
    If exclude_always_on: drop features with activation_rate >= always_on_threshold (suspicious collapse).
    """
    consistent = {}
    always_on = {}
    for s in stats:
        if s["activation_rate"] < min_activation_rate:
            continue
        idx = str(s["feature_idx"])
        entry = {
            "num_activated_images": s["num_activated_samples"],
            "activation_rate": s["activation_rate"],
            "mean_activation": s["mean_activation_when_active"],
            "max_activation": s["max_activation"],
            "std_activation_when_active": s["std_activation_when_active"],
        }
        if exclude_always_on and s["activation_rate"] >= always_on_threshold:
            always_on[idx] = entry
            continue
        consistent[idx] = entry
    return consistent, always_on


def identify_highly_activated(stats, top_n=200, exclude_always_on=True, always_on_threshold=0.99):
    """Top N features by mean activation when active (min 10 samples). Exclude always-on if requested."""
    qualified = [s for s in stats if s["num_activated_samples"] >= 10]
    if exclude_always_on:
        qualified = [s for s in qualified if s["activation_rate"] < always_on_threshold]
    qualified.sort(key=lambda s: s["mean_activation_when_active"], reverse=True)
    highly = {}
    for s in qualified[:top_n]:
        idx = str(s["feature_idx"])
        highly[idx] = {
            "num_activated_images": s["num_activated_samples"],
            "activation_rate": s["activation_rate"],
            "mean_activation": s["mean_activation_when_active"],
            "max_activation": s["max_activation"],
        }
    return highly


def main():
    parser = argparse.ArgumentParser(description="Identify consistent/highly activated NEW SAE features")
    parser.add_argument("--backbone", type=str, default="simclr")
    parser.add_argument("--simclr_checkpoint", type=str,
                        default="/media/data/divyanshu/sophont/checkpoints/brainiac_traincsv_simclr_norm_vit_cls_vitb_tejas_lr0005_best-model-epoch=11-train_loss=0.00.ckpt")
    parser.add_argument("--new_sae_config", type=str, default=None)
    parser.add_argument("--new_sae_checkpoint", type=str, required=True,
                        help="e.g. new_sae_checkpoints_brainage_age/sae_age_epoch_050.ckpt")
    parser.add_argument("--csv", type=str, default="data/csvs/developmental_train3_set_100.csv",
                        help="CSV to compute activations on (train recommended)")
    parser.add_argument("--root_dir", type=str, default="data/images/data")
    parser.add_argument("--output_dir", type=str, default="sae_statistical_analysis_new_sae_age")
    parser.add_argument("--min_activation_rate", type=float, default=0.01,
                        help="Min fraction of samples where feature is non-zero to count as consistent")
    parser.add_argument("--top_highly_activated", type=int, default=200,
                        help="Top N by mean activation (when active) to save as highly_activated")
    parser.add_argument("--exclude_always_on", action="store_true", default=True,
                        help="Exclude features with activation_rate >= 99%% (suspicious encoder collapse)")
    parser.add_argument("--no_exclude_always_on", action="store_false", dest="exclude_always_on")
    parser.add_argument("--always_on_threshold", type=float, default=0.99)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    project = MECHINTERP
    config_path = Path(args.new_sae_config) if args.new_sae_config else (project / "sae_training_new_sae" / "config_brainage.yml")
    if not config_path.exists():
        config_path = project / "new_sae_code" / "sae_adapter" / "config.yml"
    ckpt_path = project / args.new_sae_checkpoint if not os.path.isabs(args.new_sae_checkpoint) else args.new_sae_checkpoint
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading SimCLR backbone...")
    backbone = load_simclr_backbone(args.simclr_checkpoint, device)
    print("Loading NEW SAE...")
    new_sae = load_new_sae(str(config_path), str(ckpt_path), device, use_embeddings=True)

    csv_path = project / args.csv
    root_dir = project / args.root_dir
    dataset = BrainAgeDataset(csv_path=str(csv_path), root_dir=str(root_dir), transform=get_validation_transform((96, 96, 96)))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=(device.type == "cuda"))

    print("Extracting activations...")
    z, pat_ids = extract_activations(backbone, new_sae, loader, device)
    print(f"  Shape: {z.shape}")

    print("Computing per-feature statistics...")
    stats = compute_feature_stats(z)

    # Sanity: per-sample non-zero count (should be topk, e.g. 64)
    nonzeros_per_sample = (z != 0).sum(axis=1)
    n_samples = z.shape[0]
    print(f"  Non-zero per sample: min={int(nonzeros_per_sample.min())} max={int(nonzeros_per_sample.max())} (expect topk=64)")

    consistent, always_on = identify_consistent(
        stats, min_activation_rate=args.min_activation_rate,
        exclude_always_on=args.exclude_always_on, always_on_threshold=args.always_on_threshold,
    )
    print(f"Consistent features (activation_rate >= {args.min_activation_rate}, excluding always-on): {len(consistent)}")
    if always_on:
        print(f"  Excluded {len(always_on)} always-on features (activation_rate >= {args.always_on_threshold}) as suspicious")

    highly = identify_highly_activated(
        stats, top_n=args.top_highly_activated,
        exclude_always_on=args.exclude_always_on, always_on_threshold=args.always_on_threshold,
    )
    print(f"Highly activated (top {args.top_highly_activated} by mean when active, excluding always-on): {len(highly)}")

    out_dir = project / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    consistent_path = out_dir / "consistent_features.json"
    with open(consistent_path, "w") as f:
        json.dump(consistent, f, indent=2)
    print(f"Saved {consistent_path}")

    if always_on:
        always_on_path = out_dir / "suspicious_always_on_features.json"
        with open(always_on_path, "w") as f:
            json.dump(always_on, f, indent=2)
        print(f"Saved always-on (suspicious) features to {always_on_path}")

    highly_path = out_dir / "highly_activated_features.json"
    with open(highly_path, "w") as f:
        json.dump(highly, f, indent=2)
    print(f"Saved {highly_path}")

    feature_summary = []
    for s in stats:
        if s["activation_rate"] >= 1e-4:
            feature_summary.append(s)
    summary_path = out_dir / "feature_summary.json"
    with open(summary_path, "w") as f:
        json.dump(feature_summary, f, indent=2)
    print(f"Saved feature summary ({len(feature_summary)} features with rate >= 0.01%) to {summary_path}")

    print("\nDone. Use consistent_features.json with mask_and_mae_change_new_sae.py (--consistent_features_json sae_statistical_analysis_new_sae_age/consistent_features.json).")


if __name__ == "__main__":
    main()
