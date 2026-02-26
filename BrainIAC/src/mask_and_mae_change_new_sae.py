"""
Evaluate MAE impact of masking each consistent feature for NEW SAE.
Pipeline: SimCLR (default) or BrainIAC -> NEW SAE (encode -> sparsity -> decode) -> normalize -> linear probe.
SimCLR avoids domain mismatch (NEW SAE was trained on SimCLR embeddings).
Positive test_mae_delta = masking worsened MAE = feature is meaningful for brain age.
Outputs top 3 by test_mae_delta (most meaningful).
Uses probe trained on NEW SAE decoder output and sae_statistical_analysis_new_sae/consistent_features.json.
"""

import os
import sys
import json
import argparse
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
from monai.data.meta_tensor import MetaTensor
from numpy.core.multiarray import _reconstruct
from numpy import ndarray, dtype

# Add paths: BrainIAC first (for dataset), then new_sae and sophont
MECHINTERP = Path(__file__).resolve().parents[2]
BRAINIAC_SRC = Path(__file__).resolve().parent
NEW_SAE_ADAPTER = MECHINTERP / "new_sae_code" / "sae_adapter"
SOPHONT = Path("/media/data/divyanshu/sophont")
for p in [SOPHONT, NEW_SAE_ADAPTER]:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))
sys.path.insert(0, str(BRAINIAC_SRC))

from dataset import BrainAgeDataset, get_validation_transform
from load_brainiac import load_brainiac
from load_simclr import load_simclr_backbone, simclr_forward
from model_mlp_grounded import SAEAdapterMLPGrounded


def load_new_sae(config_path, checkpoint_path, device, use_embeddings=True):
    """Load NEW SAE (SAEAdapterMLPGrounded)."""
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
    """Load linear probe checkpoint."""
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
def extract_activations_and_labels(backbone, new_sae, loader, device, use_simclr=True):
    """One pass: extract all test latent z (activations) and labels."""
    all_z = []
    all_label = []
    for batch in loader:
        x = batch["image"].to(device)
        y = batch["label"]
        if use_simclr:
            emb = simclr_forward(backbone, x)
        else:
            emb = backbone(x)
        z_raw = new_sae.encoder(emb)
        z, _ = new_sae.apply_sparsity(z_raw)
        all_z.append(z.cpu())
        all_label.append(y)
    return torch.cat(all_z), torch.cat(all_label).float()


@torch.no_grad()
def mae_from_activations(new_sae, probe, activations, labels, mean, std, device, mask_feature_idx=None):
    """Decode activations (optionally masked), normalize, probe; return MAE."""
    act = activations.to(device)
    if mask_feature_idx is not None:
        act = act.clone()
        act[:, mask_feature_idx] = 0
    decoded = new_sae.decoder(act)
    decoded_norm = (decoded - mean) / (std + 1e-8)
    pred = probe(decoded_norm).squeeze(-1).cpu()
    mae = torch.mean(torch.abs(pred - labels)).item()
    return mae


def main():
    parser = argparse.ArgumentParser(description="Evaluate MAE impact of masking consistent features (NEW SAE)")
    parser.add_argument("--backbone", type=str, default="simclr", choices=["simclr", "brainiac"])
    parser.add_argument("--brainiac_checkpoint", type=str, default="BrainIAC/src/checkpoints/BrainIAC.ckpt")
    parser.add_argument("--simclr_checkpoint", type=str,
                        default="/media/data/divyanshu/sophont/checkpoints/brainiac_traincsv_simclr_norm_vit_cls_vitb_tejas_lr0005_best-model-epoch=11-train_loss=0.00.ckpt")
    parser.add_argument("--new_sae_config", type=str, default=None)
    parser.add_argument("--new_sae_checkpoint", type=str,
                        default="/media/data/divyanshu/sophont/checkpoints/sae_adapter_top64_latent49152_precomputedembeddings_vaniallasae_lr0003_tejas-epoch=49-train_loss=0.0008.ckpt")
    parser.add_argument("--linear_probe", type=str, default="linear_probe_results_new_sae/best_model.pt")
    parser.add_argument("--features_train_pt", type=str, default="sae_features_linear_probe_new_sae/features_train.pt",
                        help="Used to load normalization stats (mean, std)")
    parser.add_argument("--consistent_features_json", type=str, default="sae_statistical_analysis_new_sae/consistent_features.json")
    parser.add_argument("--test_csv", type=str, default="data/csvs/development_test_set.csv")
    parser.add_argument("--root_dir", type=str, default="data/images/data")
    parser.add_argument("--output_dir", type=str, default="linear_probe_results_new_sae")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--feature_indices", type=str, default=None,
                        help="Comma-separated list of feature indices to evaluate (default: all in consistent_features.json)")
    parser.add_argument("--no_normalize", action="store_true",
                        help="Do not normalize decoder output (use when probe was trained with --no_normalize)")
    args = parser.parse_args()

    project = MECHINTERP
    config_path = Path(args.new_sae_config) if args.new_sae_config else (MECHINTERP / "new_sae_code" / "sae_adapter" / "config.yml")
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    with open(project / args.consistent_features_json) as f:
        consistent = json.load(f)
    if args.feature_indices:
        feature_indices = [int(x.strip()) for x in args.feature_indices.split(",")]
        print(f"Using {len(feature_indices)} specified features: {feature_indices}")
    else:
        feature_indices = [int(k) for k in consistent.keys()]
        print(f"Loaded {len(feature_indices)} consistent features: {feature_indices[:10]}...")

    # Load normalization stats from features_train.pt (unless --no_normalize for probe trained on raw features)
    if args.no_normalize:
        mean = torch.zeros(1, 768, device=device)
        std = torch.ones(1, 768, device=device)
        print("Using no normalization (probe expects raw decoder output)")
    else:
        train_data = torch.load(project / args.features_train_pt, map_location="cpu", weights_only=False)
        norm_stats = train_data.get("normalization_stats")
        if norm_stats is None:
            raise ValueError("features_train.pt must contain normalization_stats. Re-run extract_new_sae_features_for_linear_probe.py")
        mean = norm_stats["mean"].to(device)
        std = norm_stats["std"].to(device)
        if mean.dim() == 1:
            mean = mean.unsqueeze(0)
            std = std.unsqueeze(0)

    if args.backbone == "simclr":
        print("Loading SimCLR backbone (avoids domain mismatch)...")
        backbone = load_simclr_backbone(args.simclr_checkpoint, device)
        use_simclr = True
    else:
        print("Loading BrainIAC...")
        backbone = load_brainiac(str(project / args.brainiac_checkpoint), device)
        backbone.eval()
        use_simclr = False
    print("Loading NEW SAE...")
    new_sae = load_new_sae(str(config_path), args.new_sae_checkpoint, device, use_embeddings=True)
    print("Loading linear probe...")
    probe = load_probe(str(project / args.linear_probe), device)

    dataset = BrainAgeDataset(
        csv_path=str(project / args.test_csv),
        root_dir=str(project / args.root_dir),
        transform=get_validation_transform((96, 96, 96)),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=(device.type == "cuda"))

    print("\nExtracting test activations...")
    activations, labels = extract_activations_and_labels(backbone, new_sae, loader, device, use_simclr=use_simclr)
    print(f"  Activations: {activations.shape}, labels: {labels.shape}")

    print("\nComputing baseline test MAE (no masking)...")
    baseline_mae = mae_from_activations(new_sae, probe, activations, labels, mean, std, device, mask_feature_idx=None)
    print(f"Baseline test MAE: {baseline_mae:.2f} months")

    print(f"\nEvaluating MAE when masking each of {len(feature_indices)} features...")
    results = []
    for idx in tqdm(feature_indices, desc="Masking"):
        mae_masked = mae_from_activations(new_sae, probe, activations, labels, mean, std, device, mask_feature_idx=idx)
        delta = mae_masked - baseline_mae
        results.append({"feature_idx": idx, "test_mae_masked": mae_masked, "test_mae_delta": delta})

    results.sort(key=lambda x: x["test_mae_delta"], reverse=True)
    top_k = min(args.top_k, len(results))
    top_features = [r["feature_idx"] for r in results[:top_k]]
    print(f"\nTop {top_k} most meaningful features (masking them worsens MAE the most):")
    for i, r in enumerate(results[:top_k], 1):
        print(f"  {i}. Feature {r['feature_idx']}: test_mae_delta = +{r['test_mae_delta']:.2f} months (masked MAE = {r['test_mae_masked']:.2f})")

    out_path = project / args.output_dir / "consistent_features_mae_impact_new_sae.json"
    with open(out_path, "w") as f:
        json.dump({
            "baseline_test_mae": baseline_mae,
            "results": results,
            "top_k_most_meaningful": top_features,
            "consistent_features_evaluated": feature_indices,
        }, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Save top-k table: Rank, Feature index, Test MAE when masked, Δ MAE (months)
    top_table_path = project / args.output_dir / "top_features_mae_impact_new_sae.csv"
    with open(top_table_path, "w") as f:
        f.write("Rank,Feature index,Test MAE when masked,Δ MAE (months)\n")
        for i, r in enumerate(results[: top_k], 1):
            f.write(f"{i},{r['feature_idx']},{r['test_mae_masked']:.2f},{r['test_mae_delta']:.2f}\n")
    print(f"Top {top_k} table saved to {top_table_path}")


if __name__ == "__main__":
    main()
