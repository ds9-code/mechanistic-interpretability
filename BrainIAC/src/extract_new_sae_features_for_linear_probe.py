"""
Extract NEW SAE decoder output for linear probe training.
Uses SimCLR (default) or BrainIAC -> NEW SAE encoder -> sparsity -> decoder -> save decoded features (768-d).
SimCLR avoids domain mismatch (NEW SAE was trained on SimCLR embeddings).
Saves normalization_stats (mean, std from train set) for consistent normalization.
Output: sae_features_linear_probe_new_sae/features_train.pt, features_val.pt, features_test.pt
"""

import os
import sys
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

# Add paths: BrainIAC first (for dataset), then new_sae and sophont (for model)
MECHINTERP = Path(__file__).resolve().parents[2]
BRAINIAC_SRC = Path(__file__).resolve().parent
NEW_SAE_ADAPTER = MECHINTERP / "new_sae_code" / "sae_adapter"
SOPHONT = Path("/media/data/divyanshu/sophont")
for p in [SOPHONT, NEW_SAE_ADAPTER]:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))
# Ensure BrainIAC/src is first so "dataset" -> BrainAgeDataset (not sae_adapter's dataset)
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


def process_split(backbone, new_sae, csv_path, root_dir, output_path, device,
                  batch_size=8, num_workers=4, use_simclr=True):
    """Extract NEW SAE decoder output for one split. No normalization on input."""
    transform = get_validation_transform(image_size=(96, 96, 96))
    dataset = BrainAgeDataset(csv_path=csv_path, root_dir=root_dir, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    backbone.eval() if hasattr(backbone, "eval") else None
    new_sae.eval()

    all_features = []
    all_labels = []
    all_pat_ids = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Extracting {os.path.basename(output_path)}"):
            images = batch["image"].to(device)
            labels = batch["label"].cpu().numpy()
            pat_ids = batch.get("pat_id", [f"sample_{i}" for i in range(len(labels))])
            if isinstance(pat_ids, torch.Tensor):
                pat_ids = pat_ids.cpu().tolist()
            elif hasattr(pat_ids, "__len__") and len(pat_ids) and not isinstance(pat_ids[0], str):
                pat_ids = [str(p) for p in pat_ids]

            if use_simclr:
                x = simclr_forward(backbone, images)
            else:
                x = backbone(images)
            z_raw = new_sae.encoder(x)
            z, _ = new_sae.apply_sparsity(z_raw)
            decoded = new_sae.decoder(z)
            all_features.append(decoded.cpu())
            all_labels.append(labels)
            all_pat_ids.extend(pat_ids if isinstance(pat_ids, list) else [pat_ids])

    features = torch.cat(all_features, dim=0)
    labels = np.concatenate(all_labels, axis=0)
    save_dict = {"features": features, "labels": labels, "pat_ids": all_pat_ids}
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save(save_dict, output_path)
    print(f"  Saved {features.shape[0]} samples to {output_path}")
    return features, labels, all_pat_ids


def main():
    parser = argparse.ArgumentParser(description="Extract NEW SAE decoder features for linear probe")
    parser.add_argument("--backbone", type=str, default="simclr", choices=["simclr", "brainiac"])
    parser.add_argument("--brainiac_checkpoint", type=str, default="BrainIAC/src/checkpoints/BrainIAC.ckpt")
    parser.add_argument("--simclr_checkpoint", type=str,
                        default="/media/data/divyanshu/sophont/checkpoints/brainiac_traincsv_simclr_norm_vit_cls_vitb_tejas_lr0005_best-model-epoch=11-train_loss=0.00.ckpt")
    parser.add_argument("--new_sae_config", type=str, default=None)
    parser.add_argument("--new_sae_checkpoint", type=str,
                        default="/media/data/divyanshu/sophont/checkpoints/sae_adapter_top64_latent49152_precomputedembeddings_vaniallasae_lr0003_tejas-epoch=49-train_loss=0.0008.ckpt")
    parser.add_argument("--train_csv", type=str, default="data/csvs/developmental_train3_set_100.csv")
    parser.add_argument("--val_csv", type=str, default="data/csvs/developmental_val3_set_100.csv")
    parser.add_argument("--test_csv", type=str, default="data/csvs/development_test_set.csv")
    parser.add_argument("--root_dir", type=str, default="data/images/data")
    parser.add_argument("--output_dir", type=str, default="sae_features_linear_probe_new_sae")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    project = MECHINTERP
    config_path = Path(args.new_sae_config) if args.new_sae_config else (MECHINTERP / "new_sae_code" / "sae_adapter" / "config.yml")
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

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

    root_dir = project / args.root_dir
    splits = [
        ("train", args.train_csv, os.path.join(args.output_dir, "features_train.pt")),
        ("val", args.val_csv, os.path.join(args.output_dir, "features_val.pt")),
        ("test", args.test_csv, os.path.join(args.output_dir, "features_test.pt")),
    ]
    train_features = None
    for name, csv_path, out_path in splits:
        print(f"\n{name}...")
        feat, lbl, _ = process_split(
            backbone, new_sae, str(project / csv_path), str(root_dir), out_path, device,
            batch_size=args.batch_size, num_workers=args.num_workers, use_simclr=use_simclr,
        )
        if name == "train":
            train_features = feat

    # Compute and save normalization stats from train set
    if train_features is not None:
        mean = train_features.mean(dim=0, keepdim=True)
        std = train_features.std(dim=0, keepdim=True) + 1e-8
        norm_path = os.path.join(args.output_dir, "features_train.pt")
        data = torch.load(norm_path, map_location="cpu", weights_only=False)
        data["normalization_stats"] = {"mean": mean, "std": std}
        torch.save(data, norm_path)
        print(f"\nSaved normalization_stats to {norm_path}")

    print("\nDone. Next: train_linear_probe.py on sae_features_linear_probe_new_sae/")
    print("  For best MAE use: --no_normalize --learning_rate 1e-2 (normalized features cause wrong-scale predictions)")


if __name__ == "__main__":
    main()
