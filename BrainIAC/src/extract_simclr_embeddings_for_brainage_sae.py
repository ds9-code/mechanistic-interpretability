"""
Extract SimCLR 768-d CLS embeddings for brainage data, for NEW SAE training.
Output CSV: pat_id, label, dataset, f0, f1, ..., f767.
Use this CSV as embedding_csv in new_sae_code config for brainage SAE training.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

BRAINIAC_SRC = Path(__file__).resolve().parent
MECHINTERP = BRAINIAC_SRC.parent.parent  # mechinterp repo root
if str(BRAINIAC_SRC) not in sys.path:
    sys.path.insert(0, str(BRAINIAC_SRC))

from dataset import BrainAgeDataset, get_validation_transform
from load_simclr import load_simclr_backbone, simclr_forward


def main():
    parser = argparse.ArgumentParser(description="Extract SimCLR embeddings for brainage SAE training")
    parser.add_argument("--simclr_checkpoint", type=str,
                        default="/media/data/divyanshu/sophont/checkpoints/brainiac_traincsv_simclr_norm_vit_cls_vitb_tejas_lr0005_best-model-epoch=11-train_loss=0.00.ckpt")
    parser.add_argument("--train_csv", type=str, default="data/csvs/developmental_train3_set_100.csv")
    parser.add_argument("--root_dir", type=str, default="data/images/data")
    parser.add_argument("--output_csv", type=str, default="sae_training_new_sae/brainage_train_simclr_embeddings.csv")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    project = MECHINTERP
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)

    print("Loading SimCLR backbone...")
    backbone = load_simclr_backbone(args.simclr_checkpoint, device)

    transform = get_validation_transform(image_size=(96, 96, 96))
    dataset = BrainAgeDataset(
        csv_path=str(project / args.train_csv),
        root_dir=str(project / args.root_dir),
        transform=transform,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=(device.type == "cuda"))

    all_emb = []
    all_pat_id = []
    all_label = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting SimCLR embeddings"):
            images = batch["image"].to(device)
            emb = simclr_forward(backbone, images)
            all_emb.append(emb.cpu().numpy())
            all_pat_id.extend(batch["pat_id"])
            all_label.extend(batch["label"].cpu().tolist())

    emb_arr = np.concatenate(all_emb, axis=0)
    n_dim = emb_arr.shape[1]
    feat_cols = [f"f{i}" for i in range(n_dim)]

    df_emb = pd.DataFrame(emb_arr, columns=feat_cols)
    df_meta = pd.DataFrame({"pat_id": all_pat_id, "label": all_label})
    df_meta["dataset"] = "brainage"
    df = pd.concat([df_meta, df_emb], axis=1)

    out_path = project / args.output_csv
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} samples to {out_path}")
    print(f"Columns: pat_id, label, dataset, f0..f{n_dim-1}")


if __name__ == "__main__":
    main()
