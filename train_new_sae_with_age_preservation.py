"""
Train the NEW SAE (SAEAdapterMLPGrounded) with frozen backbone + age preservation:
  - Frozen SimCLR backbone (same as used for embedding extraction)
  - SAE encoder/decoder trained with reconstruction + age preservation loss
  - Age probe trained on raw SimCLR features (use train_age_probe_on_simclr.py first)

Run from mechinterp root with:
  PYTHONPATH=BrainIAC/src:new_sae_code:$PYTHONPATH python train_new_sae_with_age_preservation.py \\
    --config sae_training_new_sae/config_brainage.yml \\
    --train_csv data/csvs/developmental_train3_set_100.csv \\
    --root_dir data/images/data \\
    --age_probe_path sae_training_new_sae/age_probe_simclr_brainage.pt \\
    --age_preservation_weight 0.05
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

MECHINTERP = Path(__file__).resolve().parent
BRAINIAC_SRC = MECHINTERP / "BrainIAC" / "src"
NEW_SAE = MECHINTERP / "new_sae_code"
NEW_SAE_ADAPTER = NEW_SAE / "sae_adapter"

# BrainIAC first so dataset/load_simclr resolve to BrainIAC
for p in [str(BRAINIAC_SRC), str(NEW_SAE)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from dataset import BrainAgeDataset, get_validation_transform
from load_simclr import load_simclr_backbone, simclr_forward


class LinearRegressor(torch.nn.Module):
    """Same as train_sae / train_linear_probe for loading age probe."""
    def __init__(self, input_dim=768):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, 1)
    def forward(self, x):
        return self.linear(x).squeeze(-1)


def load_age_probe(ckpt_path, device, input_dim=768):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    dim = ckpt.get("input_dim", input_dim)
    probe = LinearRegressor(input_dim=dim).to(device)
    probe.load_state_dict(ckpt["model_state_dict"])
    probe.eval()
    for p in probe.parameters():
        p.requires_grad = False
    return probe


def load_sae_from_config_and_checkpoint(config_path, checkpoint_path, device):
    """Load SAEAdapterMLPGrounded from YAML; optionally load weights from checkpoint."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    # Ensure use_embeddings so we feed precomputed x (no backbone inside SAE)
    cfg["data"] = cfg.get("data", {})
    cfg["data"]["use_embeddings"] = True

    # Import after path is set
    from sae_adapter.model_mlp_grounded import SAEAdapterMLPGrounded
    model = SAEAdapterMLPGrounded(cfg).to(device)
    if checkpoint_path and os.path.isfile(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state = ckpt.get("state_dict", ckpt)
        model.load_state_dict(state, strict=False)
        print(f"  Loaded SAE weights from {checkpoint_path}")
    return model, cfg


def train_epoch(simclr, sae, age_probe, loader, optimizer, device, age_weight, use_amp=True):
    sae.train()
    total_loss = 0.0
    total_rec = 0.0
    total_age = 0.0
    n = 0
    scaler = torch.amp.GradScaler("cuda") if use_amp and device.type == "cuda" else None

    for batch in tqdm(loader, desc="Train", leave=False):
        images = batch["image"].to(device)
        with torch.no_grad():
            x = simclr_forward(simclr, images)

        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
            z_raw = sae.encoder(x)
            z, l1_loss = sae.apply_sparsity(z_raw)
            recon = sae.decoder(z)
            rec_loss = F.mse_loss(recon, x)
            age_pred_orig = age_probe(x)
            age_pred_recon = age_probe(recon)
            age_loss = F.mse_loss(age_pred_recon, age_pred_orig)
            loss = rec_loss + age_weight * age_loss
            if sae.sparsity_type == "l1":
                loss = loss + sae.l1_weight * l1_loss

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        total_rec += rec_loss.item()
        total_age += age_loss.item()
        n += 1

    return total_loss / n, total_rec / n, total_age / n


def main():
    parser = argparse.ArgumentParser(description="Train NEW SAE with frozen SimCLR + age preservation")
    parser.add_argument("--config", type=str, default="sae_training_new_sae/config_brainage.yml")
    parser.add_argument("--train_csv", type=str, default="data/csvs/developmental_train3_set_100.csv")
    parser.add_argument("--root_dir", type=str, default="data/images/data")
    parser.add_argument("--age_probe_path", type=str, required=True, help="Path to age probe .pt (train with train_age_probe_on_simclr.py)")
    parser.add_argument("--age_preservation_weight", type=float, default=0.05)
    parser.add_argument("--resume", type=str, default=None, help="Path to SAE checkpoint to resume from")
    parser.add_argument("--save_dir", type=str, default="/media/data/divyanshu/diya/mechinterp/new_sae_checkpoints_brainage_age")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--wandb_project", type=str, default="brainiac_sae")
    parser.add_argument("--wandb_run_name", type=str, default="sae_new_brainage_age")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    config_path = MECHINTERP / args.config
    train_csv = MECHINTERP / args.train_csv
    root_dir = MECHINTERP / args.root_dir
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not train_csv.is_file():
        raise FileNotFoundError(f"Train CSV not found: {train_csv}")
    if not root_dir.is_dir():
        raise FileNotFoundError(f"Root dir not found: {root_dir}")
    age_probe_path = Path(args.age_probe_path)
    if not age_probe_path.is_absolute():
        age_probe_path = MECHINTERP / age_probe_path
    if not age_probe_path.is_file():
        raise FileNotFoundError(f"Age probe not found: {age_probe_path}")

    os.makedirs(args.save_dir, exist_ok=True)

    # Init wandb early so the run appears even if we crash during model load
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "age_preservation_weight": args.age_preservation_weight,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "save_dir": args.save_dir,
                "config": args.config,
                "resume": args.resume,
            },
            dir=args.save_dir,
        )
        print(f"Wandb run: {args.wandb_project}/{wandb.run.name}")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    simclr_ckpt = cfg["backbone"]["checkpoint"]
    if not os.path.isfile(simclr_ckpt):
        raise FileNotFoundError(f"SimCLR checkpoint not found: {simclr_ckpt}")

    print("Loading frozen SimCLR backbone...")
    simclr = load_simclr_backbone(simclr_ckpt, device)
    print("Loading age probe...")
    age_probe = load_age_probe(str(age_probe_path), device)
    print("Loading NEW SAE...")
    sae, _ = load_sae_from_config_and_checkpoint(config_path, args.resume, device)

    transform = get_validation_transform(image_size=(96, 96, 96))
    dataset = BrainAgeDataset(csv_path=str(train_csv), root_dir=str(root_dir), transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=(device.type == "cuda"))

    optimizer = torch.optim.AdamW(
        [p for p in sae.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=cfg.get("optim", {}).get("weight_decay", 1e-4),
    )

    if args.no_wandb:
        wandb.run = None

    print(f"Training for {args.epochs} epochs (age_preservation_weight={args.age_preservation_weight})")
    print(f"Checkpoints: {args.save_dir}\n")

    for epoch in range(1, args.epochs + 1):
        loss_avg, rec_avg, age_avg = train_epoch(
            simclr, sae, age_probe, loader, optimizer, device,
            age_weight=args.age_preservation_weight,
        )
        print(f"Epoch {epoch:3d}  loss={loss_avg:.4f}  rec={rec_avg:.4f}  age={age_avg:.4f}")
        if not args.no_wandb and wandb.run is not None:
            wandb.log({
                "train/loss": loss_avg,
                "train/rec_loss": rec_avg,
                "train/age_preservation_loss": age_avg,
                "epoch": epoch,
            }, step=epoch)
        # Save checkpoint: every save_every epochs, and always after epoch 1 and at last epoch
        if epoch == 1 or epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt_path = os.path.join(args.save_dir, f"sae_age_epoch_{epoch:03d}.ckpt")
            torch.save({"state_dict": sae.state_dict(), "epoch": epoch}, ckpt_path)
            print(f"  Saved {ckpt_path}")

    if not args.no_wandb and wandb.run is not None:
        wandb.finish()
    print("Done.")


if __name__ == "__main__":
    main()
