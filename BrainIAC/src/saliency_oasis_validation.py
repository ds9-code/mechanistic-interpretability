"""
Validate SAE feature intervention using saliency maps on OASIS.

For ~5 OASIS images:
- Compute gradient-based saliency (d(pred)/d(image)) for:
  1. Baseline: no feature adjustment
  2. Adjusted: increase 857 and 1796 (raise brain age), decrease 17827 (raise brain age)
- Save both saliency maps (before/after) and predictions for comparison.

We adjust features 857, 17827, 1796 so predicted age = true age using:
  pred(v) = pred_at_zero + slope_857*v_857 + slope_17827*v_17827 + slope_1796*v_1796.
Slopes computed per image; then v chosen so pred = true_age (minimum-norm solution).
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import nibabel as nib
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
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
from load_simclr import load_simclr_backbone
from model_mlp_grounded import SAEAdapterMLPGrounded
from load_brainiac import load_brainiac
from sae_model import GatedSAE
import torch.nn.functional as F

# Features we adjust to match predicted age to true age (new SAE)
FEATURE_IDS_NEW = (857, 17827, 1796)
# Old SAE: meaningful features identified earlier
FEATURE_IDS_OLD = (9607, 8700, 23673)


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


def load_old_sae(checkpoint_path, device):
    """Load old GatedSAE from sae_checkpoints_x32_full_norm."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    n_input = config.get("n_input_features", 768)
    expansion = config.get("expansion_factor", 32)
    n_dict = n_input * expansion
    model = GatedSAE(
        n_input_features=n_input,
        n_dict_features=n_dict,
        expansion_factor=expansion,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    norm_stats = ckpt.get("normalization_stats")
    return model, norm_stats


def forward_to_pred(backbone, new_sae, probe, x, intervene_z=None):
    """x: [1, 1, D, H, W]. Optional intervene_z: dict {idx: value} to set in z.
    value can be float or tensor (shape (1,) or scalar) so gradient can flow through."""
    features = backbone(x)
    emb = features[0][:, 0, :]   # [1, 768]
    z_raw = new_sae.encoder(emb)
    z, _ = new_sae.apply_sparsity(z_raw)
    if intervene_z is not None:
        z = z.clone()
        for idx, val in intervene_z.items():
            if isinstance(val, (int, float)):
                val = torch.tensor(val, dtype=z.dtype, device=z.device)
            # ensure (1,) for z[:, idx] which is (1,)
            if val.dim() == 0:
                val = val.unsqueeze(0)
            z[:, idx] = val
    decoded = new_sae.decoder(z)
    pred = probe(decoded).squeeze(-1)   # [1]
    return pred, z


def forward_old_sae(backbone, sae, probe, x, intervene_z=None, norm_stats=None):
    """Old SAE pipeline: BrainIAC -> cls -> GatedSAE (encode, gate, ReLU) -> intervene on z -> decode -> probe."""
    cls_feat = backbone(x)  # [B, 768]
    if norm_stats is not None:
        mean = norm_stats["mean"].to(x.device)
        std = norm_stats["std"].to(x.device)
        cls_feat = (cls_feat - mean) / (std + 1e-8)
    encoded = sae.encoder(cls_feat)
    gate_vals = torch.sigmoid(sae.gate(cls_feat))
    z = F.relu(encoded * gate_vals)
    if intervene_z is not None:
        z = z.clone()
        for idx, val in intervene_z.items():
            if isinstance(val, (int, float)):
                val = torch.tensor(val, dtype=z.dtype, device=z.device)
            if val.dim() == 0:
                val = val.unsqueeze(0)
            z[:, idx] = val
    decoded = sae.decoder(z)
    pred = probe(decoded).squeeze(-1)
    return pred, z


def compute_intervention_for_true_age(backbone, sae, probe, x, true_age, device, feature_ids,
                                      forward_fn=None):
    """
    Compute feature values so that predicted age = true_age.
    Uses: pred(v) = pred_at_zero + sum(slope_i * v_i). Minimum-norm solution.
    Returns dict {idx: v_i} or None if slopes degenerate.
    """
    forward_fn = forward_fn or forward_to_pred
    with torch.no_grad():
        zero_intervene = {i: 0.0 for i in feature_ids}
        base, _ = forward_fn(backbone, sae, probe, x, intervene_z=zero_intervene)
        base = base.item()
        slopes = []
        for idx in feature_ids:
            inter = {i: 0.0 for i in feature_ids}
            inter[idx] = 1.0
            p, _ = forward_fn(backbone, sae, probe, x, intervene_z=inter)
            slopes.append((p.item() - base))
        slopes = np.array(slopes, dtype=np.float64)
    delta = true_age - base
    denom = np.sum(slopes ** 2)
    if denom < 1e-12:
        return None
    coeff = delta / denom
    values = coeff * slopes
    return {idx: float(values[j]) for j, idx in enumerate(feature_ids)}


def compute_intervention_for_true_age_differentiable(backbone, sae, probe, x, true_age, device,
                                                     feature_ids, forward_fn=None):
    """
    Same as compute_intervention_for_true_age but returns intervention values as **tensors**.
    """
    forward_fn = forward_fn or forward_to_pred
    zero_intervene = {i: 0.0 for i in feature_ids}
    base, _ = forward_fn(backbone, sae, probe, x, intervene_z=zero_intervene)
    slopes = []
    for idx in feature_ids:
        inter = {i: 0.0 for i in feature_ids}
        inter[idx] = 1.0
        p, _ = forward_fn(backbone, sae, probe, x, intervene_z=inter)
        slopes.append((p - base).squeeze(0))
    slopes = torch.stack(slopes)
    delta = (torch.tensor(true_age, dtype=base.dtype, device=base.device) - base.squeeze(0)).squeeze(0)
    denom = (slopes ** 2).sum()
    if denom.item() < 1e-12:
        return None
    coeff = delta / denom
    values = coeff * slopes
    return {idx: values[j].unsqueeze(0) for j, idx in enumerate(feature_ids)}


# ViT patch size (must match backbone: 96/16 = 6 patches per dim)
PATCH_SIZE = 16
IMG_SIZE = 96
N_PATCHES = IMG_SIZE // PATCH_SIZE  # 6


def _brain_mask(input_vol, percentile=25):
    """Rough brain mask: voxels above percentile of intensity (excludes background/corners)."""
    in_flat = np.asarray(input_vol).astype(np.float64).ravel()
    in_flat = in_flat[~np.isnan(in_flat) & np.isfinite(in_flat)]
    if in_flat.size == 0:
        return np.ones_like(input_vol, dtype=bool)
    thresh = np.percentile(in_flat, percentile)
    return np.asarray(input_vol, dtype=np.float64) > thresh


def gradient_saliency(image, pred, input_vol_for_mask=None):
    """Backprop pred w.r.t. image; return normalized absolute gradient as numpy [D,H,W].
    If input_vol_for_mask is provided (same shape as image), saliency is masked to the brain
    and normalized only within the brain so hot spots appear on tissue, not background.
    """
    if image.grad is not None:
        image.grad.zero_()
    pred.backward()
    grad = image.grad.detach().squeeze().cpu().numpy()
    grad = np.abs(grad)
    grad = _patch_average_and_upsample(grad)
    # Mask to brain so we normalize and emphasize importance on tissue, not background/edges
    if input_vol_for_mask is not None:
        mask = _brain_mask(input_vol_for_mask)
        grad_out = np.full_like(grad, np.nan, dtype=np.float64)
        grad_out[mask] = grad[mask]
        valid = grad_out[mask]
        if valid.size > 0 and valid.max() > valid.min():
            grad_out[mask] = (grad_out[mask] - valid.min()) / (valid.max() - valid.min())
        grad = grad_out
    if grad.max() > grad.min():
        grad = (grad - grad.min()) / (grad.max() - grad.min())
    return grad


def _patch_average_and_upsample(grad_vol):
    """
    grad_vol: (D, H, W) e.g. (96, 96, 96).
    Average within each PATCH_SIZE^3 patch -> (6, 6, 6), then upsample to (96, 96, 96).
    """
    d, h, w = grad_vol.shape
    assert d == h == w == IMG_SIZE, f"expected {IMG_SIZE}^3, got {d}x{h}x{w}"
    # Reshape to (n_patches_d, patch_d, n_patches_h, patch_h, n_patches_w, patch_w)
    g = grad_vol.reshape(
        N_PATCHES, PATCH_SIZE,
        N_PATCHES, PATCH_SIZE,
        N_PATCHES, PATCH_SIZE,
    )
    # Average over patch dimensions -> (6, 6, 6)
    patch_importance = np.mean(g, axis=(1, 3, 5))
    # Upsample back to (96, 96, 96) with trilinear interpolation (smooth, patch-consistent)
    import scipy.ndimage as ndi
    zoom_factor = (IMG_SIZE / N_PATCHES,) * 3
    upsampled = ndi.zoom(
        patch_importance.astype(np.float64),
        zoom_factor,
        order=1,
        mode="nearest",
    )
    return upsampled.astype(np.float32)


def _middle_slices(vol):
    """vol shape (D, H, W) e.g. (96, 96, 96). Return axial, coronal, sagittal middle slices."""
    d, h, w = vol.shape
    axial = vol[:, :, w // 2]    # (D, H)
    coronal = vol[:, h // 2, :]  # (D, W)
    sagittal = vol[d // 2, :, :] # (H, W)
    return axial, coronal, sagittal


def _normalize_mri_brainiac(img):
    """Robust MRI normalization: clip to 1st and 99th percentiles, then scale to [0,1] (BrainIAC quickstart)."""
    img = np.asarray(img, dtype=np.float64)
    flat = img[np.isfinite(img)].ravel()
    if flat.size == 0:
        return np.zeros_like(img)
    p1, p99 = np.percentile(flat, (1, 99))
    img_clipped = np.clip(img, p1, p99)
    return (img_clipped - p1) / (p99 - p1) if p99 > p1 else np.zeros_like(img)


def _normalize_saliency_brainiac(saliency):
    """Saliency: clip negatives to 0, then divide by max (BrainIAC quickstart). Ignores NaN."""
    s = np.asarray(saliency, dtype=np.float64).copy()
    s[s < 0] = 0
    m = np.nanmax(s)
    if m > 0:
        s = np.where(np.isfinite(s), s / m, 0.0)
    return s


def _process_saliency_brainiac(saliency_slice):
    """Gaussian blur then max-normalize (matches BrainIAC quickstart). Handles NaN (e.g. outside brain)."""
    import scipy.ndimage as ndi
    s = np.asarray(saliency_slice, dtype=np.float64).copy()
    # Our saliency can have NaN outside brain; treat as 0 for blur so we only smooth valid values
    s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
    # Approximate (15,15) kernel with sigma ~2.5 (OpenCV sigma 0 with ksize 15)
    s_blurred = ndi.gaussian_filter(s, sigma=2.5, mode="nearest")
    return _normalize_saliency_brainiac(s_blurred)


def _overlay_saliency_contours(ax, input_slice, saliency_slice, n_levels=10):
    """
    BrainIAC quickstart style: MRI normalized (1st/99th percentiles), saliency Gaussian-blurred
    and max-normalized, contour overlay with magma colormap, 10 levels, linewidth 3.
    """
    in_slice = _normalize_mri_brainiac(input_slice)
    s_draw = _process_saliency_brainiac(saliency_slice)
    # Threshold: keep only non-negative (BrainIAC uses > 0; we use >= 0 so 0 stays 0)
    s_draw = np.where(s_draw > 0, s_draw, 0.0)
    h, w = s_draw.shape
    ax.imshow(in_slice, cmap="gray", aspect="equal", extent=[0, w, h, 0], origin="upper", interpolation="none")
    ax.contour(
        s_draw,
        levels=n_levels,
        cmap="magma",
        extent=[0, w, h, 0],
        origin="upper",
        linewidths=1.0,
    )
    ax.set_axis_off()
    ax.set_aspect("equal")


def _overlay_saliency_heatmap(ax, input_slice, saliency_slice, alpha_max=0.7):
    """
    BrainIAC quickstart style: same MRI and saliency processing (Gaussian blur, max-normalize),
    then saliency as magma heatmap overlay (alpha so MRI shows through).
    """
    in_slice = _normalize_mri_brainiac(input_slice)
    s_draw = _process_saliency_brainiac(saliency_slice)
    s_draw = np.where(s_draw > 0, s_draw, 0.0)
    h, w = s_draw.shape
    ax.imshow(in_slice, cmap="gray", aspect="equal", extent=[0, w, h, 0], origin="upper", interpolation="none")
    # Heatmap with alpha so gradient is visible and MRI shows through
    alpha_arr = np.clip(alpha_max * s_draw, 0, 1)
    ax.imshow(
        s_draw,
        cmap="magma",
        aspect="equal",
        extent=[0, w, h, 0],
        origin="upper",
        interpolation="none",
        alpha=alpha_arr,
        vmin=0,
        vmax=1,
    )
    ax.set_axis_off()
    ax.set_aspect("equal")


def save_volume_views_png(out_path, input_vol, sal_baseline, sal_adjusted, title_prefix=""):
    """
    Save one PNG with before/after saliency overlaid on original image (contour lines).
    Layout: 3 rows (axial, coronal, sagittal) x 2 columns (Before | After overlay).
    Volumes are (D, H, W) e.g. (96, 96, 96).
    """
    ax_in, co_in, sa_in = _middle_slices(input_vol)
    input_slices = [ax_in, co_in.T, sa_in.T]
    view_names = ["Axial", "Coronal", "Sagittal"]
    ax_b, co_b, sa_b = _middle_slices(sal_baseline)
    ax_a, co_a, sa_a = _middle_slices(sal_adjusted)
    before_slices = [ax_b, co_b.T, sa_b.T]
    after_slices = [ax_a, co_a.T, sa_a.T]

    fig, axes = plt.subplots(3, 2, figsize=(8, 9))
    for row, (vname, inp, sal_b, sal_a) in enumerate(zip(view_names, input_slices, before_slices, after_slices)):
        _overlay_saliency_contours(axes[row, 0], inp, sal_b, n_levels=10)
        axes[row, 0].set_ylabel(f"{vname}\nBefore", fontsize=9)
        _overlay_saliency_contours(axes[row, 1], inp, sal_a, n_levels=10)
        axes[row, 1].set_ylabel(f"{vname}\nAfter", fontsize=9)
        if row == 0:
            axes[row, 0].set_title("Saliency (before)", fontsize=10)
            axes[row, 1].set_title("Saliency (after)", fontsize=10)
    plt.suptitle(title_prefix, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


def save_volume_views_png_heatmap(out_path, input_vol, sal_baseline, sal_adjusted, title_prefix=""):
    """
    Same layout as save_volume_views_png but with heatmap overlay instead of contours.
    Saves to a separate file (e.g. *_views_heatmap.png) so the contour version is kept.
    """
    ax_in, co_in, sa_in = _middle_slices(input_vol)
    input_slices = [ax_in, co_in.T, sa_in.T]
    view_names = ["Axial", "Coronal", "Sagittal"]
    ax_b, co_b, sa_b = _middle_slices(sal_baseline)
    ax_a, co_a, sa_a = _middle_slices(sal_adjusted)
    before_slices = [ax_b, co_b.T, sa_b.T]
    after_slices = [ax_a, co_a.T, sa_a.T]

    fig, axes = plt.subplots(3, 2, figsize=(8, 9))
    for row, (vname, inp, sal_b, sal_a) in enumerate(zip(view_names, input_slices, before_slices, after_slices)):
        _overlay_saliency_heatmap(axes[row, 0], inp, sal_b)
        axes[row, 0].set_ylabel(f"{vname}\nBefore", fontsize=9)
        _overlay_saliency_heatmap(axes[row, 1], inp, sal_a)
        axes[row, 1].set_ylabel(f"{vname}\nAfter", fontsize=9)
        if row == 0:
            axes[row, 0].set_title("Saliency (before)", fontsize=10)
            axes[row, 1].set_title("Saliency (after)", fontsize=10)
    plt.suptitle(title_prefix, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Saliency before/after feature adjustment on OASIS")
    parser.add_argument("--sae_type", type=str, choices=["new", "old"], default="new",
                        help="new SAE (SimCLR+SAEAdapter) or old SAE (BrainIAC+GatedSAE)")
    parser.add_argument("--feature_ids", type=str, default=None,
                        help="Comma-separated feature IDs to adjust (default: 857,17827,1796 for new; 9607,8700,23673 for old)")
    parser.add_argument("--simclr_checkpoint", type=str,
                        default="/media/data/divyanshu/sophont/checkpoints/brainiac_traincsv_simclr_norm_vit_cls_vitb_tejas_lr0005_best-model-epoch=11-train_loss=0.00.ckpt")
    parser.add_argument("--brainiac_checkpoint", type=str, default="BrainIAC/src/checkpoints/BrainIAC.ckpt")
    parser.add_argument("--new_sae_config", type=str, default=None)
    parser.add_argument("--new_sae_checkpoint", type=str, default=None)
    parser.add_argument("--old_sae_checkpoint", type=str, default="BrainIAC/src/sae_checkpoints_x32_full_norm/best_model.pt")
    parser.add_argument("--linear_probe", type=str, required=True)
    parser.add_argument("--test_csv", type=str, default="data/csvs/oasis1_only.csv")
    parser.add_argument("--root_dir", type=str, default="oasis_data/oasis1/data")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--n_images", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    project = MECHINTERP
    sae_type = args.sae_type
    if args.feature_ids is not None:
        feature_ids = tuple(int(i.strip()) for i in args.feature_ids.split(","))
    else:
        feature_ids = FEATURE_IDS_OLD if sae_type == "old" else FEATURE_IDS_NEW

    out_dir = args.output_dir
    if out_dir is None:
        out_dir = "linear_probe_results/saliency_oasis_validation" if sae_type == "old" else "linear_probe_results_new_sae_age/saliency_oasis_validation"
    out_dir = project / out_dir if not str(out_dir).startswith("/") else Path(out_dir)
    probe_path = project / args.linear_probe if not str(args.linear_probe).startswith("/") else Path(args.linear_probe)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if sae_type == "new":
        if args.new_sae_checkpoint is None:
            raise ValueError("--new_sae_checkpoint required when sae_type=new")
        config_path = Path(args.new_sae_config) if args.new_sae_config else (project / "sae_training_new_sae" / "config_brainage.yml")
        if not config_path.exists():
            config_path = project / "new_sae_code" / "sae_adapter" / "config.yml"
        ckpt_path = project / args.new_sae_checkpoint if not str(args.new_sae_checkpoint).startswith("/") else Path(args.new_sae_checkpoint)
        print("Loading SimCLR backbone...")
        backbone = load_simclr_backbone(args.simclr_checkpoint, device)
        print("Loading NEW SAE...")
        sae = load_new_sae(str(config_path), str(ckpt_path), device, use_embeddings=True)
        norm_stats = None
        forward_fn = forward_to_pred
    else:
        brainiac_path = project / args.brainiac_checkpoint if not str(args.brainiac_checkpoint).startswith("/") else Path(args.brainiac_checkpoint)
        old_sae_path = project / args.old_sae_checkpoint if not str(args.old_sae_checkpoint).startswith("/") else Path(args.old_sae_checkpoint)
        print("Loading BrainIAC backbone...")
        backbone = load_brainiac(str(brainiac_path), device)
        print("Loading OLD SAE (GatedSAE)...")
        sae, norm_stats = load_old_sae(str(old_sae_path), device)
        forward_fn = lambda bb, s, p, x, intervene_z=None: forward_old_sae(
            bb, s, p, x, intervene_z=intervene_z, norm_stats=norm_stats)

    print("Loading linear probe...")
    probe = load_probe(str(probe_path), device)
    print(f"Using features: {feature_ids}")

    dataset = BrainAgeDataset(
        csv_path=str(project / args.test_csv),
        root_dir=str(project / args.root_dir),
        transform=get_validation_transform((96, 96, 96)),
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    results = []
    n_done = 0
    for batch_idx, batch in enumerate(loader):
        if n_done >= args.n_images:
            break
        x = batch["image"].to(device).clone().detach()
        x.requires_grad_(True)
        input_np = x.detach().squeeze().cpu().numpy()  # for brain masking of saliency
        label = batch["label"].item()
        pat_id = batch["pat_id"][0] if isinstance(batch["pat_id"], (list, tuple)) else str(batch["pat_id"])
        name = Path(pat_id).stem.replace(".", "_")

        # Baseline forward + saliency (masked to brain so hot spots are on tissue)
        pred_baseline, _ = forward_fn(backbone, sae, probe, x, intervene_z=None)
        sal_baseline = gradient_saliency(x, pred_baseline, input_vol_for_mask=input_np)
        pred_baseline_val = pred_baseline.detach().item()

        # Compute intervention so predicted age = true age (label) — floats for CSV
        intervene = compute_intervention_for_true_age(
            backbone, sae, probe, x, label, device, feature_ids, forward_fn=forward_fn
        )
        if intervene is None:
            print(f"  {name}: slopes degenerate, skipping adjusted saliency")
            intervene = {i: 0.0 for i in feature_ids}

        # Adjusted forward + saliency using *differentiable* intervention so gradient flows
        # through the updated feature values back to the image.
        intervene_tensor = compute_intervention_for_true_age_differentiable(
            backbone, sae, probe, x, label, device, feature_ids, forward_fn=forward_fn
        )
        if intervene_tensor is None:
            intervene_tensor = {
                i: torch.tensor(0.0, device=x.device).unsqueeze(0) for i in feature_ids
            }
        x.grad = None
        pred_adj, _ = forward_fn(backbone, sae, probe, x, intervene_z=intervene_tensor)
        sal_adjusted = gradient_saliency(x, pred_adj, input_vol_for_mask=input_np)
        pred_adj_val = pred_adj.detach().item()

        # Save NIfTI: input, saliency_baseline, saliency_adjusted
        affine = np.eye(4)
        nib.save(nib.Nifti1Image(input_np, affine), out_dir / f"{name}_input.nii.gz")
        nib.save(nib.Nifti1Image(sal_baseline.astype(np.float32), affine),
                 out_dir / f"{name}_saliency_baseline.nii.gz")
        nib.save(nib.Nifti1Image(sal_adjusted.astype(np.float32), affine),
                 out_dir / f"{name}_saliency_adjusted.nii.gz")

        # Save PNG: contour version (3 rows x 2 cols: before/after saliency)
        save_volume_views_png(
            out_dir / f"{name}_views.png",
            input_np,
            sal_baseline,
            sal_adjusted,
            title_prefix=f"{name}  |  true_age={label:.0f}  pred_baseline={pred_baseline_val:.0f}  pred_adj={pred_adj_val:.0f}",
        )
        # Save PNG: heatmap version (same layout, heatmap overlay instead of contours)
        save_volume_views_png_heatmap(
            out_dir / f"{name}_views_heatmap.png",
            input_np,
            sal_baseline,
            sal_adjusted,
            title_prefix=f"{name}  |  true_age={label:.0f}  pred_baseline={pred_baseline_val:.0f}  pred_adj={pred_adj_val:.0f}",
        )

        results.append({
            "pat_id": pat_id,
            "label_months": label,
            "pred_baseline": pred_baseline_val,
            "pred_adjusted": pred_adj_val,
            **{f"v_{i}": intervene[i] for i in feature_ids},
        })
        print(f"  {name}: true_age={label:.0f} pred_baseline={pred_baseline_val:.1f} pred_adj={pred_adj_val:.1f} (target=true)")
        n_done += 1

    # Save summary CSV
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(out_dir / "predictions_summary.csv", index=False)
    print(f"Saved {n_done} saliency maps and summary to {out_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
