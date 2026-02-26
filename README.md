# Mechanistic Interpretability for Brain Age Prediction

**Interpretable brain age from sparse autoencoder features—decompose a pretrained MRI encoder, probe for age, intervene on a small set of features to match true age, and validate with saliency maps.**

---

> **Scope:** This repository focuses on **brain age prediction** (predicting chronological or biological age in months from 3D T1-weighted brain MRI). We use a **sparse autoencoder (SAE)** to decompose pretrained encoder features into interpretable units, then **intervention** and **gradient saliency** to validate that a handful of SAE features can align predictions with ground-truth age and to visualize where in the brain the model “looks.”

---

## Aims and Motivation

**Brain age** is a summary metric derived from brain MRI: a model is trained to predict age (in months) from imaging. It is used in developmental and aging research. Such models are often black boxes: we get a number but not *which* image regions or *which* internal features drove the prediction.

**Our goals:**

1. **Decompose** the representation of a pretrained brain MRI encoder (Vision Transformer) into a large set of **sparse** features via an SAE, so we can reason about individual “units” instead of a monolithic 768-d vector.
2. **Predict age** from these SAE features using a simple **linear probe**, so the link from features to age is interpretable.
3. **Identify** a small set of SAE features that are “meaningful” for age (vs “noise” that hurts performance when present).
4. **Intervene** on those features so that, for any given scan, the predicted age equals the **true age** (ground truth). This shows that the same pipeline can produce the correct answer when we override a few dimensions.
5. **Validate with saliency:** Compute **gradient-based saliency maps** (where in the image the prediction is sensitive) **before** and **after** intervention on OASIS data, and visualize them (contours and heatmaps) to see whether the model’s focus aligns with brain anatomy.

No prior knowledge is assumed beyond: we have 3D brain MRIs, age labels, and a pretrained encoder that produces a 768-d CLS token per scan.

---

## Data Sources

| Data | Role |
|------|------|
| **Pretrained encoder** | A **SimCLR** ViT backbone trained on brain MRI (e.g. from Sophont checkpoints). It takes 3D volumes (e.g. 96×96×96) and outputs a 768-d embedding (CLS token). |
| **Training data** | CSV-specified train/val/test splits with paths to NIfTI volumes and age labels in months. Used for SAE training, feature extraction, and linear probe training. |
| **OASIS** | [OASIS](https://www.oasis-brains.org/) brain MRI data. We use a small set of OASIS scans (e.g. 5) with known ages for **saliency validation**: we run the full pipeline (encoder → SAE → probe), intervene so predicted age = true age, and save saliency maps (before/after) and predictions. Paths are set via `--test_csv` and `--root_dir` (e.g. `data/csvs/oasis1_only.csv`, `oasis_data/oasis1/data`). |

All imaging is 3D T1-weighted MRI; ages are in **months**.

---

## Table of Contents

* [Overview](#overview)
* [Features](#features)
* [Pipeline at a Glance](#pipeline-at-a-glance)
* [The Encoder and SAE](#the-encoder-and-sae)
* [Feature Identification](#feature-identification)
* [Linear Probe](#linear-probe)
* [Intervention (Match True Age)](#intervention-match-true-age)
* [Saliency Validation on OASIS](#saliency-validation-on-oasis)
* [Quick Start](#quick-start)
* [Repository Structure and Outputs](#repository-structure-and-outputs)
* [Troubleshooting](#troubleshooting)

---

## Overview

This repository enables:

* **SAE decomposition** of the pretrained brain MRI encoder (CLS) features into a high-dimensional sparse latent (e.g. ~49k dimensions), with an MLP-grounded SAE that can be trained with reconstruction and optional age-preservation loss.
* **Linear age probe** trained on **SAE-decoded** 768-d features (not raw CLS), so age is predicted from the reconstructed representation.
* **Feature analysis** (activation statistics, consistency across images, noise vs meaningful via **masking experiments**).
* **Targeted intervention** on a small set of SAE features so that predicted age = true age (minimum-norm solution), with **differentiable** intervention for gradient-based saliency.
* **Saliency validation** on OASIS: gradient saliency (∂pred/∂image) before and after intervention, with BrainIAC-style visualization (Gaussian blur, magma colormap, contour and heatmap overlays).

---

## Features

* 🧠 **Interpretable age:** Linear probe on SAE-decoded features; a handful of features (e.g. 3) are adjusted so prediction equals true age.
* 📊 **Sparse latent:** SAE expands 768-d CLS into a sparse high-dimensional latent (e.g. top-k or L1), then decodes back to 768-d for the probe.
* 🔬 **Feature identification:** Statistical analysis (activation rate, variance, consistency) and masking-based separation of “noise” (MAE improves when masked) vs “meaningful” (MAE degrades when masked).
* 🎯 **Minimum-norm intervention:** Closed-form solution so predicted age = true age with smallest possible change in the chosen feature dimensions.
* 📈 **Differentiable intervention:** Intervention values are computed as tensors so gradient flows through them for the “after” saliency map.
* 🗺️ **Saliency maps:** Patch-averaged gradient saliency, brain-masked, with contour and heatmap PNGs plus NIfTI volumes for external use.
* 📁 **Self-contained runs:** Scripts for feature extraction, probe training, masking experiments, and saliency validation with clear paths and CSV/JSON outputs.

---

## Pipeline at a Glance

| Step | What happens |
|------|----------------|
| **1. Encoder** | 3D MRI → pretrained SimCLR ViT → 768-d CLS token. |
| **2. SAE** | CLS → SAE encoder → sparse latent **z** (e.g. 49,152 dims) → SAE decoder → 768-d reconstructed features. |
| **3. Probe** | Reconstructed 768-d → linear layer → scalar **age (months)**. |
| **4. Feature selection** | Offline: statistics + masking to choose a small set of feature indices (e.g. 857, 17827, 1796). |
| **5. Intervention** | For a given scan and true age: set chosen z indices so predicted age = true age (minimum-norm formula). |
| **6. Saliency** | Compute ∂(pred)/∂(image) for baseline and for adjusted prediction (with differentiable intervention); visualize on OASIS. |

---

## The Encoder and SAE

* **Encoder:** SimCLR ViT backbone (e.g. from Sophont). Input: 3D volume 96×96×96. Output: 768-d CLS token.
* **SAE:** MLP-grounded sparse autoencoder (`new_sae_code/sae_adapter`): encoder → sparsity (e.g. top-k or L1) → latent **z** (e.g. 49,152 dims) → decoder → 768-d. Can be trained with reconstruction loss and optional age-preservation loss.
* **Probe:** Linear regression on SAE-decoded 768-d (e.g. `linear_probe_results_new_sae_age/best_model.pt`).
* **Intervention features:** A small set of indices chosen from analysis (e.g. 857, 17827, 1796). Override with `--feature_ids` if needed.

**Data flow:**

```
Image → SimCLR → CLS → SAE(encode, sparsity) → z
       → [intervene: set chosen z indices] → decode → probe → age
```

---

## Feature Identification

Before we can “set a few features to match true age,” we need to decide *which* features to use.

### Statistical analysis

* **Scripts:** e.g. `sae_statistical_analysis/generate_statistical_analysis.py`, plus any that consume test set activations.
* **Outputs:** Per-feature statistics (activation rate, mean/max activation, variance), which images activate which features (`feature_to_images.json`), and lists of **consistent** or **noise** features (`consistent_features.json`, `noise_features.json`).

### Noise vs meaningful (masking)

* **Idea:** For each candidate feature, **mask** it (set to 0) at test time and measure the change in age prediction MAE.
* **Interpretation:** If MAE **improves** when the feature is masked → the feature is treated as **noise** (it was hurting accuracy). If MAE **worsens** → the feature is **meaningful** (it was helping).
* **Usage:** Results (e.g. `true_noise_features_identified.json`, masking experiment JSONs) are used to pick a small set of **meaningful** features for intervention (e.g. 857, 17827, 1796).

---

## Linear Probe

* **Input:** SAE-decoded 768-d vector (same dimensionality as the encoder CLS).
* **Target:** Brain age in **months** (regression).
* **Training:** Standard linear regression (e.g. `BrainIAC/src/train_linear_probe.py`) on pre-extracted features from train/val CSVs. No gradient through the encoder or SAE during probe training.
* **Evaluation:** Baseline MAE (no intervention); after intervention, prediction is forced to true age by design, so the “accuracy” of the adjusted prediction is not the main metric—saliency and interpretability are.

---

## Intervention (Match True Age)

We want predicted age to equal true age by changing only a small set of SAE latent dimensions.

**Notation:** Let \( \mathcal{F} \) be the set of feature indices (e.g. {857, 17827, 1796}). Let \( \hat{y} \) be the predicted age and \( y_{\text{true}} \) the ground truth.

**Steps:**

1. **Prediction at zero:** Run the forward pass with \( z_i = 0 \) for all \( i \in \mathcal{F} \). Call this prediction \( \hat{y}_0 \).
2. **Slopes:** For each \( i \in \mathcal{F} \), run forward with \( z_i = 1 \) and \( z_j = 0 \) for \( j \neq i \). Define slope \( s_i = \hat{y}_i - \hat{y}_0 \).
3. **Minimum-norm adjustment:** We want \( \hat{y}_0 + \sum_{i \in \mathcal{F}} s_i v_i = y_{\text{true}} \). Set \( \Delta = y_{\text{true}} - \hat{y}_0 \) and \( v_i = \Delta \cdot s_i / \sum_j s_j^2 \). Then replacing \( z_i \) with \( v_i \) gives predicted age = true age with smallest \( \ell^2 \) change in those dimensions.

**Differentiable version:** The same formula is implemented with tensors (no `torch.no_grad()` on the intervention computation) so that when we backprop through the “adjusted” prediction, gradient flows through the \( v_i \) back to the image. That way the “after” saliency map reflects sensitivity to the image *including* through the intervention.

---

## Saliency Validation on OASIS

**Script:** `BrainIAC/src/saliency_oasis_validation.py`

### What it does

* Loads a few OASIS 3D volumes (e.g. 5) and their age labels.
* For each volume:
  * **Baseline:** Forward pass with no intervention → compute gradient saliency \( \partial(\text{pred})/\partial(\text{image}) \), patch-averaged and (optionally) brain-masked.
  * **Adjusted:** Compute intervention so predicted age = true age; run forward with **differentiable** intervention → compute gradient saliency for the adjusted prediction.
* **Visualization:** Saliency is Gaussian-blurred, max-normalized, and displayed with the **magma** colormap (BrainIAC quickstart style). Contour overlays and heatmap overlays are both saved.

### Outputs per run

| Output | Description |
|--------|-------------|
| `*_input.nii.gz` | Input 3D volume. |
| `*_saliency_baseline.nii.gz` | Saliency volume (baseline prediction). |
| `*_saliency_adjusted.nii.gz` | Saliency volume (adjusted prediction). |
| `*_views.png` | Contour overlay (axial/coronal/sagittal × before/after). |
| `*_views_heatmap.png` | Heatmap overlay (same layout). |
| `predictions_summary.csv` | True age, baseline pred, adjusted pred, intervention values per subject. |

### Saliency processing details

* Gradients are averaged over 16³ patches (ViT patch size) then upsampled back to volume size.
* Saliency can be masked to a brain region (e.g. intensity above a percentile) so hot spots stay on tissue.
* For display: 1st/99th percentile normalization for MRI; saliency is blurred (Gaussian, σ≈2.5), clipped to non-negative, max-normalized; contours use 10 levels, linewidth 1; heatmap uses alpha blending.

---

## Quick Start

### Prerequisites

* Python environment with PyTorch, MONAI, nibabel, pandas, scipy, matplotlib.
* `PYTHONPATH` must include Sophont and `new_sae_code/sae_adapter` so the saliency script can load the SimCLR backbone and the SAE adapter.
* Checkpoints: SimCLR backbone, SAE checkpoint (e.g. `new_sae_checkpoints_brainage/last.ckpt`), and linear probe (e.g. `linear_probe_results_new_sae_age/best_model.pt`).
* OASIS (or similar) CSV and root directory for a few test scans.

### Run saliency validation

```bash
cd /path/to/mechinterp
PYTHONPATH="/path/to/sophont:/path/to/mechinterp/new_sae_code/sae_adapter:$PYTHONPATH" \
  python BrainIAC/src/saliency_oasis_validation.py \
  --new_sae_checkpoint new_sae_checkpoints_brainage/last.ckpt \
  --linear_probe linear_probe_results_new_sae_age/best_model.pt \
  --test_csv data/csvs/oasis1_only.csv \
  --root_dir oasis_data/oasis1/data \
  --output_dir linear_probe_results_new_sae_age/saliency_oasis_validation \
  --n_images 5
```

Default intervention features are 857, 17827, 1796. Override with `--feature_ids` if needed (e.g. `--feature_ids 857,17827,1796`).

---

## Repository Structure and Outputs

| Path | Description |
|------|-------------|
| `BrainIAC/src/` | Dataset, loaders, **saliency_oasis_validation.py**, feature extraction, probe training, masking scripts. |
| `new_sae_code/sae_adapter/` | SAE model (SAEAdapterMLPGrounded) and config. |
| `sae_statistical_analysis/` | Scripts and JSONs for feature statistics, consistent/noise features. |
| `linear_probe_results_new_sae_age/` | Linear probe weights and **saliency OASIS outputs** (contour and heatmap PNGs, NIfTIs, CSV). |
| `new_sae_checkpoints_brainage/` | SAE checkpoints (reconstruction and optional age-preservation training). |

---

## Troubleshooting

### Saliency maps are blank

* **Cause:** Saliency volumes can contain NaNs outside the brain mask; max-normalization then fails. **Fix:** The pipeline uses `np.nanmax` and `np.nan_to_num(..., nan=0)` before blur so normalization and contouring see finite values.

### Before and after saliency look almost the same

* **Expected when intervention is not differentiable:** If the intervention values are constants, no gradient flows through them, so ∂(pred_adj)/∂(image) is the same path as baseline. **Fix:** Use the **differentiable** intervention (default in this repo): intervention values are tensors so the “after” saliency can differ.

### “Slopes degenerate, skipping adjusted saliency”

* **Cause:** The slopes \( s_i \) for the chosen features are all near zero or the denominator \( \sum_j s_j^2 \) is too small. **Fix:** Choose a different set of features (e.g. from masking/statistical analysis) or check that the probe and SAE are compatible with the backbone.

### Module not found (e.g. `simclr`, `model_mlp_grounded`)

* **Cause:** Scripts assume `PYTHONPATH` includes Sophont and `new_sae_code/sae_adapter`. **Fix:** Set `PYTHONPATH` as in the Quick Start before running the saliency script.

### OASIS paths or CSV

* **Cause:** `--test_csv` and `--root_dir` must point to a CSV with columns usable by the dataset (e.g. path and age) and to the directory where NIfTI files live. **Fix:** Adjust paths for your clone; see `data/csvs/oasis1_only.csv` and `oasis_data/oasis1/data` as reference layout.

---

## License and Contributing

See the repository root for license and contributing guidelines. For methodology details and equations, this document is the single source of truth.
