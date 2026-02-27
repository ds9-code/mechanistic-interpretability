# Mechanistic Interpretability for Brain Age Prediction

**Summary: Brain age prediction uses MRI to estimate how old a brain looks compared to a person’s real age. That number acts as a biomarker of brain health: brains that look older than expected can signal accelerated aging or higher risk for age-related disease, and researchers use it to study development, disease, and lifestyle. Models often predict brain age as a single number without explaining how they reach it. This project makes those models more interpretable: it breaks the model into understandable parts, shows which internal features drive the age prediction, intervenes on those features to correct wrong predictions, and visualizes which brain regions the model focuses on before and after correction. In healthcare, this kind of transparency matters: clinicians and patients need to understand how AI reaches its conclusions, and regulators need ways to hold AI accountable. More interpretable brain age models help detect bias, debug errors, and build trust, which is important for safely using AI in medical imaging.**

---

> **Scope:** This repository focuses on brain age prediction (predicting chronological or biological age in months from 3D T1-weighted brain MRI). We use a **sparse autoencoder (SAE)** to decompose pretrained encoder features into interpretable units, then intervention and gradient saliency to validate that a handful of SAE features can align predictions with ground-truth age and to visualize where in the brain the model “looks.”

**Context: BrainIAC.** [BrainIAC](https://github.com/AIM-KannLab/BrainIAC) is a foundation model for brain MRI: a Vision Transformer (ViT) pretrained on large-scale neuroimaging for downstream tasks (segmentation, classification, survival prediction, etc.). This repository builds on that ecosystem: we use a SimCLR-trained ViT backbone (compatible with the BrainIAC pipeline) as the encoder and add an SAE plus an interpretability stack for brain age. The `BrainIAC/` folder here holds dataset utilities, loaders, and scripts (saliency, feature extraction, probe training) that work with the same family of pretrained brain MRI encoders.

---

## Aims and Motivation

**Brain age** is a summary metric derived from brain MRI: a model is trained to predict age (in months) from imaging. It is used in developmental and aging research. Such models are often black boxes: we get a number but not *which* image regions or *which* internal features drove the prediction.

**Our goals:**

1. **Decompose** the representation of a pretrained brain MRI encoder (Vision Transformer) into a large set of sparse features via an SAE, so we can reason about individual “units” instead of a monolithic 768-d vector.
2. **Predict age** from these SAE features using a simple linear probe, so the link from features to age is interpretable.
3. **Identify** a small set of SAE features that are “meaningful” for age (vs “noise” that hurts performance when present).
4. **Intervene** on those features so that, for any given scan, the predicted age equals the true age (ground truth). This shows that the same pipeline can produce the correct answer when we override a few dimensions.
5. **Validate with saliency:** Compute gradient-based saliency maps (where in the image the prediction is sensitive) before and after intervention on OASIS data, and visualize them (contours and heatmaps) to see whether the model’s focus aligns with brain anatomy.
6. **AI safety and interpretability in healthcare:** Apply interpretability techniques (sparse features, linear probes, targeted intervention, saliency) to move from black-box brain-age models to explainable ones - so we can say *which* internal features and *where* in the image drive the prediction. In applied healthcare, interpretable models support clinical trust, accountability, and safer deployment; reducing reliance on opaque black-box predictions is important for responsible use of AI in medical imaging.

---

## Data Sources

| Data | Role |
|------|------|
| **Pretrained encoder** | A SimCLR ViT backbone trained on brain MRI (e.g. from Sophont checkpoints). It takes 3D volumes (e.g. 96×96×96) and outputs a 768-d embedding (CLS token). |
| **Training data** | CSV-specified train/val/test splits with paths to NIfTI volumes and age labels in months. Used for SAE training, feature extraction, and linear probe training. |
| **OASIS** | [OASIS](https://www.oasis-brains.org/) brain MRI data. We use a small set of OASIS scans (e.g. 5) with known ages for saliency validation: we run the full pipeline (encoder → SAE → probe), intervene so predicted age = true age, and save saliency maps (before/after) and predictions. Paths are set via `--test_csv` and `--root_dir` (e.g. `data/csvs/oasis1_only.csv`, `oasis_data/oasis1/data`). |

All imaging is 3D T1-weighted MRI; ages are in months.

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
* [Results](#results)
* [Quick Start](#quick-start)
* [Repository Structure and Outputs](#repository-structure-and-outputs)

---

## Overview

This repository enables:

* **SAE decomposition** of the pretrained brain MRI encoder (CLS) features into a high-dimensional sparse latent (e.g. ~49k dimensions), with an MLP-grounded SAE that can be trained with reconstruction and optional age-preservation loss.
* **Linear age probe** trained on SAE-decoded 768-d features (not raw CLS), so age is predicted from the reconstructed representation.
* **Feature analysis** (activation statistics, consistency across images, noise vs meaningful via masking experiments).
* **Targeted intervention** on a small set of SAE features so that predicted age = true age (minimum-norm solution), with differentiable intervention for gradient-based saliency.
* **Saliency validation** on OASIS: gradient saliency (∂pred/∂image) before and after intervention, with BrainIAC-style visualization (Gaussian blur, magma colormap, contour and heatmap overlays).

---

## Features

* **Interpretable age:** Linear probe on SAE-decoded features; a handful of features (e.g. 3) are adjusted so prediction equals true age.
* **Sparse latent:** SAE expands 768-d CLS into a sparse high-dimensional latent (e.g. top-k or L1), then decodes back to 768-d for the probe.
* **Feature identification:** Statistical analysis (activation rate, variance, consistency) and masking-based separation of “noise” (MAE improves when masked) vs “meaningful” (MAE degrades when masked).
* **Minimum-norm intervention:** Closed-form solution so predicted age = true age with smallest possible change in the chosen feature dimensions.
* **Differentiable intervention:** Intervention values are computed as tensors so gradient flows through them for the “after” saliency map.
* **Saliency maps:** Patch-averaged gradient saliency, brain-masked, with contour and heatmap PNGs plus NIfTI volumes for external use.
* **Self-contained runs:** Scripts for feature extraction, probe training, masking experiments, and saliency validation with clear paths and CSV/JSON outputs.

---

## Pipeline at a Glance

| Step | What happens |
|------|----------------|
| **1. Encoder** | 3D MRI → pretrained SimCLR ViT → 768-d CLS token. |
| **2. SAE** | CLS → SAE encoder → sparse latent **z** (e.g. 49,152 dims) → SAE decoder → 768-d reconstructed features. |
| **3. Probe** | Reconstructed 768-d → linear layer → scalar age (months). |
| **4. Feature selection** | Offline: statistics + masking to choose a small set of feature indices (e.g. 9607, 8700, 23673). |
| **5. Intervention** | For a given scan and true age: set chosen z indices so predicted age = true age (minimum-norm formula). |
| **6. Saliency** | Compute ∂(pred)/∂(image) for baseline and for adjusted prediction (with differentiable intervention); visualize on OASIS. |

---

## The Encoder and SAE

* **Encoder:** SimCLR ViT backbone (e.g. from Sophont). Input: 3D volume 96×96×96. Output: 768-d CLS token.
* **SAE:** MLP-grounded sparse autoencoder (`new_sae_code/sae_adapter`): encoder → sparsity (e.g. top-k or L1) → latent **z** (e.g. 49,152 dims) → decoder → 768-d. Can be trained with reconstruction loss and optional age-preservation loss.
* **Probe:** Linear regression on SAE-decoded 768-d (e.g. `linear_probe_results_new_sae_age/best_model.pt`).
* **Intervention features:** A small set of indices chosen from analysis (e.g. 9607, 8700, 23673). Override with `--feature_ids` if needed.

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
* **Outputs:** Per-feature statistics (activation rate, mean/max activation, variance), which images activate which features (`feature_to_images.json`), and lists of consistent or noise features (`consistent_features.json`, `noise_features.json`).

### Noise vs meaningful (masking)

* **Idea:** For each candidate feature, mask it (set to 0) at test time and measure the change in age prediction mean absolute error (MAE).
* **Interpretation:** If MAE improves when the feature is masked → the feature is treated as noise (it was hurting accuracy). If MAE worsens → the feature is meaningful (it was helping).
* **Usage:** Results (e.g. `true_noise_features_identified.json`, masking experiment JSONs) are used to pick a small set of meaningful features for intervention (e.g. 9607, 8700, 23673).

---

## Linear Probe

* **Input:** SAE-decoded 768-d vector (same dimensionality as the encoder CLS).
* **Target:** Brain age in months (regression).
* **Training:** Standard linear regression (e.g. `BrainIAC/src/train_linear_probe.py`) on pre-extracted features from train/val CSVs. No gradient through the encoder or SAE during probe training.
* **Evaluation:** Baseline MAE (no intervention); after intervention, prediction is forced to true age by design, so the “accuracy” of the adjusted prediction is not the main metric-saliency and interpretability are.

---

## Intervention (Match True Age)

We want predicted age to equal true age by changing only a small set of SAE latent dimensions.

**Notation:** Let$\mathcal{F}$be the set of feature indices (e.g. {9607, 8700, 23673}). Let$\hat{y}$be the predicted age and$y_{\text{true}}$the ground truth.

### Steps

1. **Prediction at zero:**  
   Run the forward pass with zᵢ = 0 for all i in F.  
   Call this prediction ŷ₀.

2. **Slopes:**  
   For each i in F, run forward with zᵢ = 1 and zⱼ = 0 for j ≠ i.  
   Define slope sᵢ = ŷᵢ − ŷ₀.

3. **Minimum-norm adjustment:**  

   We want:

   ŷ₀ + ∑ (sᵢ · vᵢ) = y_true

   Set:

   Δ = y_true − ŷ₀

   vᵢ = (Δ · sᵢ) / ∑ sⱼ²

   Replacing zᵢ with vᵢ gives predicted age = true age with the smallest ℓ² change in those dimensions.

**Differentiable version:** The same formula is implemented with tensors (no `torch.no_grad()` on the intervention computation) so that when we backprop through the “adjusted” prediction, gradient flows through the$v_i$back to the image. That way the “after” saliency map reflects sensitivity to the image *including* through the intervention.

---

## Saliency Validation on OASIS

**Script:** `BrainIAC/src/saliency_oasis_validation.py`

### What it does

* Loads a few OASIS 3D volumes (e.g. 5) and their age labels.
* For each volume:
  * **Baseline:** Forward pass with no intervention → compute gradient saliency (d(ŷ) / d(image)), patch-averaged and (optionally) brain-masked.
  * **Adjusted:** Compute intervention so predicted age = true age; run forward with differentiable intervention → compute gradient saliency for the adjusted prediction.
* **Visualization:** Saliency is Gaussian-blurred, max-normalized, and displayed with the magma colormap (BrainIAC quickstart). Contour overlays and heatmap overlays are both saved.

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

## Results 
<img width="800" height="600" alt="feature_adjustment" src="https://github.com/user-attachments/assets/06197c10-2798-482b-a799-0f743b93fd60" />

Feature impact: Describes the scatter and box plots as sweeping activation levels (negative, zero, median, max, above_max) for a few SAE features and states that higher activation increases predicted brain age, so those features are meaningfully linked to age.

<img width="800" height="600" alt="saliency_maps" src="https://github.com/user-attachments/assets/6c4661c7-79bb-48ec-96f5-24d4cc3e0dc3" />

Saliency before vs after: Gradient saliency for baseline and for the adjusted prediction; “before” vs “after” contrast and the interpretation that the model’s focus becomes more localized and anatomically plausible after correction. The “before” seems to be more diffuse, “after” more focused on grey matter and cortex when the prediction is corrected to true age.



Together the figures show which features drive age and where in the image the model looks before and after intervention.

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

Default intervention features are 9607, 8700, 23673. Override with `--feature_ids` if needed (e.g. `--feature_ids 9607, 8700, 23673`).

---

## Repository Structure and Outputs

| Path | Description |
|------|-------------|
| `BrainIAC/src/` | Dataset, loaders, **saliency_oasis_validation.py**, feature extraction, probe training, masking scripts. |
| `new_sae_code/sae_adapter/` | SAE model (SAEAdapterMLPGrounded) and config. |
| `sae_statistical_analysis/` | Scripts and JSONs for feature statistics, consistent/noise features. |
| `linear_probe_results_new_sae_age/` | Linear probe weights and **saliency OASIS outputs** (contour and heatmap PNGs, NIfTIs, CSV). |
| `new_sae_checkpoints_brainage/` | SAE checkpoints (reconstruction and optional age-preservation training). |
| `figures/` | **Result figures for the repo:** copy key saliency PNGs or plots here and reference them from the README or [Results](#results-example-figures) so GitHub renders them. |
