# Overall Survival Prediction

## Overview

Overall survival prediction is framed as a binary classification objective for prediction of survival within 15 months, using multi sequence MR input.

### Task Details

- Modelling: Binary Classification
- Input: Quad-modal MRI scans (T1 + T1CE + T2 + FLAIR)
- Output: Binary prediction (0: short survival, 1: long survival)
- Metric: AUC, F1-score, Accuracy, Precision, Recall
- Dataset Class: `QuadImageDataset`
- Training Script: `src/train_lightning_os.py`

### Input Format
- MRI Format: NIFTI (.nii or .nii.gz)
- Image Size: 96×96×96 voxels (automatically resized)
- Sequences: T1, T1CE, T2, FLAIR (all four required)

### CSV Format

Your CSV file should contain the following columns:

```csv
pat_id,survival
subject001,1
subject002,0
subject003,1
subject004,0
```


### Directory Structure
Format the data structure as mentioned below
```
data/
├── images/
│   ├── subject001_t1n.nii.gz
│   ├── subject001_t1ce.nii.gz
│   ├── subject001_t2w.nii.gz
│   |── subject001_t2f.nii.gz
│   ├── subject002_t1n.nii.gz
│   ├── subject002_t1ce.nii.gz
│   ├── subject002_t2w.nii.gz
│   └── subject002_t2f.nii.gz
└── csvs/
    ├── survival_train.csv
    ├── survival_val.csv
    └── survival_test.csv
```

## Configuration

Change the configurations in `src/config_finetune.yml`:

```yaml
model:
  max_epochs: 200

data:
  size: [96, 96, 96]
  batch_size: 4
  num_workers: 4
  csv_file: "./data/csvs/survival_train.csv"
  val_csv: "./data/csvs/survival_val.csv"
  root_dir: "./data/images"

simclrvit:
  ckpt_path: "./checkpoints/BrainIAC.ckpt"

optim:
  lr: 0.001
  momentum: 0.9
  weight_decay: 0.0001

logger:
  save_dir: "./results/survival_checkpoints"
  save_name: "survival_model-epoch-{epoch:02d}-{val_auc:.2f}"
  run_name: "survival_experiment"
  project_name: "brainiac_v2_survival"

gpu:
  visible_device: "0"

train:
  freeze: "no"
```

## Training

```bash
python src/train_lightning_os.py 
```

## Inference

Configure the inference task in `src/test_inference_finetune.py`:

```python
"survival_task": {
    "checkpoint_path": "./results/survival_checkpoints/survival_model-epoch-XX-val_auc-X.XX.ckpt",
    "test_csv_path": "./data/csvs/survival_test.csv",
    "root_dir": "./data/images",
    "output_csv_path": "./inference/model_output/survival_predictions.csv",
    "task_type": "classification",
    "image_type": "quad",
    "num_classes": 1
}

DATASETS_TO_RUN = [
   "survival_task"
]
```

### Run Inference
```bash
python src/test_inference_finetune.py
```

### Generate Saliency Maps

update the filepaths in `src/generate_os_vit_saliency.py`:

```python
# Update these paths in the script before running
nifti_path = "/path/to/your/single/image.nii.gz"  # Single image for saliency generation
checkpoint_path = "./results/survival_checkpoints/survival_model-epoch-XX-val_auc-X.XX.ckpt"
config_path = "./src/config_finetune.yml"
output_dir = "./inference/saliency_maps"
```

Then run the script:
```bash
python src/generate_os_vit_saliency.py
```

This generates ViT attention maps showing which brain regions are most important for survival prediction. The script processes a single image at a time and requires updating the hardcoded paths for each image you want to analyze.

### Fine-tuning Options
- **Linear Probing:** Set `train.freeze: "yes"` in config

