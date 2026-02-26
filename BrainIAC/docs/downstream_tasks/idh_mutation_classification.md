# IDH Mutation Classification

## Overview

IDH mutation classification is framed as a binary classification objective for prediction of IDH mutation status, using dual sequence MR input.

### Task Details

- Modelling: Binary Classification
- Input: Dual-modal MRI scans (FLAIR + T1CE)
- Output: Binary prediction (0: IDH wild-type, 1: IDH mutant)
- Metric: AUC, F1-score
- Dataset Class: `DualImageDataset`
- Training Script: `src/train_lightning_dual.py`

### Input Format
- MRI Format: NIFTI (.nii.gz)
- Image Size: 96×96×96 voxels (automatically resized)
- Sequences: FLAIR and T1CE (both required)

### CSV Format

Your CSV file should contain the following columns:

```csv
pat_id,label
subject001,0
subject002,1
subject003,0
subject004,1
```

**Column Descriptions:**
- `pat_id`: Patient identifier (string)
- `label`: IDH mutation status (0: wild-type, 1: mutant)

### Directory Structure
Format the data structure as mentioned below
```
data/
├── images/
│   ├── subject001_t2f.nii.gz
│   ├── subject001_t1ce.nii.gz
│   ├── subject002_t2f.nii.gz
│   ├── subject002_t1ce.nii.gz
│   ├── subject003_t2f.nii.gz
│   ├── subject003_t1ce.nii.gz
│   ├── subject004_t2f.nii.gz
│   └── subject004_t1ce.nii.gz
└── csvs/
    ├── idh_train.csv
    ├── idh_val.csv
    └── idh_test.csv
```

## Configuration

Change the configurations in `src/config_finetune.yml`:

```yaml
model:
  max_epochs: 200

data:
  size: [96, 96, 96]
  batch_size: 8
  num_workers: 4
  csv_file: "./data/csvs/idh_train.csv"
  val_csv: "./data/csvs/idh_val.csv"
  root_dir: "./data/images"

simclrvit:
  ckpt_path: "./checkpoints/BrainIAC.ckpt"

optim:
  lr: 0.001
  momentum: 0.9
  weight_decay: 0.0001

logger:
  save_dir: "./results/idh_checkpoints"
  save_name: "idh_model-epoch-{epoch:02d}-{val_auc:.2f}"
  run_name: "idh_experiment"
  project_name: "brainiac_v2_idh"

gpu:
  visible_device: "0"

train:
  freeze: "no"
```

## Training

```bash
python src/train_lightning_dual.py 
```

## Inference

Configure the inference task in `src/test_inference_finetune.py`:

```python
"idh_task": {
    "checkpoint_path": "./results/idh_checkpoints/idh_model-epoch-XX-val_auc-X.XX.ckpt",
    "test_csv_path": "./data/csvs/idh_test.csv",
    "root_dir": "./data/images",
    "output_csv_path": "./inference/model_output/idh_predictions.csv",
    "task_type": "classification",
    "image_type": "dual",
    "num_classes": 1
}

DATASETS_TO_RUN = [
   "idh_task"
]
```

### Run Inference
```bash
python src/test_inference_finetune.py
```

### Generate Saliency Maps

update the filepaths in `src/generate_idh_vit_saliency.py`:

```python
nifti_path = "/path/to/your/single/image.nii.gz"  # Single image for saliency generation
checkpoint_path = "./results/idh_checkpoints/survival_model-epoch-XX-val_auc-X.XX.ckpt"
config_path = "./src/config_finetune.yml"
output_dir = "./inference/saliency_maps"
```

Then run the script:
```bash
python src/generate_idh_vit_saliency.py
```

### Fine-tuning Options
- **Linear Probing:** Set `train.freeze: "yes"` in config 