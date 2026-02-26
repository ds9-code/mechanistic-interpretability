# MCI Classification

## Overview

MCI (Mild Cognitive Impairment) classification is framed as a binary classification objective for prediction of MCI status, using single sequence MR input.

### Task Details

- Modelling: Binary Classification
- Input: Single T1-weighted MRI scan
- Output: Binary prediction (0: healthy control, 1: MCI; subjects with MCI score > 0.5 )
- Metric: AUC, F1-score

### Input Format
- MRI Format: NIFTI (.nii.gz)
- Image Size: 96×96×96 voxels (automatically resized)
- Sequences: T1-weighted (single sequence)

### CSV Format

Your CSV file should contain the following columns:

```csv
pat_id,label
subject001,0
subject002,1
subject003,0
subject004,1
```

### Directory Structure
Format the data structure as mentioned below
```
data/
├── images/
│   ├── subject001.nii.gz
│   ├── subject002.nii.gz
│   ├── subject003.nii.gz
│   └── subject004.nii.gz
└── csvs/
    ├── mci_train.csv
    ├── mci_val.csv
    └── mci_test.csv
```

## Configuration

Change the configurations in `src/config_finetune.yml`:

```yaml
model:
  max_epochs: 200

data:
  size: [96, 96, 96]
  batch_size: 16
  num_workers: 4
  csv_file: "./data/csvs/mci_train.csv"
  val_csv: "./data/csvs/mci_val.csv"
  root_dir: "./data/images"

simclrvit:
  ckpt_path: "./checkpoints/BrainIAC.ckpt"

optim:
  lr: 0.001
  momentum: 0.9
  weight_decay: 0.0001

logger:
  save_dir: "./results/mci_checkpoints"
  save_name: "mci_model-epoch-{epoch:02d}-{val_auc:.2f}"
  run_name: "mci_experiment"
  project_name: "brainiac_v2_mci"

gpu:
  visible_device: "0"

train:
  freeze: "no"
```

## Training

```bash
python src/train_lightning_mci_stroke.py 
```

## Inference

Configure the inference task in `src/test_inference_finetune.py`:

```python
"mci_task": {
    "checkpoint_path": "./results/mci_checkpoints/mci_model-epoch-XX-val_auc-X.XX.ckpt",
    "test_csv_path": "./data/csvs/mci_test.csv",
    "root_dir": "./data/images",
    "output_csv_path": "./inference/model_output/mci_predictions.csv",
    "task_type": "classification",
    "image_type": "single",
    "num_classes": 1
}

DATASETS_TO_RUN = [
   "mci_task"
]
```

### Run Inference
```bash
python src/test_inference_finetune.py
```

### Generate Saliency Maps

update the filepaths in `src/generate_mci_stroke_vit_saliency.py`:

```python
nifti_path = "/path/to/your/single/image.nii.gz"  # Single image for saliency generation
checkpoint_path = "./results/mci_checkpoints/mci_model-epoch-XX-val_auc-X.XX.ckpt"
config_path = "./src/config_finetune.yml"
output_dir = "./inference/saliency_maps"
```

Then run the script:
```bash
python src/generate_mci_vit_saliency.py
```

### Fine-tuning Options
- **Linear Probing:** Set `train.freeze: "yes"` in config 