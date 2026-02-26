# Brain Age Prediction

## Overview

Brain age prediction is formulated as a regression objective to estimate chronological age from T1-weighted MRI scans. 

### Task Details

-  Modelling: Regression
-  Input: Single T1-weighted MRI scan
-  Output type: Continuous age value (in months)
-  Metric: Mean Absolute Error (MAE)
- Dataset Class: `BrainAgeDataset`
- Training Script: `src/train_lightning_brainage.py`


### Input Format
- MRI Format: NIFTI (.nii.gz)
- Preprocessing: Bias field corrected, registered to standard space, skull stripped
- Image Size: 96×96×96 voxels (automatically resized)

### CSV Format

Your CSV file should contain the following columns:

```csv
pat_id,label
subject001,65.5
subject002,72.3
subject003,45.2
```


### Directory Structure

```
data/
├── images/
│   ├── subject001.nii.gz
│   └── subject002.nii.gz
|
└── csvs/
    ├── brainage_train.csv
    ├── brainage_val.csv
    └── brainage_test.csv
```

## Configuration

Change the configurations `src/config_finetune.yml`:

```yaml
model:
  max_epochs: 200

data:
  size: [96, 96, 96]
  batch_size: 16
  num_workers: 4
  csv_file: "./data/csvs/brainage_train.csv"
  val_csv: "./data/csvs/brainage_val.csv"
  root_dir: "./data/images"

simclrvit:
  ckpt_path: "./checkpoints/BrainIAC.ckpt"

optim:
  lr: 0.001
  momentum: 0.9
  weight_decay: 0.0001

logger:
  save_dir: "./results/brainage_checkpoints"  # checkpoints logging dir
  save_name: "brainage_model-epoch-{epoch:02d}-{val_mae:.2f}"
  run_name: "brainage_experiment"            # wandb cofigs 
  project_name: "brainiac_v2_brainage"

gpu:
  visible_device: "0"

train:
  freeze: "no"
```

## Training
```bash
python src/train_lightning_brainage.py 
```

## Inference

Configure the inference task in `src/test_inference_finetune.py`:

```python
"brainage_task": {
    "checkpoint_path": "./results/brainage_checkpoints/brainage_model-epoch-XX-val_mae-XX.XX.ckpt",
    "test_csv_path": "./data/csvs/brainage_test.csv",
    "root_dir": "./data/images",
    "output_csv_path": "./inference/model_output/brainage_predictions.csv",
    "task_type": "regression",
    "image_type": "single",
    "num_classes": 1
}

DATASETS_TO_RUN = [
   "brainage_task"
]
```

### Run Inference
```bash
python src/test_inference_finetune.py
```


### Generate Saliency Maps
```bash
python src/generate_brainage_vit_saliency.py
```
This generates attention maps showing which brain regions are most important for age prediction.


### Fine-tuning Options
- **Linear Probing:** Set `train.freeze: "yes"` in config

