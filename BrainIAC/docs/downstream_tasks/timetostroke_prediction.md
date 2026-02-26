# Time to Stroke Prediction

## Overview

Time to stroke prediction is framed as a regression objective for prediction of time from stroke occurrence, using single sequence MR input.

### Task Details

- Modelling: Regression
- Input: Single T1-weighted MRI scan
- Output: Continuous time value (in days since stroke onset)
- Metric: MAE

### Input Format
- MRI Format: NIFTI (.nii.gz)
- Image Size: 96×96×96 voxels (automatically resized)
- Sequences: T1-weighted (single sequence)

### CSV Format

Your CSV file should contain the following columns:

```csv
pat_id,label
subject001,365.5
subject002,180.2
subject003,450.8
subject004,90.1
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
    ├── timetostroke_train.csv
    ├── timetostroke_val.csv
    └── timetostroke_test.csv
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
  csv_file: "./data/csvs/timetostroke_train.csv"
  val_csv: "./data/csvs/timetostroke_val.csv"
  root_dir: "./data/images"

simclrvit:
  ckpt_path: "./checkpoints/BrainIAC.ckpt"

optim:
  lr: 0.001
  momentum: 0.9
  weight_decay: 0.0001

logger:
  save_dir: "./results/timetostroke_checkpoints"
  save_name: "timetostroke_model-epoch-{epoch:02d}-{val_mae:.2f}"
  run_name: "timetostroke_experiment"
  project_name: "brainiac_v2_timetostroke"

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
"timetostroke_task": {
    "checkpoint_path": "./results/timetostroke_checkpoints/timetostroke_model-epoch-XX-val_mae-XX.XX.ckpt",
    "test_csv_path": "./data/csvs/timetostroke_test.csv",
    "root_dir": "./data/images",
    "output_csv_path": "./inference/model_output/timetostroke_predictions.csv",
    "task_type": "regression",
    "image_type": "single",
    "num_classes": 1
}

DATASETS_TO_RUN = [
   "timetostroke_task"
]
```

### Run Inference
```bash
python src/test_inference_finetune.py
```

### Generate Saliency Maps

update the filepaths in `src/generate_brainage_vit_saliency.py`:

```python
nifti_path = "/path/to/your/single/image.nii.gz"  # Single image for saliency generation
checkpoint_path = "./results/timetostroke_checkpoints/timetostroke_model-epoch-XX-val_mae-XX.XX.ckpt"
config_path = "./src/config_finetune.yml"
output_dir = "./inference/saliency_maps"
```

Then run the script:
```bash
python src/generate_timetostroke_vit_saliency.py
```

### Fine-tuning Options
- **Linear Probing:** Set `train.freeze: "yes"` in config
