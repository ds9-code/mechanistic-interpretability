# MRI Sequence Classification

## Overview

MRI sequence classification is framed as a multi-class classification objective for prediction of MRI sequence type, using single sequence MR input.

### Task Details

- Modelling: Multi-class Classification
- Input: Single MRI scan (any sequence)
- Output: Sequence type (0: T1, 1: T2, 2: FLAIR, 3: T1CE)
- Metric: Balanced Accuracy

### Input Format
- MRI Format: NIFTI (.nii.gz)
- Image Size: 96×96×96 voxels (automatically resized)
- Sequences: T1, T2, FLAIR, T1CE (single sequence input)

### CSV Format

Your CSV file should contain the following columns:

```csv
PatientID,label
subject001,1
subject002,2
subject003,3
subject004,4
```


### Directory Structure
Format the data structure as mentioned below
```
data/
├── images/
│   ├── subject001.nii.gz
│   ├── subject002.nii.gz
│   ├── subject003.nii.gz
│   ├── subject004.nii.gz
│   └── subject005.nii.gz
└── csvs/
    ├── sequence_train.csv
    ├── sequence_val.csv
    └── sequence_test.csv
```

## Configuration

Change the configurations in `src/config_finetune.yml`:

```yaml
model:
  max_epochs: 200

data:
  size: [96, 96, 96]
  batch_size: 32
  num_workers: 4
  csv_file: "./data/csvs/sequence_train.csv"
  val_csv: "./data/csvs/sequence_val.csv"
  root_dir: "./data/images"

simclrvit:
  ckpt_path: "./checkpoints/BrainIAC.ckpt"

optim:
  lr: 0.001
  momentum: 0.9
  weight_decay: 0.0001

logger:
  save_dir: "./results/sequence_checkpoints"
  save_name: "sequence_model-epoch-{epoch:02d}-{val_acc:.2f}"
  run_name: "sequence_experiment"
  project_name: "brainiac_v2_sequence"

gpu:
  visible_device: "0"

train:
  freeze: "no"
```

## Training

```bash
python src/train_lightning_multiclass.py 
```

## Inference

Configure the inference task in `src/test_inference_finetune.py`:

```python
"sequence_task": {
    "checkpoint_path": "./results/sequence_checkpoints/sequence_model-epoch-XX-val_acc-X.XX.ckpt",
    "test_csv_path": "./data/csvs/sequence_test.csv",
    "root_dir": "./data/images",
    "output_csv_path": "./inference/model_output/sequence_predictions.csv",
    "task_type": "classification",
    "image_type": "single",
    "num_classes": 4
}

DATASETS_TO_RUN = [
   "sequence_task"
]
```

### Run Inference
```bash
python src/test_inference_finetune.py
```

### Generate Saliency Maps

update the filepaths in `src/generate_multiclass_vit_saliency.py`:

```python
nifti_path = "/path/to/your/single/image.nii.gz"  # Single image for saliency generation
checkpoint_path = "./results/sequence_checkpoints/sequence_model-epoch-XX-val_acc-X.XX.ckpt"
config_path = "./src/config_finetune.yml"
output_dir = "./inference/saliency_maps"
```

Then run the script:
```bash
python src/generate_multiclass_vit_saliency.py
```

### Fine-tuning Options
- **Linear Probing:** Set `train.freeze: "yes"` in config 