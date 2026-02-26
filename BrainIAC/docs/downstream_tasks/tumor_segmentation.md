# Brain Tumor Segmentation

## Overview

Brain tumor segmentation is framed as a single class segmentation objective using FLAIR MRI scans. The model arcitecture consists of BrainIAC encoder and a U-Net decoder head.

### Task Details

- Modelling: Single class Segmentation
- Input: FLAIR MRI scan 
- Output: Binary segmentation mask
- Metric: Dice Score

### Input Format
- MRI Format: NIFTI (.nii.gz)
- Preprocessing: Bias field corrected, registered to standard space, skull stripped
- Image Size: 96×96×96 voxels (automatically resized)
- Mask Format: NIFTI (.nii.gz) with binary labels

### CSV Format

Your CSV file should contain the following columns:

```csv
image_path,mask_path
./data/images/subject001_images.nii.gz,./data/images/subject001_mask.nii.gz
```

### Directory Structure

```
data/
├── images/
│   ├── subject001_image.nii.gz
│   └── subject001_mask.nii.gz
└── csvs/
    ├── seg_train.csv
    ├── seg_val.csv
    └── seg_test.csv
```

## Configuration

Update the configurations in `src/config_finetune_segmentation.yml`:

```yaml
data:
  train_csv: "/path/to/seg_train.csv"
  val_csv: "/path/to/seg_val.csv"

model:
  img_size: [96, 96, 96]
  in_channels: 1
  out_channels: 1

pretrain:
  simclr_checkpoint_path: "/path/to/BrainIAC.ckpt"

training:
  batch_size: 10
  num_workers: 4
  max_epochs: 100
  lr: 5e-4
  weight_decay: 1e-4
  sw_batch_size: 2
  accumulate_grad_batches: 1
  freeze: "yes"  # Set to "no" for end-to-end training

output:
  output_dir: "/path/to/output/checkpoints"

logger:
  save_dir: "/path/to/output/checkpoints"
  save_name: "segmentation_model-{epoch:02d}-{val_dice:.2f}"
  run_name: "segmentation_experiment"
  project_name: "vit_segmentation_finetune"

gpu:
  visible_device: "0"
```

## Training

```bash
python src/train_lightning_segmentation.py 
```

## Inference

### Run Evaluation
```bash
"""
Command Line Arguments:
    --config: Path to configuration YAML file (default: config_finetune_segmentation.yml)
    --test_csv: Path to test CSV file (required)
    --checkpoint_path: Path to checkpoint file (required)
    --experiment_name: Name for the experiment (default: segmentation_task)
    --output_json: Path to save combined metrics JSON (default: ./inference/model_outputs/segmentation.json)
    --csv_output_dir: Directory to save per-case CSV files (default: ./inference/per_case_results)
"""

python src/test_segmentation.py \
    --config src/config_finetune_segmentation.yml \
    --test_csv "/path/to/seg_test.csv" \
    --checkpoint_path "/path/to/segmentation_checkpoint.ckpt" \
    --experiment_name "segmentation_task" \
    --output_json "./inference/model_outputs/segmentation.json" \
    --csv_output_dir "./inference/per_case_results"
```

### Generate Segmentation for Single Image

```bash
"""
Command Line Arguments:
    --checkpoint_path: Path to segmentation checkpoint file (required)
    --image_path: Path to input image (.nii.gz) (required)
    --output_dir: Directory to save segmentation output (required)
    --simclr_checkpoint_path: Override SimCLR checkpoint path from saved config (optional)
    --gpu_device: GPU device to use (default: 0)
"""

python src/generate_segmentation.py \
    --checkpoint_path "/path/to/segmentation_checkpoint.ckpt" \
    --image_path "/path/to/input_image.nii.gz" \
    --output_dir "/path/to/output/directory" \
    --simclr_checkpoint_path "/path/to/simclr_model.ckpt" \
    --gpu_device "0"
```


### Fine-tuning Options
- **Linear Probing:** Set `training.freeze: "yes"` in config
- **End-to-End Training:** Set `training.freeze: "no"` in config

