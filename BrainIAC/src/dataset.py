"""
Brain age dataset and transforms for BrainIAC.
CSV: pat_id, label (age in months). Images at root_dir / pat_id (pat_id may include .nii.gz).
"""

import os
import pandas as pd
from torch.utils.data import Dataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Resized,
    NormalizeIntensityd,
    ToTensord,
    RandFlipd,
)


def get_validation_transform(image_size=(96, 96, 96)):
    """No augmentation; used for val/test and feature extraction."""
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Resized(keys=["image"], spatial_size=image_size, mode="trilinear"),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        ToTensord(keys=["image"]),
    ])


def get_default_transform(image_size=(96, 96, 96)):
    """With light augmentation for training."""
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Resized(keys=["image"], spatial_size=image_size, mode="trilinear"),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
        ToTensord(keys=["image"]),
    ])


class BrainAgeDataset(Dataset):
    """
    CSV must have: pat_id (filename or path relative to root_dir), label (age in months).
    Optional: dataset, etc.
    """

    def __init__(self, csv_path, root_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        if "pat_id" not in self.df.columns or "label" not in self.df.columns:
            raise ValueError(
                f"CSV must have 'pat_id' and 'label'. Got: {list(self.df.columns)}"
            )
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pat_id = str(row["pat_id"]).strip()
        label = float(row["label"])
        img_path = os.path.join(self.root_dir, pat_id)
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        sample = {"image": img_path}
        if self.transform is not None:
            sample = self.transform(sample)
        sample["label"] = label
        sample["pat_id"] = pat_id
        return sample
