import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class NiftiSingleViewDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.dataframe = pd.read_csv(csv_file, dtype={"pat_id": str, "scandate": str})
        self.has_scandate = "scandate" in self.dataframe.columns

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        pat_id = str(self.dataframe.loc[idx, "pat_id"])
        dataset = str(self.dataframe.loc[idx, "dataset"])
        if self.has_scandate:
            scan_date = str(self.dataframe.loc[idx, "scandate"])
            img_name = os.path.join(self.root_dir, f"{dataset}/data/{pat_id}_{scan_date}.nii.gz")
        else:
            img_name = os.path.join(self.root_dir, f"{dataset}/data/{pat_id}")

        if not os.path.exists(img_name):
            raise FileNotFoundError(f"Image file not found: {img_name}")

        sample = {"image": img_name}
        if self.transform:
            sample = self.transform(sample)

        return sample


class EmbeddingDataset(Dataset):
    def __init__(self, embedding_csv=None, embedding_npy=None):
        if embedding_csv:
            df = pd.read_csv(embedding_csv)
            feature_cols = [c for c in df.columns if c.startswith("f")]
            if feature_cols:
                df = df[feature_cols]
            else:
                df = df.select_dtypes(include=["number"])
            self.embeddings = torch.tensor(df.values, dtype=torch.float32)
        elif embedding_npy:
            arr = torch.from_numpy(np.load(embedding_npy))
            self.embeddings = arr.float()
        else:
            raise ValueError("Provide embedding_csv or embedding_npy.")

    def __len__(self):
        return self.embeddings.shape[0]

    def __getitem__(self, idx):
        return {"embedding": self.embeddings[idx]}
