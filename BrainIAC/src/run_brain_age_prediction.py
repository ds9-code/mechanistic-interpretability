"""
Brain age prediction using: BrainIAC → SAE (normalize + decode) → Linear probe.
Uses the same setup as training: sae_checkpoints_x32_full_norm, linear_probe_results/best_model.pt.
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, r2_score

from dataset import BrainAgeDataset, get_validation_transform
from load_brainiac import load_brainiac
from sae_model import GatedSAE


def load_sae(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get('config', {})
    n_in = config.get('n_input_features', 768)
    exp = config.get('expansion_factor', 32)
    n_dict = config.get('n_dict_features', n_in * exp)
    model = GatedSAE(n_input_features=n_in, n_dict_features=n_dict, expansion_factor=exp).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    norm_stats = ckpt.get('normalization_stats', None)
    return model, norm_stats


class LinearProbe(nn.Module):
    def __init__(self, input_dim=768):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
    def forward(self, x):
        return self.linear(x).squeeze(-1)


def load_linear_probe(checkpoint_path, device, input_dim=768):
    model = LinearProbe(input_dim=input_dim).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
    if isinstance(state, dict):
        new_state = {}
        for k, v in state.items():
            if k.startswith('model.'): k = k[6:]
            if k.startswith('linear_probe.'): k = k[13:]
            new_state[k] = v
        state = new_state
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description='Brain age prediction: BrainIAC → SAE → Linear probe')
    parser.add_argument('--brainiac_checkpoint', type=str, default='BrainIAC/src/checkpoints/BrainIAC.ckpt')
    parser.add_argument('--sae_checkpoint', type=str, default='BrainIAC/src/sae_checkpoints_x32_full_norm/best_model.pt')
    parser.add_argument('--linear_probe', type=str, default='linear_probe_results/best_model.pt')
    parser.add_argument('--test_csv', type=str, default='data/csvs/development_test_set.csv')
    parser.add_argument('--root_dir', type=str, default='data/images/data')
    parser.add_argument('--output_dir', type=str, default='linear_probe_results')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    print('Loading BrainIAC...')
    brainiac = load_brainiac(args.brainiac_checkpoint, device)
    brainiac.eval()
    print('Loading SAE...')
    sae, norm_stats = load_sae(args.sae_checkpoint, device)
    print('Loading linear probe...')
    linear_probe = load_linear_probe(args.linear_probe, device, input_dim=768)

    transform = get_validation_transform(image_size=(96, 96, 96))
    dataset = BrainAgeDataset(csv_path=args.test_csv, root_dir=args.root_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=(device.type=='cuda'))

    all_pred = []
    all_labels = []
    all_pat_ids = []

    with torch.no_grad():
        for batch in tqdm(loader, desc='Predicting'):
            images = batch['image'].to(device)
            labels = batch['label'].cpu().numpy()
            pat_ids = batch.get('pat_id', [f'sample_{i}' for i in range(len(labels))])
            if isinstance(pat_ids, torch.Tensor):
                pat_ids = pat_ids.cpu().tolist()
            if hasattr(pat_ids, '__iter__') and not isinstance(pat_ids[0], str):
                pat_ids = [str(p) for p in pat_ids]

            cls_raw = brainiac(images)
            if norm_stats is not None:
                mean = norm_stats['mean'].to(device)
                std = norm_stats['std'].to(device)
                cls_norm = (cls_raw - mean) / (std + 1e-8)
            else:
                cls_norm = cls_raw
            decoded, _, _ = sae(cls_norm)
            # Match training: linear probe was trained on (SAE_decoded - CLS_mean) / CLS_std
            if norm_stats is not None:
                decoded = (decoded - mean) / (std + 1e-8)
            pred_months = linear_probe(decoded).cpu().numpy()

            all_pred.append(pred_months)
            all_labels.append(labels)
            all_pat_ids.extend(pat_ids if isinstance(pat_ids, list) else [pat_ids])

    pred = np.concatenate(all_pred).flatten()
    labels = np.concatenate(all_labels).flatten()
    mae_months = mean_absolute_error(labels, pred)
    r2 = r2_score(labels, pred)
    mae_years = mae_months / 12.0

    print(f'\nTest MAE: {mae_months:.2f} months ({mae_years:.2f} years)')
    print(f'Test R²:  {r2:.4f}')

    out_csv = os.path.join(args.output_dir, 'brain_age_predictions.csv')
    df = pd.DataFrame({
        'pat_id': all_pat_ids,
        'true_age_months': labels,
        'true_age_years': labels / 12.0,
        'predicted_age_months': pred,
        'predicted_age_years': pred / 12.0,
        'error_months': np.abs(pred - labels),
    })
    df.to_csv(out_csv, index=False)
    print(f'Predictions saved to {out_csv}')

    out_json = os.path.join(args.output_dir, 'brain_age_prediction_metrics.json')
    with open(out_json, 'w') as f:
        json.dump({
            'n_samples': len(pred),
            'mae_months': float(mae_months),
            'mae_years': float(mae_years),
            'r2': float(r2),
            'predictions_csv': out_csv,
        }, f, indent=2)
    print(f'Metrics saved to {out_json}')


if __name__ == '__main__':
    main()
