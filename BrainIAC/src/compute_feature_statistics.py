"""
Compute mean activations (μ) for features across different datasets

For each feature in {19999, 12826, 14253}:
- μ_train: Mean activation on training set
- μ_dev: Mean activation on developmental validation/test set
- μ_OASIS: Mean activation on OASIS evaluation set
- σ_train: Standard deviation on training set
"""

import os
import torch
import numpy as np
import pandas as pd
import argparse
import json
from tqdm import tqdm
from torch.utils.data import DataLoader

from load_brainiac import load_brainiac
from dataset import BrainAgeDataset, get_validation_transform
from baseline_oasis_inference import create_oasis_dataset
from sae_model import GatedSAE


def load_sae_model(checkpoint_path, device):
    """Load trained SAE model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})
    
    n_input_features = config.get('n_input_features', 768)
    expansion_factor = config.get('expansion_factor', 32)
    n_dict_features = config.get('n_dict_features', None)
    
    if n_dict_features is None:
        n_dict_features = n_input_features * expansion_factor
    
    model = GatedSAE(
        n_input_features=n_input_features,
        n_dict_features=n_dict_features,
        expansion_factor=expansion_factor
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    normalization_stats = checkpoint.get('normalization_stats', None)
    
    return model, normalization_stats, config


def extract_sae_activations(brainiac_model, sae_model, dataloader, device, 
                           normalization_stats=None):
    """
    Extract SAE activations from dataset
    
    Returns:
        activations: Tensor [n_samples, n_dict_features]
    """
    brainiac_model.eval()
    sae_model.eval()
    
    all_activations = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting SAE activations"):
            images = batch['image'].to(device)
            
            # Extract CLS features
            cls_features = brainiac_model(images)  # [batch, 768]
            
            # Normalize if needed
            if normalization_stats is not None:
                mean = normalization_stats['mean'].to(device)
                std = normalization_stats['std'].to(device)
                cls_features = (cls_features - mean) / (std + 1e-8)
            
            # Pass through SAE encoder
            encoded = sae_model.encoder(cls_features)  # [batch, n_dict_features]
            gate_values = sae_model.gate(cls_features)
            gate_activations = torch.sigmoid(gate_values)
            activations = encoded * gate_activations
            activations = torch.relu(activations)  # [batch, n_dict_features]
            
            all_activations.append(activations.cpu())
    
    # Concatenate all activations
    all_activations = torch.cat(all_activations, dim=0)  # [n_samples, n_dict_features]
    
    return all_activations


def compute_feature_statistics(
    training_csv_path,
    training_root_dir,
    dev_csv_path,
    dev_root_dir,
    oasis_csv_path,
    oasis_root_dir,
    brainiac_ckpt,
    sae_ckpt,
    feature_indices,
    output_file,
    batch_size=1,
    image_size=(96, 96, 96),
    device='cuda',
    normalize_cls_features=True
):
    """
    Compute mean activations (μ) for features across training, developmental, and OASIS datasets
    """
    print("=" * 80)
    print("Feature Activation Statistics Computation")
    print("=" * 80)
    print()
    
    # Load models
    print("Step 1: Loading models...")
    print(f"  Loading BrainIAC encoder from: {brainiac_ckpt}")
    brainiac_model = load_brainiac(brainiac_ckpt, device=device)
    brainiac_model.eval()
    
    print(f"  Loading SAE from: {sae_ckpt}")
    sae_model, sae_norm_stats, sae_config = load_sae_model(sae_ckpt, device)
    print()
    
    # Create datasets
    print("Step 2: Creating datasets...")
    transform = get_validation_transform(image_size=image_size)
    
    datasets_info = [
        ("Training", training_csv_path, training_root_dir, BrainAgeDataset),
        ("Developmental", dev_csv_path, dev_root_dir, BrainAgeDataset),
        ("OASIS", oasis_csv_path, oasis_root_dir, None),  # Will use create_oasis_dataset
    ]
    
    all_activations = {}
    
    for dataset_name, csv_path, root_dir, dataset_class in datasets_info:
        print(f"  {dataset_name} dataset: {csv_path}")
        
        if dataset_name == "OASIS":
            from baseline_oasis_inference import create_oasis_dataset
            dataset = create_oasis_dataset(csv_path, root_dir, image_size=image_size)
        else:
            dataset = dataset_class(csv_path, root_dir, transform=transform)
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        print(f"    {len(dataset)} samples")
        
        # Extract activations
        print(f"  Extracting SAE activations from {dataset_name} set...")
        activations = extract_sae_activations(
            brainiac_model, sae_model, dataloader, device,
            normalization_stats=sae_norm_stats if normalize_cls_features else None
        )
        all_activations[dataset_name.lower()] = activations
        print(f"    Shape: {activations.shape}")
        print()
    
    # Compute statistics for each feature
    print("Step 3: Computing statistics for each feature...")
    print()
    
    results = []
    
    for feat_idx in feature_indices:
        if feat_idx >= all_activations['training'].shape[1]:
            print(f"  Warning: Feature {feat_idx} out of range [0, {all_activations['training'].shape[1]-1}]")
            continue
        
        # Training set statistics
        train_activations = all_activations['training'][:, feat_idx].numpy()
        mu_train = float(train_activations.mean())
        sigma_train = float(train_activations.std())
        
        # Developmental set statistics
        dev_activations = all_activations['developmental'][:, feat_idx].numpy()
        mu_dev = float(dev_activations.mean())
        
        # OASIS set statistics
        oasis_activations = all_activations['oasis'][:, feat_idx].numpy()
        mu_oasis = float(oasis_activations.mean())
        
        results.append({
            'feature_idx': feat_idx,
            'mu_train': mu_train,
            'mu_dev': mu_dev,
            'mu_oasis': mu_oasis,
            'sigma_train': sigma_train,
        })
        
        print(f"Feature {feat_idx}:")
        print(f"  μ_train = {mu_train:.6f}")
        print(f"  μ_dev = {mu_dev:.6f}")
        print(f"  μ_OASIS = {mu_oasis:.6f}")
        print(f"  σ_train = {sigma_train:.6f}")
        print()
    
    # Save results
    print("Step 4: Saving results...")
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f"  Saved to: {output_file}")
    
    # Save JSON
    json_output = output_file.replace('.csv', '.json')
    with open(json_output, 'w') as f:
        json.dump({
            'features_analyzed': feature_indices,
            'results': results
        }, f, indent=2)
    print(f"  Saved JSON to: {json_output}")
    
    print()
    print("=" * 80)
    print("Statistics Computation Complete")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Compute feature activation statistics')
    parser.add_argument('--training_csv', type=str, required=True,
                        help='Path to training CSV')
    parser.add_argument('--training_root_dir', type=str, required=True,
                        help='Root directory for training images')
    parser.add_argument('--dev_csv', type=str, required=True,
                        help='Path to developmental validation/test CSV')
    parser.add_argument('--dev_root_dir', type=str, required=True,
                        help='Root directory for developmental images')
    parser.add_argument('--oasis_csv', type=str, required=True,
                        help='Path to OASIS evaluation CSV')
    parser.add_argument('--oasis_root_dir', type=str, required=True,
                        help='Root directory for OASIS images')
    parser.add_argument('--brainiac_ckpt', type=str, required=True,
                        help='Path to BrainIAC checkpoint')
    parser.add_argument('--sae_ckpt', type=str, required=True,
                        help='Path to SAE checkpoint')
    parser.add_argument('--feature_indices', type=int, nargs='+', required=True,
                        help='Feature indices to analyze (e.g., 19999 12826 14253)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output CSV file path')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size (default: 1)')
    parser.add_argument('--image_size', type=int, nargs=3, default=[96, 96, 96],
                        help='Image size (default: 96 96 96)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (default: cuda)')
    parser.add_argument('--normalize_cls', action='store_true',
                        help='Normalize CLS features')
    
    args = parser.parse_args()
    
    if args.device == 'cuda' and torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
    
    compute_feature_statistics(
        training_csv_path=args.training_csv,
        training_root_dir=args.training_root_dir,
        dev_csv_path=args.dev_csv,
        dev_root_dir=args.dev_root_dir,
        oasis_csv_path=args.oasis_csv,
        oasis_root_dir=args.oasis_root_dir,
        brainiac_ckpt=args.brainiac_ckpt,
        sae_ckpt=args.sae_ckpt,
        feature_indices=args.feature_indices,
        output_file=args.output,
        batch_size=args.batch_size,
        image_size=tuple(args.image_size),
        device=args.device,
        normalize_cls_features=args.normalize_cls
    )


if __name__ == '__main__':
    main()
