"""
Training script for Sparse Autoencoder (SAE) on BrainIAC features
"""

import os
import sys
from pathlib import Path

# When running from IDE ("Run" on this file), set cwd to repo root and add BrainIAC/src to path
_script_dir = Path(__file__).resolve().parent
_repo_root = _script_dir.parent.parent  # BrainIAC/src -> BrainIAC -> mechinterp
if (_repo_root / "data").is_dir() and (_repo_root / "BrainIAC").is_dir():
    os.chdir(_repo_root)
    if str(_script_dir) not in sys.path:
        sys.path.insert(0, str(_script_dir))

import torch
import torch.nn as nn
import numpy as np
import argparse
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import yaml
from pathlib import Path
import wandb
from scipy.stats import pearsonr

from sae_model import GatedSAE, SAELoss, compute_explained_variance
from dataset import BrainAgeDataset, get_default_transform, get_validation_transform
from load_brainiac import load_brainiac


class LinearRegressor(nn.Module):
    """Simple linear regressor for brain age prediction"""
    def __init__(self, input_dim=768):
        super(LinearRegressor, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        return self.linear(x).squeeze(-1)


def compute_reconstruction_accuracy(reconstructed, original, threshold=0.1):
    """
    Compute reconstruction accuracy - percentage of features reconstructed within threshold
    
    Args:
        reconstructed: Reconstructed features [batch_size, n_features]
        original: Original features [batch_size, n_features]
        threshold: Relative error threshold (default: 0.1 = 10% error)
    
    Returns:
        accuracy: Percentage of features within threshold (0-1)
    """
    # Compute relative error for each feature
    abs_error = torch.abs(reconstructed - original)
    abs_original = torch.abs(original) + 1e-8  # Avoid division by zero
    relative_error = abs_error / abs_original
    
    # Features within threshold
    within_threshold = (relative_error < threshold).float()
    
    # Accuracy = percentage of features within threshold
    accuracy = within_threshold.mean().item()
    
    return accuracy


def compute_feature_matching_accuracy(reconstructed, original, tolerance=0.05):
    """
    Compute feature matching accuracy - percentage of features matching within tolerance
    
    Args:
        reconstructed: Reconstructed features [batch_size, n_features]
        original: Original features [batch_size, n_features]
        tolerance: Absolute tolerance for matching
    
    Returns:
        accuracy: Percentage of features matching (0-1)
    """
    # Features within absolute tolerance
    abs_diff = torch.abs(reconstructed - original)
    matching = (abs_diff < tolerance).float()
    
    # Accuracy = percentage of matching features
    accuracy = matching.mean().item()
    
    return accuracy


class ImageDataset(Dataset):
    """Dataset that loads images only (features extracted in batches during training for efficiency)"""
    
    def __init__(self, csv_path, root_dir, transform):
        """
        Args:
            csv_path: Path to CSV file with dataset info
            root_dir: Root directory containing images
            transform: Image transform pipeline (with augmentations for training)
        """
        self.image_dataset = BrainAgeDataset(csv_path, root_dir, transform=transform)
    
    def __len__(self):
        return len(self.image_dataset)
    
    def __getitem__(self, idx):
        # Return images only - feature extraction happens in batches during training
        sample = self.image_dataset[idx]
        return {
            'image': sample['image'],  # [C, D, H, W]
            'label': sample['label'],
            'pat_id': sample.get('pat_id', f'sample_{idx}')
        }
    
    def normalize_features(self, features):
        """Normalize features using computed stats"""
        if self.mean is None or self.std is None:
            raise ValueError("Normalization stats not computed.")
        return (features - self.mean) / self.std


def train_step(model, batch, criterion, optimizer, device, step, log_every=100,
               age_probe=None, age_preservation_weight=0.0, age_loss_type='mse',
               brainiac_model=None, mean=None, std=None, scaler=None):
    """Train for one step (one batch)"""
    model.train()
    if age_probe is not None:
        age_probe.eval()  # Keep age probe frozen
    if brainiac_model is not None:
        brainiac_model.eval()  # Keep BrainIAC frozen
    
    # Extract features in batches (more efficient than per-sample)
    images = batch['image'].to(device)  # [B, C, D, H, W]
    
    # Use mixed precision for forward pass
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            # Extract CLS token features using BrainIAC (batched)
            features_raw = brainiac_model(images)  # [B, 768]
        
        # Normalize features
        if mean is not None and std is not None:
            mean_tensor = mean.to(device)
            std_tensor = std.to(device)
            features = (features_raw - mean_tensor) / std_tensor
        else:
            features = features_raw
        
        # Forward pass
        reconstructed, activations, gate_values = model(features)
        
        # Compute base loss (reconstruction + sparsity)
        loss, recon_loss, sparsity_loss = criterion(reconstructed, features, activations)
        
        # Compute age preservation loss if age probe is provided
        age_loss = torch.tensor(0.0, device=device)
        if age_probe is not None and age_preservation_weight > 0:
            # Denormalize features for age probe (age probe expects original scale)
            if mean is not None and std is not None:
                mean_tensor = mean.to(device)
                std_tensor = std.to(device)
                features_denorm = features * std_tensor + mean_tensor
                reconstructed_denorm = reconstructed * std_tensor + mean_tensor
            else:
                features_denorm = features
                reconstructed_denorm = reconstructed
            
            # Pass through frozen age probe
            with torch.no_grad():
                age_pred_orig = age_probe(features_denorm)
            age_pred_recon = age_probe(reconstructed_denorm)
            
            # Age preservation loss
            if age_loss_type == 'mse':
                age_loss = nn.functional.mse_loss(age_pred_recon, age_pred_orig.detach())
            elif age_loss_type == 'mae':
                age_loss = nn.functional.l1_loss(age_pred_recon, age_pred_orig.detach())
            else:
                raise ValueError(f"Unknown age_loss_type: {age_loss_type}")
            
            loss = loss + age_preservation_weight * age_loss
    
    # Backward pass with mixed precision
    optimizer.zero_grad()
    if scaler is not None:
        scaler.scale(loss).backward()
        # Gradient clipping with scaler
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    
    # Metrics
    l0 = model.get_l0_norm(activations)
    explained_var = compute_explained_variance(reconstructed, features)
    recon_acc = compute_reconstruction_accuracy(reconstructed, features, threshold=0.1)
    feature_acc = compute_feature_matching_accuracy(reconstructed, features, tolerance=0.05)
    
    metrics = {
        'loss': loss.item(),
        'recon_loss': recon_loss.item(),
        'sparsity_loss': sparsity_loss.item(),
        'age_loss': age_loss.item(),
        'l0': l0.item(),
        'explained_variance': explained_var,
        'reconstruction_accuracy': recon_acc,
        'feature_accuracy': feature_acc
    }
    
    # Log to wandb periodically (log at step 1 and then every log_every steps)
    if step == 1 or step % log_every == 0:
        # Calculate sparsity fraction (L0 / dict_size)
        dict_size = model.n_dict_features
        sparsity_fraction = metrics['l0'] / dict_size
        log_dict = {
            'train/loss': metrics['loss'],
            'train/recon_loss': metrics['recon_loss'],
            'train/sparsity_loss': metrics['sparsity_loss'],
            'train/l0': metrics['l0'],
            'train/sparsity_fraction': sparsity_fraction,  # L0 / dict_size (percentage)
            'train/explained_variance': metrics['explained_variance'],
            'train/reconstruction_accuracy': metrics['reconstruction_accuracy'],
            'train/feature_accuracy': metrics['feature_accuracy'],
            'train/learning_rate': optimizer.param_groups[0]['lr'],
            'step': step
        }
        if age_probe is not None and age_preservation_weight > 0:
            log_dict['train/age_preservation_loss'] = metrics['age_loss']
        wandb.log(log_dict, step=step)
    
    return metrics


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, log_every=100, 
                age_probe=None, age_preservation_weight=0.0, age_loss_type='mse',
                brainiac_model=None, mean=None, std=None):
    """Train for one epoch (legacy function, kept for backward compatibility)"""
    model.train()
    if age_probe is not None:
        age_probe.eval()  # Keep age probe frozen
    if brainiac_model is not None:
        brainiac_model.eval()  # Keep BrainIAC frozen
    
    total_loss = 0.0
    total_recon_loss = 0.0
    total_sparsity_loss = 0.0
    total_age_loss = 0.0
    total_l0 = 0.0
    total_explained_var = 0.0
    total_recon_accuracy = 0.0
    total_feature_accuracy = 0.0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for step, batch in enumerate(pbar):
        # Extract features in batches (more efficient than per-sample)
        images = batch['image'].to(device)  # [B, C, D, H, W]
        
        with torch.no_grad():
            # Extract CLS token features using BrainIAC (batched)
            features_raw = brainiac_model(images)  # [B, 768]
        
        # Normalize features
        if mean is not None and std is not None:
            mean_tensor = mean.to(device)
            std_tensor = std.to(device)
            features = (features_raw - mean_tensor) / std_tensor
        else:
            features = features_raw
        
        # Forward pass
        reconstructed, activations, gate_values = model(features)
        
        # Compute base loss (reconstruction + sparsity)
        loss, recon_loss, sparsity_loss = criterion(reconstructed, features, activations)
        
        # Compute age preservation loss if age probe is provided
        age_loss = torch.tensor(0.0, device=device)
        if age_probe is not None and age_preservation_weight > 0:
            # Denormalize features for age probe (age probe expects original scale)
            # Note: We need to denormalize both original and reconstructed
            if mean is not None and std is not None:
                mean_tensor = mean.to(device)
                std_tensor = std.to(device)
                features_denorm = features * std_tensor + mean_tensor
                reconstructed_denorm = reconstructed * std_tensor + mean_tensor
            else:
                # No normalization was applied
                features_denorm = features
                reconstructed_denorm = reconstructed
            
            # Pass through frozen age probe
            # Age probe is frozen (requires_grad=False), so gradients won't flow through it
            # but will flow through the SAE decoder/encoder
            with torch.no_grad():
                # Original prediction - just for comparison, no gradients needed
                age_pred_orig = age_probe(features_denorm)
            
            # Reconstructed prediction - gradients flow through SAE (but not age probe since it's frozen)
            age_pred_recon = age_probe(reconstructed_denorm)
            
            # Age preservation loss: MSE or MAE between age predictions
            if age_loss_type == 'mse':
                age_loss = nn.functional.mse_loss(age_pred_recon, age_pred_orig.detach())
            elif age_loss_type == 'mae':
                age_loss = nn.functional.l1_loss(age_pred_recon, age_pred_orig.detach())
            else:
                raise ValueError(f"Unknown age_loss_type: {age_loss_type}")
            
            # Add age preservation loss to total loss
            loss = loss + age_preservation_weight * age_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Metrics
        l0 = model.get_l0_norm(activations)
        explained_var = compute_explained_variance(reconstructed, features)
        recon_acc = compute_reconstruction_accuracy(reconstructed, features, threshold=0.1)
        feature_acc = compute_feature_matching_accuracy(reconstructed, features, tolerance=0.05)
        
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_sparsity_loss += sparsity_loss.item()
        total_age_loss += age_loss.item()
        total_l0 += l0.item()
        total_explained_var += explained_var
        total_recon_accuracy += recon_acc
        total_feature_accuracy += feature_acc
        
        # Update progress bar
        postfix_dict = {
            'loss': f'{loss.item():.4f}',
            'recon': f'{recon_loss.item():.4f}',
            'sparsity': f'{sparsity_loss.item():.4f}',
            'L0': f'{l0.item():.2f}',
            'EV': f'{explained_var:.3f}',
            'Acc': f'{recon_acc:.3f}'
        }
        if age_probe is not None and age_preservation_weight > 0:
            postfix_dict['age'] = f'{age_loss.item():.4f}'
        pbar.set_postfix(postfix_dict)
        
        # Log to wandb periodically
        if step % log_every == 0:
            # Note: This is legacy train_epoch function - step here is batch step within epoch
            # For step-based training, use train_step function which uses actual step number
            log_dict = {
                'train/loss': loss.item(),
                'train/recon_loss': recon_loss.item(),
                'train/sparsity_loss': sparsity_loss.item(),
                'train/l0': l0.item(),
                'train/explained_variance': explained_var,
                'train/reconstruction_accuracy': recon_acc,
                'train/feature_accuracy': feature_acc,
                'train/learning_rate': optimizer.param_groups[0]['lr'],
                'epoch': epoch
            }
            if age_probe is not None and age_preservation_weight > 0:
                log_dict['train/age_preservation_loss'] = age_loss.item()
            # Use epoch-based step for legacy function
            wandb.log(log_dict, step=epoch)
    
    n_batches = len(dataloader)
    result = {
        'loss': total_loss / n_batches,
        'recon_loss': total_recon_loss / n_batches,
        'sparsity_loss': total_sparsity_loss / n_batches,
        'l0': total_l0 / n_batches,
        'explained_variance': total_explained_var / n_batches,
        'reconstruction_accuracy': total_recon_accuracy / n_batches,
        'feature_accuracy': total_feature_accuracy / n_batches
    }
    if age_probe is not None and age_preservation_weight > 0:
        result['age_preservation_loss'] = total_age_loss / n_batches
    return result


def validate(model, dataloader, criterion, device, 
             age_probe=None, age_preservation_weight=0.0, age_loss_type='mse',
             brainiac_model=None, mean=None, std=None):
    """Validate the model"""
    model.eval()
    if age_probe is not None:
        age_probe.eval()
    if brainiac_model is not None:
        brainiac_model.eval()
    
    total_loss = 0.0
    total_recon_loss = 0.0
    total_sparsity_loss = 0.0
    total_age_loss = 0.0
    total_l0 = 0.0
    total_explained_var = 0.0
    
    # Collect all features for correlation/metrics
    all_reconstructed = []
    all_original = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            # Extract features in batches (more efficient than per-sample)
            images = batch['image'].to(device)  # [B, C, D, H, W]
            labels = batch['label'].cpu().numpy()
            
            # Use mixed precision for validation (faster, no gradients needed)
            with torch.cuda.amp.autocast():
                # Extract CLS token features using BrainIAC (batched)
                features_raw = brainiac_model(images)  # [B, 768]
                
                # Normalize features
                if mean is not None and std is not None:
                    mean_tensor = mean.to(device)
                    std_tensor = std.to(device)
                    features = (features_raw - mean_tensor) / std_tensor
                else:
                    features = features_raw
                
                # Forward pass
                reconstructed, activations, gate_values = model(features)
                
                # Compute base loss
                loss, recon_loss, sparsity_loss = criterion(reconstructed, features, activations)
                
                # Compute age preservation loss if age probe is provided
                age_loss = torch.tensor(0.0, device=device)
                if age_probe is not None and age_preservation_weight > 0:
                    # Denormalize features for age probe
                    if mean is not None and std is not None:
                        mean_tensor = mean.to(device)
                        std_tensor = std.to(device)
                        features_denorm = features * std_tensor + mean_tensor
                        reconstructed_denorm = reconstructed * std_tensor + mean_tensor
                    else:
                        features_denorm = features
                        reconstructed_denorm = reconstructed
                    
                    # Pass through frozen age probe
                    age_pred_orig = age_probe(features_denorm)
                    age_pred_recon = age_probe(reconstructed_denorm)
                    
                    # Age preservation loss
                    if age_loss_type == 'mse':
                        age_loss = nn.functional.mse_loss(age_pred_recon, age_pred_orig)
                    elif age_loss_type == 'mae':
                        age_loss = nn.functional.l1_loss(age_pred_recon, age_pred_orig)
                    else:
                        age_loss = nn.functional.mse_loss(age_pred_recon, age_pred_orig)
                    loss = loss + age_preservation_weight * age_loss
            
            # Metrics
            l0 = model.get_l0_norm(activations)
            explained_var = compute_explained_variance(reconstructed, features)
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_sparsity_loss += sparsity_loss.item()
            total_age_loss += age_loss.item()
            total_l0 += l0.item()
            total_explained_var += explained_var
            
            # Collect for correlation/metrics
            all_reconstructed.append(reconstructed.cpu())
            all_original.append(features.cpu())
            all_labels.append(labels)
    
    n_batches = len(dataloader)
    metrics = {
        'loss': total_loss / n_batches,
        'recon_loss': total_recon_loss / n_batches,
        'sparsity_loss': total_sparsity_loss / n_batches,
        'l0': total_l0 / n_batches,
        'explained_variance': total_explained_var / n_batches
    }
    if age_probe is not None and age_preservation_weight > 0:
        metrics['age_preservation_loss'] = total_age_loss / n_batches
    
    # Calculate correlation between reconstructed and original features
    all_reconstructed = torch.cat(all_reconstructed, dim=0)
    all_original = torch.cat(all_original, dim=0)
    
    # Flatten for correlation
    recon_flat = all_reconstructed.flatten().numpy()
    orig_flat = all_original.flatten().numpy()
    
    try:
        correlation, _ = pearsonr(recon_flat, orig_flat)
        metrics['correlation'] = correlation
    except:
        metrics['correlation'] = 0.0
    
    # Calculate correlation between reconstructed and original features
    recon_flat = all_reconstructed.flatten().numpy()
    orig_flat = all_original.flatten().numpy()
    
    try:
        correlation, _ = pearsonr(recon_flat, orig_flat)
        metrics['correlation'] = correlation
    except:
        metrics['correlation'] = 0.0
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Train Sparse Autoencoder on BrainIAC features')
    parser.add_argument('--config', type=str, default='config_sae.yml',
                       help='Path to config file')
    parser.add_argument('--train_csv', type=str, required=True,
                       help='Path to training CSV file')
    parser.add_argument('--val_csv', type=str, default=None,
                       help='Path to validation CSV file (optional)')
    parser.add_argument('--root_dir', type=str, required=True,
                       help='Root directory containing images')
    parser.add_argument('--brainiac_checkpoint', type=str, required=True,
                       help='Path to BrainIAC checkpoint (frozen model for feature extraction)')
    parser.add_argument('--output_dir', type=str, default='./sae_checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--n_input_features', type=int, default=768,
                       help='Input feature dimension (CLS token size)')
    parser.add_argument('--expansion_factor', type=int, default=32,
                       help='Expansion factor for dictionary size')
    parser.add_argument('--l1_coefficient', type=float, default=1e-3,
                       help='L1 sparsity regularization coefficient')
    parser.add_argument('--reconstruction_loss_type', type=str, default='mse',
                       choices=['mse', 'cosine', 'combined'],
                       help='Type of reconstruction loss: mse, cosine, or combined (default: mse)')
    parser.add_argument('--mse_weight', type=float, default=1.0,
                        help='Weight for MSE component in combined loss (default: 1.0, only used if reconstruction_loss_type is combined)')
    parser.add_argument('--cosine_weight', type=float, default=0.0,
                        help='Weight for cosine component in combined loss (default: 0.0, only used if reconstruction_loss_type is combined)')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for training (default: 64, recommended: 64-256)')
    parser.add_argument('--num_steps', type=int, default=30000,
                       help='Number of training steps (default: 30000, recommended: 20000-50000)')
    parser.add_argument('--num_epochs', type=int, default=None,
                       help='[DEPRECATED] Number of epochs. Use --num_steps instead. If provided, will be converted to steps.')
    parser.add_argument('--learning_rate', type=float, default=5e-4,
                       help='Learning rate (default: 5e-4 for step-based training)')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                       help='Weight decay')
    parser.add_argument('--warmup_steps', type=int, default=2000,
                       help='Number of warmup steps for learning rate')
    parser.add_argument('--save_every', type=int, default=5000,
                       help='Save checkpoint every N steps (default: 5000)')
    parser.add_argument('--validate_every', type=int, default=1000,
                       help='Run validation every N steps (default: 1000, recommended: 500-1000)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--wandb_project', type=str, default='brainiac-sae',
                       help='Wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                       help='Wandb run name (default: auto-generated)')
    parser.add_argument('--wandb_entity', type=str, default=None,
                       help='Wandb entity/team name')
    parser.add_argument('--log_every', type=int, default=100,
                       help='Log metrics to wandb every N steps')
    parser.add_argument('--age_probe_path', type=str, default=None,
                       help='Path to frozen age probe checkpoint (trained on raw CLS features). If provided, age preservation loss will be added.')
    parser.add_argument('--age_preservation_weight', type=float, default=0.0,
                       help='Weight (α) for age preservation loss. Should be small (0.01-0.1). Default: 0.0 (disabled)')
    parser.add_argument('--age_loss_type', type=str, default='mse', choices=['mse', 'mae'],
                       help='Type of age preservation loss: mse or mae (default: mse)')
    
    args = parser.parse_args()
    
    # Set device and verify GPU
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available. Falling back to CPU.")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("  WARNING: Using CPU - training will be very slow!")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load frozen age probe if provided
    age_probe = None
    if args.age_probe_path and args.age_preservation_weight > 0:
        print(f"\nLoading frozen age probe from {args.age_probe_path}...")
        age_probe_checkpoint = torch.load(args.age_probe_path, map_location=device, weights_only=False)
        input_dim = age_probe_checkpoint.get('input_dim', args.n_input_features)
        age_probe = LinearRegressor(input_dim=input_dim).to(device)
        age_probe.load_state_dict(age_probe_checkpoint['model_state_dict'])
        
        # Freeze age probe - no gradients should flow through it
        for param in age_probe.parameters():
            param.requires_grad = False
        
        age_probe.eval()
        print(f"✓ Age probe loaded and frozen")
        print(f"  Age preservation weight (α): {args.age_preservation_weight}")
        print(f"  Age loss type: {args.age_loss_type}")
    elif args.age_probe_path and args.age_preservation_weight == 0:
        print(f"\n⚠ Age probe path provided but weight is 0.0 - age preservation disabled")
    elif args.age_preservation_weight > 0 and not args.age_probe_path:
        raise ValueError("age_preservation_weight > 0 but no age_probe_path provided!")
    
    # Fail fast if required paths are missing (so wandb run doesn't show "crashed" with no logs)
    if not os.path.isfile(args.brainiac_checkpoint):
        raise FileNotFoundError(
            f"BrainIAC checkpoint not found: {args.brainiac_checkpoint}\n"
            "Download it and place in BrainIAC/src/checkpoints/BrainIAC.ckpt (see BrainIAC README)."
        )
    if not os.path.isfile(args.train_csv):
        raise FileNotFoundError(f"Train CSV not found: {args.train_csv}")
    if not os.path.isdir(args.root_dir):
        raise FileNotFoundError(
            f"Image root_dir not found: {args.root_dir}\n"
            "Training loads images from root_dir; ensure the path exists and contains the NIfTI files."
        )
    
    # Initialize wandb (config will be updated after datasets are loaded)
    wandb_config = {
        'n_input_features': args.n_input_features,
        'expansion_factor': args.expansion_factor,
        'n_dict_features': args.n_input_features * args.expansion_factor,
        'l1_coefficient': args.l1_coefficient,
        'reconstruction_loss_type': args.reconstruction_loss_type,
        'mse_weight': args.mse_weight,
        'cosine_weight': args.cosine_weight,
        'batch_size': args.batch_size,
        'num_steps': args.num_steps,
        'num_epochs': args.num_epochs,  # Keep for backward compatibility
        'validate_every': args.validate_every,
        'save_every': args.save_every,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'warmup_steps': args.warmup_steps,
        'device': str(device),
        'age_preservation_enabled': age_probe is not None,
        'age_preservation_weight': args.age_preservation_weight if age_probe is not None else 0.0,
        'age_loss_type': args.age_loss_type if age_probe is not None else None,
        'use_augmentations': True,
        'training_mode': 'images_with_augmentations_steps',
        'brainiac_checkpoint': args.brainiac_checkpoint,
        'mixed_precision': device.type == 'cuda',  # Enable mixed precision on CUDA
    }
    
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        entity=args.wandb_entity,
        config=wandb_config,
        dir=args.output_dir
    )
    
    run_url = getattr(wandb.run, "url", None) or getattr(wandb.run, "get_url", lambda: None)()
    print(f"Wandb initialized: project={args.wandb_project}, run={wandb.run.name}")
    if run_url:
        print(f"  View run at: {run_url}")
    else:
        # Fallback: construct URL from entity/project/run_id (run_id available after sync)
        try:
            e = getattr(wandb.run, "entity", "") or ""
            p = getattr(wandb.run, "project", "") or args.wandb_project
            i = getattr(wandb.run, "id", "")
            if e and i:
                print(f"  View run at: https://wandb.ai/{e}/{p}/r/{i}")
            elif i:
                print(f"  View run at: https://wandb.ai/{p}/r/{i}")
        except Exception:
            pass
    
    # Log immediately so the run appears in wandb (otherwise it may not show until training loop)
    wandb.log({"run_initialized": 1, "training/num_steps_planned": args.num_steps}, step=0)
    print("  (Logged to wandb. If run still doesn't appear: check 'wandb login', project 'brainiac-sae', and that WANDB_MODE is not 'offline'.)")
    
    # Load BrainIAC model (frozen) for on-the-fly feature extraction
    print("Loading BrainIAC model (frozen) for on-the-fly feature extraction...")
    brainiac_model = load_brainiac(args.brainiac_checkpoint, device)
    brainiac_model.eval()  # Freeze BrainIAC model
    for param in brainiac_model.parameters():
        param.requires_grad = False
    print("✓ BrainIAC model loaded and frozen")
    
    # Load datasets - images only (features extracted in batches during training for efficiency)
    print("Loading training dataset with augmentations (images only, features extracted in batches)...")
    val_dataset = None  # Initialize for scoping
    
    # Training: load images with augmentations
    train_transform = get_default_transform(image_size=(96, 96, 96))  # With augmentations
    train_dataset = ImageDataset(
        csv_path=args.train_csv,
        root_dir=args.root_dir,
        transform=train_transform
    )
    
    # Compute normalization stats from training data (extract features in batches for efficiency)
    print("Computing normalization statistics from training data...")
    train_dataloader_temp = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    all_features = []
    # Use full training set for normalization stats (must match linear probe's cls_norm_stats)
    with torch.no_grad():
        for batch in tqdm(train_dataloader_temp, desc="Computing normalization stats"):
            images = batch['image'].to(device)
            features_batch = brainiac_model(images)  # [B, 768]
            all_features.append(features_batch.cpu())
    
    # Stack and compute stats from full dataset
    all_features = torch.cat(all_features, dim=0)  # [n_samples, 768]
    n_samples = all_features.shape[0]
    mean = all_features.mean(dim=0, keepdim=True)  # [1, 768]
    std = all_features.std(dim=0, keepdim=True) + 1e-8  # [1, 768]
    
    print(f"Normalization stats computed from {n_samples} samples:")
    print(f"  Mean shape: {mean.shape}, Std shape: {std.shape}")
    print(f"  Mean range: [{mean.min():.4f}, {mean.max():.4f}]")
    print(f"  Std range: [{std.min():.4f}, {std.max():.4f}]")
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Validation: load images without augmentations
    val_dataloader = None
    if args.val_csv:
        print("Loading validation dataset (no augmentations, images only)...")
        val_transform = get_validation_transform(image_size=(96, 96, 96))  # No augmentations
        val_dataset = ImageDataset(
            csv_path=args.val_csv,
            root_dir=args.root_dir,
            transform=val_transform
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True if device.type == 'cuda' else False
        )
    
    # Create model
    print("Creating SAE model...")
    model = GatedSAE(
        n_input_features=args.n_input_features,
        expansion_factor=args.expansion_factor
    ).to(device)
    
    print(f"Model architecture:")
    print(f"  Input features: {args.n_input_features}")
    print(f"  Dictionary size: {model.n_dict_features}")
    print(f"  Expansion factor: {args.expansion_factor}")
    
    # Create loss function
    criterion = SAELoss(
        l1_coefficient=args.l1_coefficient,
        reconstruction_loss_type=args.reconstruction_loss_type,
        mse_weight=args.mse_weight,
        cosine_weight=args.cosine_weight
    )
    
    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Initialize mixed precision scaler (for FP16 training on RTX 3090)
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    if scaler is not None:
        print("✓ Mixed precision training enabled (FP16)")
    
    # Handle num_epochs to num_steps conversion (backward compatibility)
    if args.num_epochs is not None:
        # Convert epochs to steps
        steps_per_epoch = len(train_dataloader)
        num_steps = args.num_epochs * steps_per_epoch
        print(f"Converting {args.num_epochs} epochs to {num_steps:,} steps ({steps_per_epoch} steps/epoch)")
        # Update wandb config
        wandb.config.update({'num_steps': num_steps, 'steps_per_epoch': steps_per_epoch}, allow_val_change=True)
    else:
        num_steps = args.num_steps
    
    # Learning rate scheduler with warmup
    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / args.warmup_steps
        else:
            # Linear warmdown for last 20% of training
            warmdown_start = int(num_steps * 0.8)
            if step > warmdown_start:
                return (num_steps - step) / (num_steps - warmdown_start)
            return 1.0
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training loop - step-based
    print(f"\nStarting training for {num_steps:,} steps...")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Validate every: {args.validate_every} steps")
    print(f"  Save every: {args.save_every} steps")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Log every: {args.log_every} steps")
    best_val_loss = float('inf')
    
    # Log initial config to wandb to verify connection
    wandb.log({'training/started': True, 'training/num_steps': num_steps}, step=0)
    
    # Create iterator for dataloader (allows cycling through dataset)
    train_dataloader_iter = iter(train_dataloader)
    
    # Track metrics for averaging
    train_metrics_accum = {
        'loss': [], 'recon_loss': [], 'sparsity_loss': [], 'age_loss': [],
        'l0': [], 'explained_variance': [], 'reconstruction_accuracy': [], 'feature_accuracy': []
    }
    
    try:
        pbar = tqdm(range(1, num_steps + 1), desc="Training")
        for step in pbar:
            # Get next batch (cycle through dataset if needed)
            try:
                batch = next(train_dataloader_iter)
            except StopIteration:
                train_dataloader_iter = iter(train_dataloader)
                batch = next(train_dataloader_iter)
            
            # Train one step
            metrics = train_step(
                model, batch, criterion, optimizer, device, step, args.log_every,
                age_probe=age_probe, age_preservation_weight=args.age_preservation_weight,
                age_loss_type=args.age_loss_type, brainiac_model=brainiac_model,
                mean=mean, std=std, scaler=scaler
            )
            
            # Accumulate metrics for averaging
            for key in train_metrics_accum:
                if key in metrics:
                    train_metrics_accum[key].append(metrics[key])
            
            # Update learning rate
            scheduler.step()
            
            # Update progress bar
            if step % 10 == 0:  # Update every 10 steps
                avg_loss = sum(train_metrics_accum['loss'][-100:]) / min(100, len(train_metrics_accum['loss']))
                avg_l0 = sum(train_metrics_accum['l0'][-100:]) / min(100, len(train_metrics_accum['l0']))
                avg_ev = sum(train_metrics_accum['explained_variance'][-100:]) / min(100, len(train_metrics_accum['explained_variance']))
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'L0': f'{avg_l0:.2f}',
                    'EV': f'{avg_ev:.3f}',
                    'step': step
                })
            
            # Validate periodically
            if val_dataloader is not None and step % args.validate_every == 0:
                val_metrics = validate(
                    model, val_dataloader, criterion, device,
                    age_probe=age_probe, age_preservation_weight=args.age_preservation_weight,
                    age_loss_type=args.age_loss_type, brainiac_model=brainiac_model,
                    mean=mean, std=std
                )
                
                # Compute averaged train metrics
                avg_train_metrics = {
                    'loss': sum(train_metrics_accum['loss'][-args.validate_every:]) / min(args.validate_every, len(train_metrics_accum['loss'])),
                    'recon_loss': sum(train_metrics_accum['recon_loss'][-args.validate_every:]) / min(args.validate_every, len(train_metrics_accum['recon_loss'])),
                    'sparsity_loss': sum(train_metrics_accum['sparsity_loss'][-args.validate_every:]) / min(args.validate_every, len(train_metrics_accum['sparsity_loss'])),
                    'l0': sum(train_metrics_accum['l0'][-args.validate_every:]) / min(args.validate_every, len(train_metrics_accum['l0'])),
                    'explained_variance': sum(train_metrics_accum['explained_variance'][-args.validate_every:]) / min(args.validate_every, len(train_metrics_accum['explained_variance'])),
                    'reconstruction_accuracy': sum(train_metrics_accum['reconstruction_accuracy'][-args.validate_every:]) / min(args.validate_every, len(train_metrics_accum['reconstruction_accuracy'])),
                    'feature_accuracy': sum(train_metrics_accum['feature_accuracy'][-args.validate_every:]) / min(args.validate_every, len(train_metrics_accum['feature_accuracy']))
                }
                if age_probe is not None and args.age_preservation_weight > 0:
                    avg_train_metrics['age_preservation_loss'] = sum(train_metrics_accum['age_loss'][-args.validate_every:]) / min(args.validate_every, len(train_metrics_accum['age_loss']))
                
                print_str = (f"\nStep {step:,}/{num_steps:,} - Train Loss: {avg_train_metrics['loss']:.4f}, "
                            f"Val Loss: {val_metrics['loss']:.4f}, "
                            f"L0: {avg_train_metrics['l0']:.2f}, "
                            f"EV: {avg_train_metrics['explained_variance']:.3f}")
                if age_probe is not None and args.age_preservation_weight > 0:
                    print_str += (f", Age Loss: {avg_train_metrics.get('age_preservation_loss', 0.0):.4f}")
                print(print_str)
                
                # Log validation metrics
                dict_size = model.n_dict_features
                val_sparsity_fraction = val_metrics['l0'] / dict_size
                val_log_dict = {
                    'val/loss': val_metrics['loss'],
                    'val/recon_loss': val_metrics['recon_loss'],
                    'val/sparsity_loss': val_metrics['sparsity_loss'],
                    'val/l0': val_metrics['l0'],
                    'val/sparsity_fraction': val_sparsity_fraction,  # L0 / dict_size (percentage)
                    'val/explained_variance': val_metrics['explained_variance'],
                    'val/correlation': val_metrics.get('correlation', 0.0),
                    'step': step
                }
                if age_probe is not None and args.age_preservation_weight > 0:
                    val_log_dict['val/age_preservation_loss'] = val_metrics.get('age_preservation_loss', 0.0)
                wandb.log(val_log_dict, step=step)
                
                # Save best model
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    checkpoint_path = os.path.join(args.output_dir, 'best_model.pt')
                    torch.save({
                        'step': step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_metrics': avg_train_metrics,
                        'val_metrics': val_metrics,
                        'normalization_stats': {'mean': mean, 'std': std},
                        'config': vars(args)
                    }, checkpoint_path)
                    print(f"Saved best model to {checkpoint_path}")
                    wandb.log({'val/best_loss': best_val_loss}, step=step)
            
            # Periodic checkpoint
            if step % args.save_every == 0:
                checkpoint_path = os.path.join(args.output_dir, f'checkpoint_step_{step}.pt')
                avg_train_metrics = {
                    'loss': sum(train_metrics_accum['loss'][-args.save_every:]) / min(args.save_every, len(train_metrics_accum['loss'])),
                    'recon_loss': sum(train_metrics_accum['recon_loss'][-args.save_every:]) / min(args.save_every, len(train_metrics_accum['recon_loss'])),
                    'sparsity_loss': sum(train_metrics_accum['sparsity_loss'][-args.save_every:]) / min(args.save_every, len(train_metrics_accum['sparsity_loss'])),
                    'l0': sum(train_metrics_accum['l0'][-args.save_every:]) / min(args.save_every, len(train_metrics_accum['l0'])),
                    'explained_variance': sum(train_metrics_accum['explained_variance'][-args.save_every:]) / min(args.save_every, len(train_metrics_accum['explained_variance'])),
                    'reconstruction_accuracy': sum(train_metrics_accum['reconstruction_accuracy'][-args.save_every:]) / min(args.save_every, len(train_metrics_accum['reconstruction_accuracy'])),
                    'feature_accuracy': sum(train_metrics_accum['feature_accuracy'][-args.save_every:]) / min(args.save_every, len(train_metrics_accum['feature_accuracy']))
                }
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_metrics': avg_train_metrics,
                    'normalization_stats': {'mean': mean, 'std': std},
                    'config': vars(args),
                    'age_preservation_enabled': age_probe is not None,
                    'age_preservation_weight': args.age_preservation_weight if age_probe is not None else 0.0
                }, checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")
    
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user. Saving final checkpoint...")
        step_str = str(step) if 'step' in locals() else 'unknown'
        checkpoint_path = os.path.join(args.output_dir, f'checkpoint_interrupted_step_{step_str}.pt')
        avg_train_metrics = {}
        if 'train_metrics_accum' in locals() and train_metrics_accum['loss']:
            avg_train_metrics = {
                'loss': sum(train_metrics_accum['loss'][-100:]) / min(100, len(train_metrics_accum['loss'])),
                'recon_loss': sum(train_metrics_accum['recon_loss'][-100:]) / min(100, len(train_metrics_accum['recon_loss'])),
                'sparsity_loss': sum(train_metrics_accum['sparsity_loss'][-100:]) / min(100, len(train_metrics_accum['sparsity_loss'])),
                'l0': sum(train_metrics_accum['l0'][-100:]) / min(100, len(train_metrics_accum['l0'])),
                'explained_variance': sum(train_metrics_accum['explained_variance'][-100:]) / min(100, len(train_metrics_accum['explained_variance']))
            }
        torch.save({
            'step': step if 'step' in locals() else 0,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_metrics': avg_train_metrics,
            'normalization_stats': {'mean': mean, 'std': std} if 'mean' in locals() and 'std' in locals() else {},
            'config': vars(args),
            'age_preservation_enabled': age_probe is not None,
            'age_preservation_weight': args.age_preservation_weight if age_probe is not None else 0.0,
            'interrupted': True
        }, checkpoint_path)
        print(f"Saved interrupted checkpoint to {checkpoint_path}")
        wandb.finish()
        raise
    
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        print("\nSaving error checkpoint...")
        step_str = str(step) if 'step' in locals() else 'unknown'
        checkpoint_path = os.path.join(args.output_dir, f'checkpoint_error_step_{step_str}.pt')
        try:
            torch.save({
                'step': step if 'step' in locals() else 0,
                'model_state_dict': model.state_dict() if 'model' in locals() else {},
                'optimizer_state_dict': optimizer.state_dict() if 'optimizer' in locals() else {},
                'normalization_stats': {'mean': mean, 'std': std} if 'mean' in locals() and 'std' in locals() else {},
                'config': vars(args),
                'age_preservation_enabled': age_probe is not None if 'age_probe' in locals() else False,
                'age_preservation_weight': args.age_preservation_weight if 'age_probe' in locals() and age_probe is not None else 0.0,
                'error': str(e),
                'error_type': type(e).__name__
            }, checkpoint_path)
            print(f"Saved error checkpoint to {checkpoint_path}")
        except:
            print("Could not save error checkpoint")
        wandb.finish()
        raise
    
    print("\nTraining complete!")
    
    # Finish wandb run
    wandb.finish()


if __name__ == "__main__":
    main()


