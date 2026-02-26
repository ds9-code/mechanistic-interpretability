"""
Train Linear Regressor for Brain Age Prediction on Raw CLS Features
Trains a simple linear layer on raw CLS token features from BrainIAC (768-dim)
This creates the baseline linear probe used for mechanistic interpretability experiments.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import argparse
import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt


class LinearRegressor(nn.Module):
    """Simple linear regressor for brain age prediction"""
    def __init__(self, input_dim=768):
        super(LinearRegressor, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        return self.linear(x).squeeze(-1)


def load_features(features_path):
    """Load CLS features from .pt file
    
    Expected format: {'features': tensor/array, 'labels': array, 'normalization_stats': dict (optional)}
    Handles both torch tensors and numpy arrays
    """
    data = torch.load(features_path, map_location='cpu', weights_only=False)
    features = data['features']  # [n_samples, n_features] - can be torch tensor or numpy array
    labels = data['labels']  # [n_samples] - age in months
    
    # Convert features to tensor if needed (FIXES Issue C: handle numpy arrays)
    if isinstance(features, np.ndarray):
        features = torch.from_numpy(features).float()
    elif isinstance(features, torch.Tensor):
        features = features.float()
    
    # Convert labels to tensor if needed
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels).float()
    elif isinstance(labels, torch.Tensor):
        labels = labels.float()
    
    # Check if normalization stats are available
    # For CLS features, normalization stats should be computed from the training set
    # and saved in the features file for consistent normalization
    normalization_stats = data.get('normalization_stats', None)
    
    return features, labels, normalization_stats


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for features, labels in tqdm(dataloader, desc="Training"):
        features = features.to(device)
        labels = labels.to(device)
        
        # Forward pass
        predictions = model(features)
        loss = criterion(predictions, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in tqdm(dataloader, desc="Validating"):
            features = features.to(device)
            labels = labels.to(device)
            
            predictions = model(features)
            loss = criterion(predictions, labels)
            
            total_loss += loss.item()
            all_predictions.append(predictions.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    all_predictions = np.concatenate(all_predictions)
    all_labels = np.concatenate(all_labels)
    
    # Calculate metrics
    mae = np.mean(np.abs(all_predictions - all_labels))
    mse = np.mean((all_predictions - all_labels) ** 2)
    rmse = np.sqrt(mse)
    
    # R² score
    ss_res = np.sum((all_labels - all_predictions) ** 2)
    ss_tot = np.sum((all_labels - np.mean(all_labels)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return total_loss / len(dataloader), mae, mse, rmse, r2


def main():
    parser = argparse.ArgumentParser(description='Train linear regressor on raw CLS features for brain age prediction')
    parser.add_argument('--train_features', type=str, required=True,
                       help='Path to train CLS features .pt file (from extract_raw_cls_features_for_linear_probe.py)')
    parser.add_argument('--val_features', type=str, required=True,
                       help='Path to validation CLS features .pt file')
    parser.add_argument('--test_features', type=str, default=None,
                       help='Optional: Path to test CLS features .pt file')
    parser.add_argument('--output_dir', type=str, default='./linear_probe_results',
                       help='Output directory for model and results')
    parser.add_argument('--results_json', type=str, default=None,
                       help='Optional: Path to save results JSON (e.g., baseline_cls_results.json)')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay (L2 regularization)')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--use_mse_loss', action='store_true',
                       help='Use MSE loss instead of MAE (aligns better with R² metric)')
    parser.add_argument('--no_normalize', action='store_true',
                       help='Do not normalize features (use raw; can help if probe predictions are wrong scale)')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load features
    print("\n" + "="*60)
    print("Loading Features")
    print("="*60)
    
    print("Loading train features...")
    train_features, train_labels, train_norm_stats = load_features(args.train_features)
    print(f"  Train: {train_features.shape[0]} samples, {train_features.shape[1]} features")
    
    print("Loading validation features...")
    val_features, val_labels, val_norm_stats = load_features(args.val_features)
    print(f"  Val: {val_features.shape[0]} samples, {val_features.shape[1]} features")
    
    # Load test features if provided
    test_features, test_labels, test_norm_stats = None, None, None
    if args.test_features:
        print("Loading test features...")
        test_features, test_labels, test_norm_stats = load_features(args.test_features)
        print(f"  Test: {test_features.shape[0]} samples, {test_features.shape[1]} features")
    
    # Normalize features using saved normalization stats (unless --no_normalize)
    # For CLS features: normalize using statistics computed from the training set
    # This ensures consistent normalization across train/val/test splits
    if args.no_normalize:
        print("\n--no_normalize: using raw features (no normalization)")
    elif train_norm_stats is not None:
        # Use saved normalization stats (from extraction)
        print("\n✓ Normalization stats found. Normalizing features using saved stats...")
        mean = train_norm_stats.get('mean', None)
        std = train_norm_stats.get('std', None)
        if mean is not None and std is not None:
            # Convert to tensor if needed
            if isinstance(mean, np.ndarray):
                mean = torch.from_numpy(mean).float()
            elif not isinstance(mean, torch.Tensor):
                mean = torch.tensor(mean).float()
            if isinstance(std, np.ndarray):
                std = torch.from_numpy(std).float()
            elif not isinstance(std, torch.Tensor):
                std = torch.tensor(std).float()
            
            # Ensure correct shape
            if mean.dim() == 1:
                mean = mean.unsqueeze(0)
            if std.dim() == 1:
                std = std.unsqueeze(0)
            
            # Normalize train, val, and test features using saved stats
            train_features = (train_features - mean) / std
            val_features = (val_features - mean) / std
            if test_features is not None:
                test_features = (test_features - mean) / std
            print(f"  ✓ CLS features normalized using saved statistics")
            print(f"    Mean: {mean.mean().item():.6f}, Std: {std.mean().item():.6f}")
        else:
            print("  ⚠️  Normalization stats found but mean/std are None. Normalizing in current space...")
            train_mean = train_features.mean(dim=0, keepdim=True)
            train_std = train_features.std(dim=0, keepdim=True) + 1e-8
            train_features = (train_features - train_mean) / train_std
            val_features = (val_features - train_mean) / train_std
            if test_features is not None:
                test_features = (test_features - train_mean) / train_std
    else:
        # No normalization stats: normalize CLS features using train stats
        print("\n⚠️  No normalization stats found. Normalizing CLS features using train set statistics...")
        train_mean = train_features.mean(dim=0, keepdim=True)
        train_std = train_features.std(dim=0, keepdim=True) + 1e-8
        train_features = (train_features - train_mean) / train_std
        val_features = (val_features - train_mean) / train_std
        if test_features is not None:
            test_features = (test_features - train_mean) / train_std
        print("  ✓ CLS features normalized using training set statistics")
        print(f"    Mean: {train_mean.mean().item():.6f}, Std: {train_std.mean().item():.6f}")
    
    # Use original train/val split (do NOT re-split)
    # This preserves the original data split and prevents data leakage
    print("\nUsing original train/val/test split (no re-splitting)...")
    train_dataset = TensorDataset(train_features, train_labels)
    val_dataset = TensorDataset(val_features, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    print(f"  Train split: {len(train_dataset)} samples")
    print(f"  Val split: {len(val_dataset)} samples")
    
    # Create test loader if test features are provided
    test_loader = None
    if test_features is not None:
        test_dataset = TensorDataset(test_features, test_labels)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        print(f"  Test split: {len(test_dataset)} samples")
    
    # Initialize model
    input_dim = train_features.shape[1]
    model = LinearRegressor(input_dim=input_dim).to(device)
    print(f"\nModel: Linear({input_dim} -> 1)")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss function: MSE aligns better with R² metric (variance-based)
    if args.use_mse_loss:
        criterion = nn.MSELoss()  # MSE loss (aligns with R² metric)
        loss_name = "MSE"
        print("  Loss: MSE (aligns with R² metric)")
    else:
        criterion = nn.L1Loss()  # MAE loss
        loss_name = "MAE"
        print("  Loss: MAE (R² should be interpreted cautiously - it's variance-based)")
    
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Training loop
    print("\n" + "="*60)
    print("Training")
    print("="*60)
    
    best_val_mae = float('inf')
    best_epoch = 0
    patience_counter = 0
    train_losses = []
    val_losses = []
    val_maes = []
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        
        # Validate
        val_loss, val_mae, val_mse, val_rmse, val_r2 = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_maes.append(val_mae)
        
        print(f"  Train Loss ({loss_name}): {train_loss:.4f}")
        print(f"  Val Loss ({loss_name}): {val_loss:.4f}")
        print(f"  Val MAE: {val_mae:.4f} months")
        print(f"  Val RMSE: {val_rmse:.4f} months")
        print(f"  Val R²: {val_r2:.4f}")
        
        # Early stopping
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mae': val_mae,
                'val_r2': val_r2,
                'input_dim': input_dim
            }, os.path.join(args.output_dir, 'best_model.pt'))
            print(f"  ✓ Saved best model (Val MAE: {val_mae:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                print(f"Best model was at epoch {best_epoch+1} with Val MAE: {best_val_mae:.4f}")
                break
    
    # Load best model for final evaluation
    print("\n" + "="*60)
    print("Final Evaluation")
    print("="*60)
    
    checkpoint = torch.load(os.path.join(args.output_dir, 'best_model.pt'), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final validation metrics
    final_val_loss, final_val_mae, final_val_mse, final_val_rmse, final_val_r2 = validate(
        model, val_loader, criterion, device
    )
    
    print(f"Final Validation Metrics:")
    print(f"  MAE: {final_val_mae:.4f} months")
    print(f"  RMSE: {final_val_rmse:.4f} months")
    print(f"  R²: {final_val_r2:.4f}")
    
    # Evaluate on test set if provided
    final_test_mae, final_test_rmse, final_test_r2 = None, None, None
    if test_loader is not None:
        print("\nFinal Test Metrics:")
        _, final_test_mae, final_test_mse, final_test_rmse, final_test_r2 = validate(
            model, test_loader, criterion, device
        )
        print(f"  MAE: {final_test_mae:.4f} months")
        print(f"  RMSE: {final_test_rmse:.4f} months")
        print(f"  R²: {final_test_r2:.4f}")
    
    # Prepare results dictionary
    results = {
        'val_mae': float(final_val_mae),
        'val_rmse': float(final_val_rmse),
        'val_r2': float(final_val_r2),
    }
    
    if final_test_mae is not None:
        results['test_mae'] = float(final_test_mae)
        results['test_rmse'] = float(final_test_rmse)
        results['test_r2'] = float(final_test_r2)
    
    # Save results JSON if path provided
    if args.results_json:
        with open(args.results_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to: {args.results_json}")
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_maes': val_maes,
        'best_epoch': best_epoch,
        'best_val_mae': best_val_mae,
        'final_val_mae': final_val_mae,
        'final_val_rmse': final_val_rmse,
        'final_val_r2': final_val_r2,
        'num_train_samples': len(train_dataset),
        'num_val_samples': len(val_dataset),
        'input_dim': input_dim
    }
    
    if final_test_mae is not None:
        history['final_test_mae'] = final_test_mae
        history['final_test_rmse'] = final_test_rmse
        history['final_test_r2'] = final_test_r2
        history['num_test_samples'] = len(test_dataset) if test_loader is not None else 0
    
    # Convert numpy types to Python types for JSON serialization
    def convert_to_python_type(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_python_type(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_python_type(item) for item in obj]
        return obj
    
    history_converted = convert_to_python_type(history)
    
    with open(os.path.join(args.output_dir, 'training_history.json'), 'w') as f:
        json.dump(history_converted, f, indent=2)
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label=f'Train Loss ({loss_name})', alpha=0.7)
    plt.plot(val_losses, label=f'Val Loss ({loss_name})', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel(f'Loss ({loss_name})')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(val_maes, label='Val MAE', color='orange', alpha=0.7)
    plt.axhline(y=best_val_mae, color='red', linestyle='--', label=f'Best: {best_val_mae:.4f}')
    plt.xlabel('Epoch')
    plt.ylabel('MAE (months)')
    plt.title('Validation MAE')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Training complete!")
    print(f"  Model saved to: {os.path.join(args.output_dir, 'best_model.pt')}")
    print(f"  History saved to: {os.path.join(args.output_dir, 'training_history.json')}")
    print(f"  Curves saved to: {os.path.join(args.output_dir, 'training_curves.png')}")


if __name__ == "__main__":
    main()



