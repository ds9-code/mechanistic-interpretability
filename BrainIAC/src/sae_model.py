"""
Sparse Autoencoder (SAE) Model for BrainIAC Feature Decomposition
Based on SAE-Rad and gated SAE architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedSAE(nn.Module):
    """
    Gated Sparse Autoencoder for decomposing BrainIAC CLS token features
    into interpretable sparse features.
    
    Architecture:
    - Encoder: Linear layer with ReLU activation and gating mechanism
    - Decoder: Linear layer to reconstruct original features
    - Sparsity: L1 regularization on activations
    """
    
    def __init__(self, n_input_features=768, n_dict_features=None, expansion_factor=32):
        """
        Args:
            n_input_features: Dimension of input features (CLS token from ViT = 768)
            n_dict_features: Number of dictionary features (if None, uses expansion_factor)
            expansion_factor: Multiplier for dictionary size (n_dict_features = n_input_features * expansion_factor)
        """
        super(GatedSAE, self).__init__()
        
        if n_dict_features is None:
            n_dict_features = n_input_features * expansion_factor
        
        self.n_input_features = n_input_features
        self.n_dict_features = n_dict_features
        self.expansion_factor = expansion_factor
        
        # Encoder: maps input to dictionary features
        self.encoder = nn.Linear(n_input_features, n_dict_features, bias=False)
        
        # Gating mechanism: learns which features to activate
        self.gate = nn.Linear(n_input_features, n_dict_features, bias=True)
        
        # Decoder: reconstructs input from dictionary features
        self.decoder = nn.Linear(n_dict_features, n_input_features, bias=False)
        
        # Initialize decoder weights using geometric median initialization
        self._initialize_decoder()
        
    def _initialize_decoder(self):
        """Initialize decoder weights using geometric median (unit norm)"""
        with torch.no_grad():
            nn.init.normal_(self.decoder.weight, mean=0.0, std=1.0 / (self.n_input_features ** 0.5))
            # Normalize each decoder weight to unit norm
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, p=2, dim=1)
    
    def forward(self, x):
        """
        Forward pass through SAE
        
        Args:
            x: Input features [batch_size, n_input_features]
        
        Returns:
            reconstructed: Reconstructed input [batch_size, n_input_features]
            activations: Sparse activations [batch_size, n_dict_features]
            gate_values: Gate values [batch_size, n_dict_features]
        """
        # Encode to dictionary space
        encoded = self.encoder(x)  # [batch_size, n_dict_features]
        
        # Compute gate values
        gate_values = self.gate(x)  # [batch_size, n_dict_features]
        
        # Apply gating: element-wise multiplication with sigmoid gate
        # This allows selective activation of features
        gate_activations = torch.sigmoid(gate_values)
        activations = encoded * gate_activations
        
        # Apply ReLU for sparsity
        activations = F.relu(activations)
        
        # Decode back to input space
        reconstructed = self.decoder(activations)
        
        return reconstructed, activations, gate_values
    
    def get_l0_norm(self, activations):
        """
        Compute L0 norm (number of active features) for sparsity measurement
        
        Args:
            activations: Feature activations [batch_size, n_dict_features]
        
        Returns:
            l0: Average number of active features per sample
        """
        return (activations > 0).float().sum(dim=1).mean()
    
    def get_l1_norm(self, activations):
        """
        Compute L1 norm for sparsity regularization
        
        Args:
            activations: Feature activations [batch_size, n_dict_features]
        
        Returns:
            l1: Average L1 norm of activations
        """
        return activations.abs().sum(dim=1).mean()


class SAELoss(nn.Module):
    """
    Loss function for SAE training
    Combines reconstruction loss and sparsity regularization
    """
    
    def __init__(self, l1_coefficient=1e-3, reconstruction_loss_type='combined', mse_weight=0.9, cosine_weight=0.1):
        """
        Args:
            l1_coefficient: Weight for L1 sparsity regularization
            reconstruction_loss_type: Type of reconstruction loss ('mse', 'cosine', or 'combined')
            mse_weight: Weight for MSE component when using 'combined' loss (default: 0.9)
            cosine_weight: Weight for cosine component when using 'combined' loss (default: 0.1)
        """
        super(SAELoss, self).__init__()
        self.l1_coefficient = l1_coefficient
        self.reconstruction_loss_type = reconstruction_loss_type
        self.mse_weight = mse_weight
        self.cosine_weight = cosine_weight
        
    def forward(self, reconstructed, original, activations):
        """
        Compute SAE loss
        
        Args:
            reconstructed: Reconstructed features [batch_size, n_input_features]
            original: Original input features [batch_size, n_input_features]
            activations: Sparse activations [batch_size, n_dict_features]
        
        Returns:
            total_loss: Combined reconstruction + sparsity loss
            recon_loss: Reconstruction loss component
            sparsity_loss: Sparsity regularization component
        """


        # Reconstruction loss
        if self.reconstruction_loss_type == 'mse':
            recon_loss = F.mse_loss(reconstructed, original)
        elif self.reconstruction_loss_type == 'cosine':
            # Cosine similarity loss (1 - cosine_similarity)
            recon_loss = 1 - F.cosine_similarity(reconstructed, original, dim=1).mean()
        elif self.reconstruction_loss_type == 'combined':
            # Combined loss: mse_weight * MSE + cosine_weight * cosine_loss
            mse_loss = F.mse_loss(reconstructed, original)
            cosine_loss = 1 - F.cosine_similarity(reconstructed, original, dim=1).mean()
            recon_loss = self.mse_weight * mse_loss + self.cosine_weight * cosine_loss
        else:
            raise ValueError(f"Unknown reconstruction loss type: {self.reconstruction_loss_type}. "
                           f"Must be 'mse', 'cosine', or 'combined'")
        
       # Sparsity loss (L1 regularization)
        sparsity_loss = activations.abs().sum(dim=1).mean()
        
        # Total loss
        total_loss = recon_loss + self.l1_coefficient * sparsity_loss
        
        return total_loss, recon_loss, sparsity_loss


def compute_explained_variance(reconstructed, original):
    """
    Compute explained variance (R² score) for SAE reconstruction quality
    
    Args:
        reconstructed: Reconstructed features [batch_size, n_input_features]
        original: Original input features [batch_size, n_input_features]
    
    Returns:
        explained_variance: R² score (1.0 = perfect reconstruction)
    """
    # Flatten for computation
    recon_flat = reconstructed.flatten()
    orig_flat = original.flatten()
    
    # Compute variance
    total_variance = torch.var(orig_flat)
    residual_variance = torch.var(orig_flat - recon_flat)
    
    # Explained variance (R²)
    explained_variance = 1 - (residual_variance / (total_variance + 1e-8))
    
    return explained_variance.item()


