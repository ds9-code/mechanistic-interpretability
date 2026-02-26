import os
import torch
import numpy as np
import nibabel as nib
import yaml
from monai.transforms import (
    Compose, LoadImaged, Resized, NormalizeIntensityd, ToTensord
)
from train_lightning_idh import DualInputBinaryClassificationLightningModule

# Add safe globals for MONAI MetaTensor and numpy objects to handle PyTorch 2.6 compatibility
try:
    from monai.data.meta_tensor import MetaTensor
    torch.serialization.add_safe_globals([MetaTensor])
except ImportError:
    pass

# Add numpy safe globals
try:
    from numpy.core.multiarray import _reconstruct
    torch.serialization.add_safe_globals([_reconstruct])
except ImportError:
    pass

# ---- USER-DEFINED PATHS AND SETTINGS ----
# IMPORTANT: Please update these paths before running the script.
nifti_path = ""  # Single image for saliency generation
checkpoint_path = ""
config_path = ""
output_dir = ""
layer = -1  # Transformer layer index to visualize (-1 for last layer)
img_size = (96, 96, 96)  # Input image size (DxHxW), should match training
patch_size = 16  # ViT patch size, should match training
# -----------------------------------------

def get_preprocessing_transform(img_size):
    """Returns the MONAI preprocessing transforms for the input image."""
    return Compose([
        LoadImaged(keys=["image"], ensure_channel_first=True),
        Resized(keys=["image"], spatial_size=img_size),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        ToTensord(keys=["image"])
    ])

def extract_attention_map(vit_model, image, layer_idx=-1, img_size=(96, 96, 96), patch_size=16):
    """
    Extracts the attention map from a Vision Transformer (ViT) model.

    This function wraps the attention blocks of the ViT to capture the attention
    weights during a forward pass. It then processes these weights to generate
    a 3D saliency map corresponding to the model's focus on the input image.
    """
    attention_maps = {}

    # A wrapper class to intercept and store attention weights from a ViT block.
    class AttentionWithWeights(torch.nn.Module):
        def __init__(self, original_attn_module):
            super().__init__()
            self.original_attn_module = original_attn_module
            self.attn_weights = None

        def forward(self, x):
            # The original implementation of the attention module may not return
            # the attention weights. This wrapper recalculates them to ensure they
            # are captured. This is based on the standard ViT attention mechanism.
            output = self.original_attn_module(x)
            if hasattr(self.original_attn_module, 'qkv'):
                qkv = self.original_attn_module.qkv(x)
                batch_size, seq_len, _ = x.shape
                # Assuming qkv has been fused and has shape (batch_size, seq_len, 3 * num_heads * head_dim)
                qkv = qkv.reshape(batch_size, seq_len, 3, self.original_attn_module.num_heads, -1)
                qkv = qkv.permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]
                attn = (q @ k.transpose(-2, -1)) * self.original_attn_module.scale
                self.attn_weights = attn.softmax(dim=-1)
            return output

    # Replace the attention module in each block with our wrapper
    for i, block in enumerate(vit_model.blocks):
        if hasattr(block, 'attn'):
            block.attn = AttentionWithWeights(block.attn)

    # Perform a forward pass to execute the wrapped modules and capture weights
    with torch.no_grad():
        _ = vit_model(image)

    # Collect the captured attention weights from each block
    for i, block in enumerate(vit_model.blocks):
        if hasattr(block.attn, 'attn_weights') and block.attn.attn_weights is not None:
            attention_maps[f"layer_{i}"] = block.attn.attn_weights.detach()

    if not attention_maps:
        raise RuntimeError("Could not extract any attention maps. Please check the ViT model structure.")

    # Select the attention map from the specified layer
    if layer_idx < 0:
        layer_idx = len(attention_maps) + layer_idx
    layer_name = f"layer_{layer_idx}"
    if layer_name not in attention_maps:
        raise ValueError(f"Layer {layer_idx} not found. Available layers: {list(attention_maps.keys())}")

    layer_attn = attention_maps[layer_name]
    # Average attention across all heads
    head_attn = layer_attn[0].mean(dim=0)
    # Get attention from the [CLS] token to all other image patches
    cls_attn = head_attn[0, 1:]

    # Reshape the 1D attention vector into a 3D volume
    patches_per_dim = img_size[0] // patch_size
    total_patches = patches_per_dim ** 3
    
    # Pad or truncate if the number of patches doesn't align
    if cls_attn.shape[0] != total_patches:
        if cls_attn.shape[0] > total_patches:
            cls_attn = cls_attn[:total_patches]
        else:
            padded = torch.zeros(total_patches, device=cls_attn.device)
            padded[:cls_attn.shape[0]] = cls_attn
            cls_attn = padded

    cls_attn_3d = cls_attn.reshape(patches_per_dim, patches_per_dim, patches_per_dim)
    cls_attn_3d = cls_attn_3d.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims

    # Upsample the attention map to the full image resolution
    upsampled_attn = torch.nn.functional.interpolate(
        cls_attn_3d,
        size=img_size,
        mode='trilinear',
        align_corners=False
    ).squeeze()

    # Normalize the map to [0, 1] for visualization
    upsampled_attn = upsampled_attn.cpu().numpy()
    upsampled_attn = (upsampled_attn - upsampled_attn.min()) / (upsampled_attn.max() - upsampled_attn.min())
    return upsampled_attn

def main():
    """Main function to load the model, generate, and save the saliency map."""
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cpu")

    # Load the YAML config file to instantiate the model correctly
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load the trained model from the checkpoint
    print(f"Loading model from checkpoint: {checkpoint_path}")
    
    # First, load the checkpoint manually to handle PyTorch 2.6 compatibility
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Create the model instance
    model = DualInputBinaryClassificationLightningModule(config)
    
    # Load the state dict
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'], strict=True)
    else:
        model.load_state_dict(checkpoint, strict=True)
    
    model.to(device)
    model.eval()

    # Extract the ViT backbone from the Lightning module
    vit_model = model.backbone.backbone

    # Preprocess the input NIfTI image
    transforms = get_preprocessing_transform(img_size)
    print(f"Loading and preprocessing image: {nifti_path}")
    image_dict = transforms({"image": nifti_path})
    image = image_dict["image"].unsqueeze(0).to(device)

    # Generate the saliency map
    print("Extracting attention map...")
    attn_map = extract_attention_map(vit_model, image, layer_idx=layer, img_size=img_size, patch_size=patch_size)
    print("...extraction complete.")

    # Save the preprocessed input and the saliency map as NIfTI files
    base_filename = os.path.basename(nifti_path).split('.')[0]
    checkpoint_name = os.path.basename(checkpoint_path).split('.')[0]
    
    input_nifti = nib.Nifti1Image(image.cpu().squeeze().numpy(), np.eye(4))
    saliency_nifti = nib.Nifti1Image(attn_map, np.eye(4)) # Using identity affine
    
    input_save_path = os.path.join(output_dir, f"{base_filename}_{checkpoint_name}_input.nii.gz")
    saliency_save_path = os.path.join(output_dir, f"{base_filename}_{checkpoint_name}_saliencymap_layer{layer}.nii.gz")
    
    nib.save(input_nifti, input_save_path)
    nib.save(saliency_nifti, saliency_save_path)
    
    print(f"Successfully saved input image to: {input_save_path}")
    print(f"Successfully saved saliency map to: {saliency_save_path}")

if __name__ == "__main__":
    main() 