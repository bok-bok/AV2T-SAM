import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from model.avt_sam import AVTSAM
from model.segment_anything_2.sam2.modeling.backbones.hieradet import (
    MultiScaleAttention,
)
from utils.avs_dataset_m3 import MS3Dataset_SAM


def visualize_attention_maps(model, batch, layer_indices=None, save_path=None):
    """
    Visualize attention maps from specified layers of the Hiera backbone.
    
    Args:
        model: The AVTSAM model
        batch: Tuple containing (sam_imgs_tensor, beit_imgs_tensor, clip_embeddings, 
               clap_embeddings, masks_tensor, original_size_list, video_name)
        layer_indices: List of layer indices to visualize attention from. 
                      Default indices: [1, 8, 23, 33, 43]
        save_path: Path to save the visualization
    """
    if layer_indices is None:
        layer_indices = [1, 8, 23, 33, 43]
    
    # Unpack batch
    sam_imgs_tensor, beit_imgs_tensor, clip_embeddings, clap_embeddings, _, original_size_list, _ = batch
    
    # Move to device
    device = next(model.parameters()).device
    sam_imgs_tensor = sam_imgs_tensor.to(device)
    beit_imgs_tensor = beit_imgs_tensor.to(device)
    clip_embeddings = clip_embeddings.to(device)
    clap_embeddings = clap_embeddings.to(device)
    
    # Set all attention modules to save attention weights
    def set_attention_flags(module):
        if isinstance(module, MultiScaleAttention):
            module.save_attention = True
            module.attention_weights = None
    
    model.apply(set_attention_flags)
    
    # Forward pass
    with torch.no_grad():
        _ = model(sam_imgs_tensor, beit_imgs_tensor, clip_embeddings, clap_embeddings, original_size_list)
    
    # Collect attention maps
    attention_maps = []
    blocks = model.model.visual_model.image_encoder.trunk.blocks
    for i, block in enumerate(blocks):
        if hasattr(block.attn, 'attention_weights') and block.attn.attention_weights is not None:
            if i in layer_indices:
                # Average attention weights across heads
                attn = block.attn.attention_weights.mean(1)  # Average across heads
                attention_maps.append((f"Layer {i}{' (Global)' if i in [23, 33, 43] else ''}", attn[0]))
    
    # Visualize
    n_maps = len(attention_maps)
    n_cols = min(3, n_maps)
    n_rows = (n_maps + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    if n_maps == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, (title, attn_map) in enumerate(attention_maps):
        ax = axes[idx]
        im = ax.imshow(attn_map.cpu().numpy(), cmap='viridis')
        ax.set_title(title)
        plt.colorbar(im, ax=ax)
    
    # Hide empty subplots
    for idx in range(len(attention_maps), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    # Reset attention flags
    def reset_attention_flags(module):
        if isinstance(module, MultiScaleAttention):
            module.save_attention = False
            module.attention_weights = None
    
    model.apply(reset_attention_flags)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument("--weight_path", type=str, default="")
    parser.add_argument("--projector_type", type=str, default="mul")
    parser.add_argument("--use_adapter", action="store_true")
    
    # Always use EVF-SAM2
    parser.add_argument("--evf_version", type=str, default="evf_sam2")
    parser.add_argument("--dataset", type=str, default="ms3")
    
    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model and weights
    model = AVTSAM(args)
    # model = load_model_with_weights(model, args.weight_path)

    if args.weight_path:
        print(f"Loading weights from {args.weight_path}")
        model.load_state_dict(torch.load(args.weight_path), strict=False)

    model = model.cuda()
    model.eval()

    # Create dataloader
    test_loader = DataLoader(
        MS3Dataset_SAM(split='test'), 
        batch_size=1, 
        shuffle=False, 
        num_workers=0,
        pin_memory=True
    )

    # Get first batch for visualization
    batch = next(iter(test_loader))
    
    # Visualize attention maps
    visualize_attention_maps(
        model, 
        batch,
        layer_indices=[1, 8, 23, 33, 43],  # Key layers including global attention blocks
        save_path='attention_maps.png'
    )
