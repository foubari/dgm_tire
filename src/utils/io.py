"""
Generic image I/O utilities for saving component images.
"""

import os
import numpy as np
import torch
import imageio
from typing import List


def save_component_images(
    samples: torch.Tensor,
    output_dir: str,
    prefix: str,
    component_names: List[str],
    threshold: float = 0.5,
    save_grayscale: bool = True,
):
    """
    Save stacked samples as individual component images.
    
    Args:
        samples: Tensor of shape (B, C, H, W) in [0, 1] range
        output_dir: Output directory root
        prefix: Filename prefix (e.g., "img0000")
        component_names: List of component names (length must match C)
        threshold: Threshold for binary images (default: 0.5)
        save_grayscale: If True, also save grayscale versions for debugging
    """
    # Ensure samples are in [0, 1] range
    if isinstance(samples, torch.Tensor):
        samples = torch.clamp(samples, 0.0, 1.0)
        samples_np = samples.cpu().numpy()
    else:
        samples_np = np.array(samples)
        samples_np = np.clip(samples_np, 0.0, 1.0)
    
    B, C, H, W = samples_np.shape
    
    if len(component_names) != C:
        raise ValueError(
            f"Number of component_names ({len(component_names)}) does not match "
            f"number of channels ({C})"
        )
    
    # Create output directories
    os.makedirs(os.path.join(output_dir, "full"), exist_ok=True)
    for comp_name in component_names:
        os.makedirs(os.path.join(output_dir, comp_name), exist_ok=True)
    
    # Save images
    for b in range(B):
        # Save full (all components combined)
        full_img = samples_np[b].max(axis=0)  # (H, W) - max across channels
        
        # Binary version (threshold)
        full_bin = (full_img > threshold).astype(np.uint8) * 255
        imageio.imwrite(
            os.path.join(output_dir, "full", f"{prefix}_{b:02d}.png"),
            full_bin
        )
        
        # Grayscale version (for debugging)
        if save_grayscale:
            full_gray = (full_img * 255).astype(np.uint8)
            imageio.imwrite(
                os.path.join(output_dir, "full", f"{prefix}_{b:02d}_gray.png"),
                full_gray
            )
        
        # Save per-component images
        for comp_idx, comp_name in enumerate(component_names):
            comp_img = samples_np[b, comp_idx]  # (H, W)
            
            # Binary version
            comp_bin = (comp_img > threshold).astype(np.uint8) * 255
            imageio.imwrite(
                os.path.join(output_dir, comp_name, f"{prefix}_{b:02d}.png"),
                comp_bin
            )
            
            # Grayscale version
            if save_grayscale:
                comp_gray = (comp_img * 255).astype(np.uint8)
                imageio.imwrite(
                    os.path.join(output_dir, comp_name, f"{prefix}_{b:02d}_gray.png"),
                    comp_gray
                )

