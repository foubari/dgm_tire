"""
Multi-component dataset for DDPM (grayscale images stacked along channel dimension).
"""

from pathlib import Path
from typing import Sequence, Optional, Union, List

import torch
from torchvision import transforms
from PIL import Image

from ..base import GenericDataset


class MultiComponentDataset(GenericDataset):
    """
    Dataset for DDPM that loads k grayscale images (one per component) and stacks them.
    
    Returns:
        - If stacked=True: (images, condition) where images is (k, H, W) tensor
        - If stacked=False: (images_tuple, condition) where images_tuple is tuple of (H, W) tensors
    """
    
    def __init__(
        self,
        root_dir: Union[str, Path],
        component_dirs: Sequence[str],
        condition_csv: Union[str, Path],
        condition_columns: List[str],
        prefix_column: str = "matching",
        filename_pattern: str = "{prefix}_{component}.png",
        split: str = "train",
        split_column: str = "train",
        transform: Optional[transforms.Compose] = None,
        stacked: bool = True,
        normalized: bool = False,
    ) -> None:
        """
        Initialize multi-component dataset for DDPM.
        
        Args:
            root_dir: Root directory containing component subdirectories
            component_dirs: List of component directory names
            condition_csv: Path to CSV file with conditioning data
            condition_columns: List of column names to use for conditioning
            prefix_column: Name of CSV column containing the prefix/ID
            filename_pattern: Pattern for image filenames
            split: "train" or "test"
            split_column: Name of CSV column for train/test split
            transform: Optional transform to apply to images (default: ToTensor())
            stacked: If True, return stacked tensor; if False, return tuple
            normalized: If True, use normalized versions of condition columns
        """
        # Default transform: convert to grayscale and tensor
        if transform is None:
            transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
            ])
        
        super().__init__(
            root_dir=root_dir,
            component_dirs=component_dirs,
            condition_csv=condition_csv,
            condition_columns=condition_columns,
            prefix_column=prefix_column,
            filename_pattern=filename_pattern,
            split=split,
            split_column=split_column,
            transform=transform,
            stacked=stacked,
            normalized=normalized,
        )
    
    def _load_image(self, path: Path) -> torch.Tensor:
        """
        Load a grayscale image as a tensor in range [0, 1].
        
        Args:
            path: Path to image file
        
        Returns:
            Tensor of shape (1, H, W) in range [0, 1]
        """
        with Image.open(path) as img:
            # Ensure grayscale
            if img.mode != 'L':
                img = img.convert('L')
            # Apply transform (includes ToTensor which normalizes to [0, 1])
            return self.transform(img)

