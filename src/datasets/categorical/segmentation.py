"""
Segmentation dataset for MDM (categorical masks).
"""

import os
from pathlib import Path
from typing import Sequence, Optional, List, Union, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image


class SegmentationDataset(Dataset):
    """
    Dataset for MDM that loads categorical segmentation masks.
    
    Supports two modes:
    1. From numpy files: Loads pre-computed masks from .npy files (faster)
    2. From images: Loads masks from individual image files
    
    Returns:
        (mask, condition) where:
        - mask: LongTensor of shape (1, H, W) with class indices [0, num_classes-1]
        - condition: FloatTensor of shape (cond_dim,) with conditioning values
    """
    
    def __init__(
        self,
        root_dir: Union[str, Path],
        condition_csv: Union[str, Path],
        condition_columns: List[str],
        prefix_column: str = "matching",
        split: str = "train",
        split_column: str = "train",
        resolution: Tuple[int, int] = (64, 32),
        mask_format: str = "numpy",  # "numpy" or "images"
        mask_path: Optional[str] = None,  # Path to .npy file or directory with images
        filename_pattern: str = "{prefix}.png",  # For image format
        normalized: bool = False,
    ) -> None:
        """
        Initialize segmentation dataset for MDM.
        
        Args:
            root_dir: Root directory containing data
            condition_csv: Path to CSV file with conditioning data
            condition_columns: List of column names to use for conditioning
            prefix_column: Name of CSV column containing the prefix/ID
            split: "train", "val", or "test"
            split_column: Name of CSV column for train/test split
            resolution: Tuple (H, W) for mask resolution
            mask_format: "numpy" to load from .npy file, "images" to load from individual images
            mask_path: Path to .npy file (if mask_format="numpy") or directory with images (if mask_format="images")
            filename_pattern: Pattern for image filenames (only used if mask_format="images")
            normalized: If True, use normalized versions of condition columns
        """
        self.root_dir = Path(root_dir)
        if not self.root_dir.is_absolute():
            self.root_dir = self.root_dir.resolve()
        
        self.condition_csv = Path(condition_csv)
        if not self.condition_csv.is_absolute():
            # Try relative to root_dir first, then parent
            if (self.root_dir / self.condition_csv).exists():
                self.condition_csv = self.root_dir / self.condition_csv
            elif (self.root_dir.parent / self.condition_csv).exists():
                self.condition_csv = self.root_dir.parent / self.condition_csv
            else:
                self.condition_csv = Path(condition_csv).resolve()
        
        self.condition_columns = condition_columns
        self.prefix_column = prefix_column
        self.split = split
        self.split_column = split_column
        self.resolution = resolution
        self.mask_format = mask_format
        self.mask_path = mask_path
        self.filename_pattern = filename_pattern
        self.normalized = normalized
        
        # Load CSV
        self._load_csv()
        
        # Load masks
        self._load_masks()
        
        # Verify consistency
        if len(self.df) != len(self.masks):
            raise ValueError(
                f"Mismatch between CSV rows ({len(self.df)}) and masks ({len(self.masks)}) "
                f"for split '{split}'. The CSV and masks must have the same number of entries."
            )
    
    def _load_csv(self) -> None:
        """Load and filter CSV."""
        df = pd.read_csv(self.condition_csv)
        
        # Filter by split if split_column exists
        if self.split_column in df.columns:
            if self.split == "train":
                df = df.query(f"{self.split_column} == True")
            elif self.split in ("test", "val"):
                df = df.query(f"{self.split_column} == False")
            else:
                raise ValueError(f"Unknown split: {self.split}, must be 'train', 'test', or 'val'")
        
        # Normalize prefix column to lowercase
        if self.prefix_column in df.columns:
            df = df.assign(**{self.prefix_column: df[self.prefix_column].str.lower()})
        
        # Sort by prefix for consistency
        df = df.sort_values(self.prefix_column)
        self.df = df.reset_index(drop=True)
    
    def _load_masks(self) -> None:
        """Load masks from numpy file or images."""
        H, W = self.resolution
        
        if self.mask_format == "numpy":
            # Load from .npy file
            if self.mask_path is None:
                # Default: look for {split}_{H}x{W}.npy in preprocessed/
                npy_path = self.root_dir / "preprocessed" / f"{self.split}_{H}x{W}.npy"
            else:
                npy_path = Path(self.mask_path)
                if not npy_path.is_absolute():
                    # mask_path is relative to root_dir (not preprocessed/)
                    npy_path = self.root_dir / npy_path
            
            if not npy_path.exists():
                raise FileNotFoundError(f"Mask numpy file not found: {npy_path}")
            
            # Load masks: shape should be (N, H, W) or (N, 1, H, W)
            masks_np = np.load(npy_path)
            if len(masks_np.shape) == 4 and masks_np.shape[1] == 1:
                masks_np = masks_np.squeeze(1)  # (N, 1, H, W) -> (N, H, W)
            
            self.masks = torch.from_numpy(masks_np).long()
            
        elif self.mask_format == "images":
            # Load from individual image files
            if self.mask_path is None:
                mask_dir = self.root_dir / "masks" / self.split
            else:
                mask_dir = Path(self.mask_path)
                if not mask_dir.is_absolute():
                    mask_dir = self.root_dir / mask_dir
            
            if not mask_dir.exists():
                raise FileNotFoundError(f"Mask directory not found: {mask_dir}")
            
            # Load masks in CSV order
            masks_list = []
            for prefix in self.df[self.prefix_column]:
                filename = self.filename_pattern.format(prefix=prefix)
                mask_path = mask_dir / filename
                
                if not mask_path.exists():
                    raise FileNotFoundError(f"Mask file not found: {mask_path}")
                
                # Load as grayscale and convert to class indices
                mask_img = Image.open(mask_path).convert('L')
                mask_img = mask_img.resize((W, H), Image.NEAREST)
                mask_array = np.array(mask_img)
                
                # Convert to class indices (assuming pixel values are class indices)
                masks_list.append(torch.from_numpy(mask_array).long())
            
            self.masks = torch.stack(masks_list)  # (N, H, W)
        
        else:
            raise ValueError(f"Unknown mask_format: {self.mask_format}, must be 'numpy' or 'images'")
    
    def _process_condition(self, row) -> torch.Tensor:
        """Process a CSV row into a condition tensor."""
        condition_values = []
        
        for col in self.condition_columns:
            # Try normalized version if normalized=True
            if self.normalized:
                norm_col = f"{col}_norm"
                if hasattr(row, norm_col):
                    val = getattr(row, norm_col)
                else:
                    # Fallback to non-normalized
                    val = getattr(row, col)
            else:
                val = getattr(row, col)
            
            # Convert to float
            if pd.isna(val):
                raise ValueError(
                    f"Missing value in column '{col}' for prefix '{getattr(row, self.prefix_column)}'"
                )
            
            condition_values.append(float(val))
        
        return torch.tensor(condition_values, dtype=torch.float32)
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int):
        """
        Get item at index.
        
        Returns:
            (mask, condition) where:
            - mask: LongTensor of shape (1, H, W) with class indices
            - condition: FloatTensor of shape (cond_dim,) with conditioning values
        """
        row = self.df.iloc[idx]
        
        # Get mask: shape (H, W) -> (1, H, W)
        mask = self.masks[idx].unsqueeze(0)
        
        # Get condition
        condition = self._process_condition(row)
        
        return mask, condition

