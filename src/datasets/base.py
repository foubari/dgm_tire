"""
Base generic dataset class for multi-component conditional datasets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple, Optional, List, Union
from abc import ABC, abstractmethod

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class GenericDataset(Dataset, ABC):
    """
    Generic dataset base class for multi-component conditional datasets.
    
    This class handles:
    - Loading images from multiple component directories
    - Parsing CSV files with conditioning information
    - Flexible filename patterns and column names
    - Train/test splitting
    
    Subclasses should implement:
    - `_load_image()`: How to load and process individual images
    - `_process_condition()`: How to process condition vectors (optional override)
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
        Initialize generic dataset.
        
        Args:
            root_dir: Root directory containing component subdirectories
            component_dirs: List of component directory names (e.g., ["comp_a", "comp_b"])
            condition_csv: Path to CSV file with conditioning data
            condition_columns: List of column names to use for conditioning (e.g., ["param1", "param2"])
            prefix_column: Name of CSV column containing the prefix/ID (default: "matching")
            filename_pattern: Pattern for image filenames, supports {prefix} and {component} placeholders
                             (default: "{prefix}_{component}.png")
            split: "train" or "test"
            split_column: Name of CSV column for train/test split (default: "train")
            transform: Optional transform to apply to images
            stacked: If True, return stacked tensor (C, H, W); if False, return tuple
            normalized: If True, use normalized versions of condition columns (adds "_norm" suffix)
        """
        # Convert to absolute paths
        root_dir_path = Path(root_dir)
        if not root_dir_path.is_absolute():
            root_dir_path = root_dir_path.resolve()
        self.root_dir = root_dir_path
        self.component_dirs = list(component_dirs)
        self.condition_csv = Path(condition_csv)
        if not self.condition_csv.is_absolute():
            self.condition_csv = Path(root_dir_path).parent / self.condition_csv if not (Path(root_dir_path).parent / self.condition_csv).exists() else Path(root_dir_path).parent / self.condition_csv
        
        self.condition_columns = condition_columns
        self.prefix_column = prefix_column
        self.filename_pattern = filename_pattern
        self.split = split
        self.split_column = split_column
        self.transform = transform or transforms.ToTensor()
        self.stacked = stacked
        self.normalized = normalized
        
        # Initialize data
        self._init_conditional()
    
    def _init_conditional(self) -> None:
        """Load CSV and build (prefix, condition) pairs."""
        # Load CSV
        df = pd.read_csv(self.condition_csv)
        
        # Filter by split if split_column exists
        if self.split_column in df.columns:
            if self.split == "train":
                df = df.query(f"{self.split_column} == True")
            elif self.split == "test":
                df = df.query(f"{self.split_column} == False")
            else:
                raise ValueError(f"Unknown split: {self.split}, must be 'train' or 'test'")
        
        # Normalize prefix column to lowercase
        if self.prefix_column in df.columns:
            df = df.assign(**{self.prefix_column: df[self.prefix_column].str.lower()})
        
        # Sort by prefix for consistency
        df = df.sort_values(self.prefix_column)
        self.df = df
        
        # Build (prefix, condition) pairs
        self._pairs: list[Tuple[str, torch.Tensor]] = []
        for row in df.itertuples(index=False):
            prefix = getattr(row, self.prefix_column)
            condition = self._process_condition(row)
            self._pairs.append((prefix, condition))
        
        # Verify all components exist for each prefix
        self._verify_components()
    
    def _process_condition(self, row) -> torch.Tensor:
        """
        Process a CSV row into a condition tensor.
        
        Args:
            row: Named tuple from pandas DataFrame.itertuples()
        
        Returns:
            Condition tensor of shape (cond_dim,)
        """
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
                raise ValueError(f"Missing value in column '{col}' for prefix '{getattr(row, self.prefix_column)}'")
            
            condition_values.append(float(val))
        
        return torch.tensor(condition_values, dtype=torch.float32)
    
    def _verify_components(self) -> None:
        """Verify that all component directories exist and contain files for all prefixes."""
        for comp_dir in self.component_dirs:
            comp_path = self.root_dir / comp_dir
            if not comp_path.exists():
                raise FileNotFoundError(f"Component directory not found: {comp_path}")
        
        # Check that files exist for all prefixes
        missing_files = []
        for prefix, _ in self._pairs:
            for comp_dir in self.component_dirs:
                filename = self.filename_pattern.format(prefix=prefix, component=comp_dir)
                filepath = self.root_dir / comp_dir / filename
                if not filepath.exists():
                    missing_files.append((prefix, comp_dir, filename))
        
        if missing_files:
            missing_str = "\n".join([f"  {p}/{c}/{f}" for p, c, f in missing_files[:10]])
            if len(missing_files) > 10:
                missing_str += f"\n  ... and {len(missing_files) - 10} more"
            raise FileNotFoundError(
                f"Missing image files for {len(missing_files)} prefix/component combinations:\n{missing_str}"
            )
    
    @abstractmethod
    def _load_image(self, path: Path) -> torch.Tensor:
        """
        Load and process an image from path.
        
        Args:
            path: Path to image file
        
        Returns:
            Tensor representation of the image
        """
        pass
    
    def __len__(self) -> int:
        return len(self._pairs)
    
    def __getitem__(self, idx: int):
        """
        Get item at index.
        
        Returns:
            If stacked=True: (images_tensor, condition_tensor) where images_tensor is (C, H, W)
            If stacked=False: (images_tuple, condition_tensor) where images_tuple is tuple of (H, W) tensors
        """
        prefix, condition = self._pairs[idx]
        
        # Load images for all components
        images = []
        for comp_dir in self.component_dirs:
            filename = self.filename_pattern.format(prefix=prefix, component=comp_dir)
            filepath = self.root_dir / comp_dir / filename
            img = self._load_image(filepath)
            images.append(img)
        
        # Stack or return as tuple
        if self.stacked:
            images_tensor = torch.cat(images, dim=0)  # (C, H, W)
            return images_tensor, condition
        else:
            return tuple(images), condition

