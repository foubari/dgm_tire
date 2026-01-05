"""
Dataset-specific configuration for EPURE and TOY datasets.
"""

from dataclasses import dataclass
from typing import List, Tuple
from pathlib import Path


@dataclass
class DatasetConfig:
    """Configuration spécifique à un dataset."""
    name: str
    components: List[str]
    image_size: Tuple[int, int]
    num_classes: int  # Including background
    data_root: Path
    conditions_csv: Path
    condition_columns: List[str]
    has_geometric_validation: bool = False  # Pour vérification géométrique (toy uniquement)


# EPURE Dataset
EPURE_CONFIG = DatasetConfig(
    name="epure",
    components=["group_nc", "group_km", "bt", "fpu", "tpc"],
    image_size=(64, 32),
    num_classes=6,  # 5 components + background
    data_root=Path("data/epure"),
    conditions_csv=Path("data/epure/performances.csv"),
    condition_columns=["d_cons_norm", "d_rigid_norm", "d_life_norm", "d_stab_norm"],
    has_geometric_validation=False
)

# TOY Dataset
TOY_CONFIG = DatasetConfig(
    name="toy",
    components=["group_nc", "group_km", "fpu"],
    image_size=(64, 32),
    num_classes=4,  # 3 components + background
    data_root=Path("data/toy_epure"),
    conditions_csv=Path("data/toy_epure/performances.csv"),
    condition_columns=["d_cons_norm", "d_rigid_norm", "d_life_norm", "d_stab_norm"],
    has_geometric_validation=True  # TOY a validation géométrique
)


def get_dataset_config(dataset_name: str) -> DatasetConfig:
    """Retourne config selon nom du dataset."""
    configs = {
        "epure": EPURE_CONFIG,
        "toy": TOY_CONFIG
    }
    if dataset_name not in configs:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(configs.keys())}")
    return configs[dataset_name]
