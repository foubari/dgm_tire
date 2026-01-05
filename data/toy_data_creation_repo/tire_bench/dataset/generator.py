"""Dataset generator with Latin Hypercube Sampling."""

from typing import Dict, Any, List, Optional
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import qmc
import yaml

from tire_bench.core.tire import Tire
from tire_bench.core.materials import MaterialProperties
from tire_bench.geometry.generator import (
    CarcassComponent,
    CrownComponent,
    FlanksComponent,
)
from tire_bench.metrics.registry import MetricRegistry
from tire_bench.mechanics.rigid_ring import RigidRingModel


class DatasetGenerator:
    """
    Generate tire datasets based on configuration.

    Supports multiple sampling strategies:
    - Latin Hypercube Sampling (better parameter space coverage)
    - Random sampling
    - Grid sampling

    Examples:
        >>> config = yaml.safe_load(open('dataset_config.yaml'))
        >>> generator = DatasetGenerator(config)
        >>> dataset = generator.generate()
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize dataset generator.

        Args:
            config: Configuration dictionary with dataset parameters
        """
        self.config = config
        self.output_dir = Path(config["dataset"]["output_dir"])
        self.num_samples = config["dataset"]["num_samples"]

        # Load metrics
        metric_names = config["dataset"].get("metrics", [])
        self.metrics = []
        for m in metric_names:
            try:
                self.metrics.append(MetricRegistry.create(m))
            except ValueError:
                print(f"Warning: Metric '{m}' not found, skipping")

        # Initialize mechanics model for deformation
        self.mechanics = RigidRingModel()

    def generate(self) -> pd.DataFrame:
        """
        Generate complete dataset.

        Returns:
            DataFrame with metadata for all generated samples
        """
        print(f"Generating dataset with {self.num_samples} samples...")
        print(f"Output directory: {self.output_dir}")

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Sample parameters
        print("Sampling parameters...")
        params_list = self._sample_parameters()

        # 2. Generate tires
        print("Generating tires...")
        metadata = []

        for idx, params in enumerate(params_list):
            if (idx + 1) % 100 == 0:
                print(f"  Generated {idx + 1}/{self.num_samples} samples")

            sample_data = self._generate_sample(idx, params)
            metadata.append(sample_data)

        # 3. Create DataFrame
        df = pd.DataFrame(metadata)

        # 4. Save metadata
        metadata_format = self.config["dataset"].get("metadata", {}).get("format", "csv")
        if metadata_format == "csv":
            metadata_path = self.output_dir / "metadata.csv"
            df.to_csv(metadata_path, index=False)
            print(f"Saved metadata to {metadata_path}")
        elif metadata_format == "json":
            metadata_path = self.output_dir / "metadata.json"
            df.to_json(metadata_path, orient="records", indent=2)
            print(f"Saved metadata to {metadata_path}")

        print(f"Dataset generation complete! {len(df)} samples generated.")
        return df

    def _sample_parameters(self) -> List[Dict[str, Any]]:
        """Sample parameter combinations based on strategy."""
        strategy = self.config["dataset"]["sampling"]["strategy"]
        ranges = self.config["dataset"]["parameter_ranges"]

        if strategy == "latin_hypercube":
            return self._latin_hypercube_sampling(ranges)
        elif strategy == "random":
            return self._random_sampling(ranges)
        elif strategy == "grid":
            return self._grid_sampling(ranges)
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")

    def _latin_hypercube_sampling(self, ranges: Dict) -> List[Dict]:
        """Latin Hypercube Sampling for better parameter space coverage."""
        seed = self.config["dataset"]["sampling"].get("seed", 42)
        sampler = qmc.LatinHypercube(d=len(ranges), seed=seed)

        # Generate samples in [0,1]^d
        samples = sampler.random(n=self.num_samples)

        # Scale to parameter ranges
        params_list = []
        param_names = list(ranges.keys())

        for sample in samples:
            params = {}
            for i, name in enumerate(param_names):
                param_config = ranges[name]

                if param_config["type"] == "uniform":
                    value = param_config["min"] + sample[i] * (
                        param_config["max"] - param_config["min"]
                    )
                    params[name] = value

                elif param_config["type"] == "choice":
                    idx = int(sample[i] * len(param_config["values"]))
                    idx = min(idx, len(param_config["values"]) - 1)
                    params[name] = param_config["values"][idx]

            params_list.append(params)

        return params_list

    def _random_sampling(self, ranges: Dict) -> List[Dict]:
        """Random sampling."""
        rng = np.random.RandomState(
            self.config["dataset"]["sampling"].get("seed", 42)
        )
        params_list = []

        for _ in range(self.num_samples):
            params = {}
            for name, param_config in ranges.items():
                if param_config["type"] == "uniform":
                    value = rng.uniform(param_config["min"], param_config["max"])
                    params[name] = value
                elif param_config["type"] == "choice":
                    value = rng.choice(param_config["values"])
                    params[name] = value

            params_list.append(params)

        return params_list

    def _grid_sampling(self, ranges: Dict) -> List[Dict]:
        """Grid sampling (all combinations)."""
        # This would create a grid of all parameter combinations
        # For simplicity, using random for now
        return self._random_sampling(ranges)

    def _generate_sample(self, idx: int, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate single tire sample."""
        # Create tire from parameters
        tire = self._create_tire_from_params(params)

        # Create sample directory
        sample_dir = self.output_dir / f"sample_{idx:05d}"
        if self.config["dataset"]["outputs"].get("save_images", True):
            sample_dir.mkdir(parents=True, exist_ok=True)

            # Save component images
            for comp_name, component in tire.components.items():
                img_path = sample_dir / f"{comp_name}.png"
                self._save_image(component.image, img_path)

        # Compute metrics
        metric_values = {}
        for metric in self.metrics:
            value = metric.compute(tire)
            metric_values[metric.name] = value

        # Generate deformed versions if requested
        if self.config["dataset"]["outputs"].get("deformation", {}).get("enabled", False):
            deform_data = self._generate_deformed(tire, sample_dir, idx)
            metric_values.update(deform_data)

        # Compile metadata
        sample_data = {
            "sample_id": idx,
            "sample_dir": str(sample_dir.relative_to(self.output_dir)),
            **params,
            **metric_values,
        }

        return sample_data

    def _create_tire_from_params(self, params: Dict) -> Tire:
        """Create tire from parameter dict."""
        resolution = params.get("resolution", 64)
        tire = Tire(resolution=resolution)

        # Create materials
        mat_carcass = MaterialProperties(
            E=params.get("E_carcass", 1.0),
            rho=params.get("rho_carcass", 1.0),
            name="carcass",
        )
        mat_crown = MaterialProperties(
            E=params.get("E_crown", 0.8),
            rho=params.get("rho_crown", 1.2),
            name="crown",
        )
        mat_flanks = MaterialProperties(
            E=params.get("E_flanks", 0.5),
            rho=params.get("rho_flanks", 0.8),
            name="flanks",
        )

        # Create components
        carcass = CarcassComponent(
            name="carcass",
            material=mat_carcass,
            resolution=resolution,
            y_top=int(params.get("y_top", 8)),
            y_bottom=int(params.get("y_bottom", 56)),
            w_belly=params.get("w_belly", 24),
            w_bottom=params.get("w_bottom", 14),
            belly_position=params.get("belly_position", 0.40),
            thickness=int(params.get("thickness_carcass", 4)),
            lip_rounding=int(params.get("lip_rounding", 3)),
        )
        tire.add_component("carcass", carcass)

        # Add crown(s)
        num_crowns = int(params.get("num_crown_layers", 1))
        for i in range(num_crowns):
            crown = CrownComponent(
                name=f"crown_{i+1}" if num_crowns > 1 else "crown",
                material=mat_crown,
                resolution=resolution,
                y_top=int(params.get("y_top", 8)),
                y_bottom=int(params.get("y_bottom", 56)),
                w_belly=params.get("w_belly", 24),
                belly_position=params.get("belly_position", 0.40),
                thickness=int(params.get("thickness_crown", 5)) - i,
                thickness_carcass=int(params.get("thickness_carcass", 4)),
            )
            tire.add_component(crown.name, crown)

        # Add flanks
        flanks = FlanksComponent(
            name="flanks",
            material=mat_flanks,
            resolution=resolution,
            y_top=int(params.get("y_top", 8)),
            y_bottom=int(params.get("y_bottom", 56)),
            w_belly=params.get("w_belly", 24),
            w_bottom=params.get("w_bottom", 14),
            belly_position=params.get("belly_position", 0.40),
            thickness_top=int(params.get("thickness_crown", 5)),
            thickness_bottom=int(params.get("thickness_flanks_bottom", 2)),
            thickness_carcass=int(params.get("thickness_carcass", 4)),
            thickness_crown=int(params.get("thickness_crown", 5)),
            lip_rounding=int(params.get("lip_rounding", 3)),
        )
        tire.add_component("flanks", flanks)

        return tire

    def _generate_deformed(
        self, tire: Tire, sample_dir: Path, idx: int
    ) -> Dict[str, Any]:
        """Generate deformed versions at different forces."""
        forces = self.config["dataset"]["outputs"]["deformation"].get(
            "forces", [100, 200, 400]
        )

        deform_data = {}

        for force in forces:
            result = self.mechanics.apply_load(tire, force)
            deform_data[f"delta_F{force}"] = result["delta"]

            # Save deformed images if requested
            if self.config["dataset"]["outputs"].get("save_deformed", True):
                deformed_tire = result["deformed_tire"]
                for comp_name, component in deformed_tire.components.items():
                    img_path = sample_dir / f"{comp_name}_deformed_F{force}.png"
                    self._save_image(component.image, img_path)

        return deform_data

    def _save_image(self, image: np.ndarray, path: Path):
        """Save image to file."""
        import matplotlib.pyplot as plt

        plt.imsave(str(path), image, cmap="gray")

    @classmethod
    def from_yaml(cls, config_path: str) -> "DatasetGenerator":
        """
        Create dataset generator from YAML config file.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            DatasetGenerator instance
        """
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return cls(config)
