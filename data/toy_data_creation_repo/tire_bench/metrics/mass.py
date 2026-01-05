"""Mass metric."""

from tire_bench.metrics.base import BaseMetric
from tire_bench.metrics.registry import MetricRegistry
from tire_bench.core.tire import Tire


@MetricRegistry.register("mass")
class MassMetric(BaseMetric):
    """
    Mass index based on component volumes and densities.

    Mass = sum(volume_i * rho_i) for all components
    """

    def __init__(
        self,
        rho_carcass: float = 1.0,
        rho_crown: float = 1.2,
        rho_flanks: float = 0.8,
    ):
        """
        Initialize mass metric.

        Args:
            rho_carcass: Default density for carcass
            rho_crown: Default density for crown
            rho_flanks: Default density for flanks
        """
        super().__init__(
            name="mass_index",
            description="Total mass based on component densities",
            unit="",
        )
        self.rho_defaults = {
            "carcass": rho_carcass,
            "crown": rho_crown,
            "flanks": rho_flanks,
        }

    def compute(self, tire: Tire, **kwargs) -> float:
        """
        Compute mass index.

        Args:
            tire: Tire instance
            **kwargs: Additional parameters (unused)

        Returns:
            Mass value
        """
        mass = 0.0

        for comp_name, component in tire.components.items():
            volume = component.image.sum()  # pixel count

            # Get rho from component material or use default
            if hasattr(component.material, "rho"):
                rho_val = component.material.rho
            else:
                # Fallback to default based on component name
                rho_val = 1.0
                for key in self.rho_defaults:
                    if key in comp_name.lower():
                        rho_val = self.rho_defaults[key]
                        break

            mass += volume * rho_val

        return float(mass)
