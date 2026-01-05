"""Performance ratio metric."""

from tire_bench.metrics.base import BaseMetric
from tire_bench.metrics.registry import MetricRegistry
from tire_bench.metrics.stiffness import VerticalStiffnessMetric
from tire_bench.metrics.mass import MassMetric
from tire_bench.core.tire import Tire


@MetricRegistry.register("performance_ratio")
class PerformanceRatioMetric(BaseMetric):
    """
    Performance ratio = K_vert / mass

    Higher values indicate better structural efficiency (stiffer with less weight).

    This metric depends on stiffness and mass metrics.
    """

    def __init__(
        self,
        E_carcass: float = 1.0,
        E_crown: float = 0.8,
        E_flanks: float = 0.5,
        rho_carcass: float = 1.0,
        rho_crown: float = 1.2,
        rho_flanks: float = 0.8,
    ):
        """
        Initialize performance ratio metric.

        Args:
            E_carcass: Default Young's modulus for carcass
            E_crown: Default Young's modulus for crown
            E_flanks: Default Young's modulus for flanks
            rho_carcass: Default density for carcass
            rho_crown: Default density for crown
            rho_flanks: Default density for flanks
        """
        super().__init__(
            name="performance_index",
            description="Stiffness to mass ratio (structural efficiency)",
            unit="",
        )
        self._stiffness_metric = VerticalStiffnessMetric(E_carcass, E_crown, E_flanks)
        self._mass_metric = MassMetric(rho_carcass, rho_crown, rho_flanks)

    def compute(self, tire: Tire, **kwargs) -> float:
        """
        Compute performance ratio.

        Args:
            tire: Tire instance
            **kwargs: Additional parameters (unused)

        Returns:
            Performance ratio (K_vert / mass)
        """
        K = self._stiffness_metric.compute(tire, **kwargs)
        m = self._mass_metric.compute(tire, **kwargs)

        return float(K / m if m > 0 else 0.0)
