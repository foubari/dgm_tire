"""
Custom metrics example: Add a new performance metric.

This example demonstrates:
- Creating a custom metric by inheriting from BaseMetric
- Registering the metric using the MetricRegistry
- Using the custom metric alongside existing metrics
"""

from tire_bench.core.tire import Tire
from tire_bench.core.materials import MaterialProperties
from tire_bench.geometry.generator import (
    CarcassComponent,
    CrownComponent,
    FlanksComponent,
)
from tire_bench.metrics.base import BaseMetric
from tire_bench.metrics.registry import MetricRegistry


# Define custom metric
@MetricRegistry.register("damping_coefficient")
class DampingMetric(BaseMetric):
    """
    Custom metric: Estimate damping based on material composition.

    This is a simplified metric that assumes materials with lower
    Young's modulus (E < 0.7) contribute more to damping.
    """

    def __init__(self):
        super().__init__(
            name="damping_coefficient",
            description="Estimated damping based on rubber content",
            unit="",
        )

    def compute(self, tire: Tire, **kwargs) -> float:
        """
        Compute damping coefficient.

        Higher damping when more low-E (rubber-like) material is present.
        """
        rubber_volume = 0
        total_volume = 0

        for comp in tire.components.values():
            vol = comp.image.sum()
            total_volume += vol

            # Assume materials with E < 0.7 are rubber-like
            if comp.material.E < 0.7:
                rubber_volume += vol

        if total_volume == 0:
            return 0.0

        damping = rubber_volume / total_volume
        return float(damping)


# Another custom metric
@MetricRegistry.register("weight_distribution")
class WeightDistributionMetric(BaseMetric):
    """
    Custom metric: Evaluate weight distribution symmetry.

    Measures how evenly distributed the mass is.
    """

    def __init__(self):
        super().__init__(
            name="weight_distribution",
            description="Mass distribution symmetry index",
            unit="",
        )

    def compute(self, tire: Tire, **kwargs) -> float:
        """
        Compute weight distribution symmetry.

        Returns value close to 1.0 for symmetric distribution.
        """
        import numpy as np

        full_mask = tire.get_full_mask()
        if full_mask.sum() == 0:
            return 0.0

        # Compute center of mass
        resolution = tire.resolution
        y_coords, x_coords = np.ogrid[:resolution, :resolution]

        # Weight by density
        weighted_mask = np.zeros_like(full_mask)
        for comp in tire.components.values():
            weighted_mask += comp.image * comp.material.rho

        total_mass = weighted_mask.sum()
        if total_mass == 0:
            return 0.0

        cx_actual = (weighted_mask * x_coords).sum() / total_mass
        cx_expected = (resolution - 1) / 2.0

        # Symmetry score (deviation from center)
        deviation = abs(cx_actual - cx_expected) / cx_expected
        symmetry = np.exp(-5 * deviation)

        return float(symmetry)


def main():
    print("=" * 60)
    print("CUSTOM METRICS EXAMPLE")
    print("=" * 60)

    # Create tire
    print("\n1. Creating tire...")
    tire = Tire(resolution=64)

    mat_carcass = MaterialProperties(E=1.0, rho=1.0, name="carcass")
    mat_crown = MaterialProperties(E=0.8, rho=1.2, name="crown")
    mat_flanks = MaterialProperties(E=0.5, rho=0.8, name="flanks")  # Low E = rubber

    tire.add_component(
        "carcass",
        CarcassComponent(
            name="carcass", material=mat_carcass, resolution=64, thickness=4, w_belly=24
        ),
    )
    tire.add_component(
        "crown",
        CrownComponent(
            name="crown",
            material=mat_crown,
            resolution=64,
            thickness=5,
            thickness_carcass=4,
            w_belly=24,
        ),
    )
    tire.add_component(
        "flanks",
        FlanksComponent(
            name="flanks",
            material=mat_flanks,
            resolution=64,
            thickness_carcass=4,
            thickness_crown=5,
            w_belly=24,
        ),
    )

    print(f"✓ Tire created")

    # 2. Compute all metrics (including custom ones)
    print("\n2. Computing metrics...")

    # Standard metrics
    stiffness = MetricRegistry.create("vertical_stiffness")
    mass = MetricRegistry.create("mass")
    performance = MetricRegistry.create("performance_ratio")

    # Custom metrics
    damping = MetricRegistry.create("damping_coefficient")
    weight_dist = MetricRegistry.create("weight_distribution")

    print("\n   Standard metrics:")
    print(f"   - K_vert              : {stiffness.compute(tire):.2f}")
    print(f"   - Mass index          : {mass.compute(tire):.0f}")
    print(f"   - Performance ratio   : {performance.compute(tire):.4f}")

    print("\n   Custom metrics:")
    print(f"   - Damping coefficient : {damping.compute(tire):.4f}")
    print(f"   - Weight distribution : {weight_dist.compute(tire):.4f}")

    # 3. List all available metrics
    print("\n3. All registered metrics:")
    all_metrics = MetricRegistry.list_metrics()
    for metric_name in all_metrics:
        print(f"   - {metric_name}")

    print("\n" + "=" * 60)
    print("✓ Custom metrics example completed!")
    print("\nKey takeaway:")
    print("  Adding new metrics requires NO modification to existing code.")
    print("  Just inherit from BaseMetric and use @MetricRegistry.register()")
    print("=" * 60)


if __name__ == "__main__":
    main()
