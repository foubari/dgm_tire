"""
Basic usage example: Create and visualize a tire.

This example demonstrates:
- Creating a tire with basic components
- Computing mechanical properties
- Applying deformation
- Visualizing results
"""

from tire_bench.core.tire import Tire
from tire_bench.core.materials import MaterialProperties
from tire_bench.geometry.generator import (
    CarcassComponent,
    CrownComponent,
    FlanksComponent,
)
from tire_bench.mechanics.rigid_ring import RigidRingModel
from tire_bench.metrics.registry import MetricRegistry
from tire_bench.visualization.tire_plot import visualize_tire, visualize_deformation


def main():
    print("=" * 60)
    print("TIRE DEFORMATION BENCHMARK - Basic Usage Example")
    print("=" * 60)

    # 1. Create tire
    print("\n1. Creating tire...")
    tire = Tire(resolution=64)

    # Define materials
    mat_carcass = MaterialProperties(E=1.0, rho=1.0, name="rubber_carcass")
    mat_crown = MaterialProperties(E=0.8, rho=1.2, name="steel_belt")
    mat_flanks = MaterialProperties(E=0.5, rho=0.8, name="rubber_sidewall")

    # Create components
    carcass = CarcassComponent(
        name="carcass",
        material=mat_carcass,
        resolution=64,
        y_top=8,
        y_bottom=56,
        w_belly=24,
        w_bottom=14,
        belly_position=0.40,
        thickness=4,
        lip_rounding=3,
    )

    crown = CrownComponent(
        name="crown",
        material=mat_crown,
        resolution=64,
        y_top=8,
        y_bottom=56,
        w_belly=24,
        belly_position=0.40,
        thickness=5,
        thickness_carcass=4,
    )

    flanks = FlanksComponent(
        name="flanks",
        material=mat_flanks,
        resolution=64,
        y_top=8,
        y_bottom=56,
        w_belly=24,
        w_bottom=14,
        belly_position=0.40,
        thickness_top=5,
        thickness_bottom=2,
        thickness_carcass=4,
        thickness_crown=5,
        lip_rounding=3,
    )

    # Add components to tire
    tire.add_component("carcass", carcass)
    tire.add_component("crown", crown)
    tire.add_component("flanks", flanks)

    print(f"✓ Tire created with {len(tire.components)} components")

    # 2. Compute metrics
    print("\n2. Computing mechanical properties...")
    mechanics = RigidRingModel()
    props = mechanics.compute_properties(tire)

    print(f"   K_vert (stiffness)    : {props['K_vert']:.2f}")
    print(f"   Mass index            : {props['mass_index']:.0f}")
    print(f"   Performance (K/mass)  : {props['performance_index']:.4f}")
    print(f"   → For δ=10px, F ≈ {props['K_vert'] * 10:.0f}")

    # 3. Apply deformation
    print("\n3. Applying deformation (F=200)...")
    result = mechanics.apply_load(tire, force=200)
    print(f"   Deflection δ = {result['delta']:.1f} px")

    # 4. Visualize
    print("\n4. Visualizing results...")
    print("   (Close the plot windows to continue)")

    # Original tire
    visualize_tire(tire, title="Original Tire Cross-Section")

    # Deformation comparison
    visualize_deformation(tire, force=200)

    print("\n" + "=" * 60)
    print("✓ Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
