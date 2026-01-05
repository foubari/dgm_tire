"""Deformation progression visualization - ported from notebook."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Dict, Optional

from tire_bench.core.tire import Tire
from tire_bench.mechanics.rigid_ring import RigidRingModel
from tire_bench.visualization.tire_plot import create_overlay


def show_progression(
    tire: Tire,
    forces: List[float] = None,
    mechanics: Optional[RigidRingModel] = None,
    colors: Optional[Dict[str, list]] = None,
    y_rim: int = 56,
    figsize_per_force: tuple = (4, 8)
):
    """
    Show tire deformation progression at multiple force levels.

    Ported from notebook's show_progression() function.

    Args:
        tire: Original tire instance
        forces: List of forces to apply (default: [50, 100, 200, 400, 600])
        mechanics: Mechanics model (default: RigidRingModel())
        colors: Color mapping
        y_rim: Y-coordinate of rim
        figsize_per_force: Figure size per force column
    """
    if forces is None:
        forces = [50, 100, 200, 400, 600]

    if mechanics is None:
        mechanics = RigidRingModel()

    if colors is None:
        colors = {
            "carcass": [0.9, 0.2, 0.2],
            "crown": [0.2, 0.5, 0.9],
            "flanks": [0.2, 0.8, 0.3],
        }

    # Create figure
    n_forces = len(forces)
    figsize = (figsize_per_force[0] * n_forces, figsize_per_force[1])
    fig, axes = plt.subplots(2, n_forces, figsize=figsize)

    # Get original overlay
    orig = create_overlay(tire, colors)

    for i, F in enumerate(forces):
        # Apply deformation
        result = mechanics.apply_load(tire, F)
        deformed_tire = result["deformed_tire"]
        def_img = create_overlay(deformed_tire, colors)

        # Top row: Original
        axes[0, i].imshow(orig, origin="upper")
        axes[0, i].axhline(y=y_rim, color="yellow", linestyle="--", linewidth=1)
        axes[0, i].set_title("Original")
        axes[0, i].axis("off")

        # Bottom row: Deformed
        axes[1, i].imshow(def_img, origin="upper")
        axes[1, i].axhline(y=y_rim, color="yellow", linestyle="--", linewidth=1)
        axes[1, i].set_title(f"F={F}\nδ={result['delta']:.1f}px")
        axes[1, i].axis("off")

    # Legend
    patches = [mpatches.Patch(color=colors[n], label=n) for n in colors]
    fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=12)

    # Suptitle
    fig.suptitle("Progression de la déformation (modèle anneau rigide)", fontsize=14)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.show()


def compare_geometries(
    tire_configs: List[Dict],
    force: float = 300,
    mechanics: Optional[RigidRingModel] = None,
    colors: Optional[Dict[str, list]] = None,
    figsize_per_tire: tuple = (4, 8)
):
    """
    Compare different tire geometries.

    Args:
        tire_configs: List of tire configuration dicts
        force: Force to apply for comparison
        mechanics: Mechanics model
        colors: Color mapping
        figsize_per_tire: Figure size per tire column
    """
    from tire_bench.core.tire import Tire
    from tire_bench.core.materials import MaterialProperties
    from tire_bench.geometry.generator import (
        CarcassComponent,
        CrownComponent,
        FlanksComponent,
    )

    if mechanics is None:
        mechanics = RigidRingModel()

    if colors is None:
        colors = {
            "carcass": [0.9, 0.2, 0.2],
            "crown": [0.2, 0.5, 0.9],
            "flanks": [0.2, 0.8, 0.3],
        }

    # Create figure
    n_tires = len(tire_configs)
    figsize = (figsize_per_tire[0] * n_tires, figsize_per_tire[1])
    fig, axes = plt.subplots(2, n_tires, figsize=figsize)

    results = []

    for i, config in enumerate(tire_configs):
        # Create tire from config
        tire = Tire(resolution=64)

        # Create materials
        mat_carcass = MaterialProperties(E=1.0, rho=1.0, name="carcass")
        mat_crown = MaterialProperties(E=0.8, rho=1.2, name="crown")
        mat_flanks = MaterialProperties(E=0.5, rho=0.8, name="flanks")

        # Create components
        carcass = CarcassComponent(
            name="carcass",
            material=mat_carcass,
            resolution=64,
            thickness=config.get("thickness_carcass", 4),
            w_belly=config.get("w_belly", 24),
        )
        crown = CrownComponent(
            name="crown",
            material=mat_crown,
            resolution=64,
            thickness=config.get("thickness_crown", 5),
            thickness_carcass=config.get("thickness_carcass", 4),
            w_belly=config.get("w_belly", 24),
        )
        flanks = FlanksComponent(
            name="flanks",
            material=mat_flanks,
            resolution=64,
            thickness_carcass=config.get("thickness_carcass", 4),
            thickness_crown=config.get("thickness_crown", 5),
            w_belly=config.get("w_belly", 24),
        )

        tire.add_component("carcass", carcass)
        tire.add_component("crown", crown)
        tire.add_component("flanks", flanks)

        # Compute properties
        props = mechanics.compute_properties(tire)
        result = mechanics.apply_load(tire, force)
        deformed_tire = result["deformed_tire"]

        results.append(
            {
                "name": config["name"],
                "K": props["K_vert"],
                "m": props["mass_index"],
                "perf": props["performance_index"],
                "delta": result["delta"],
            }
        )

        # Plot original (top row)
        orig_img = create_overlay(tire, colors)
        axes[0, i].imshow(orig_img, origin="upper")
        axes[0, i].set_title(f"{config['name']}\nK={props['K_vert']:.1f}")
        axes[0, i].axis("off")

        # Plot deformed (bottom row)
        def_img = create_overlay(deformed_tire, colors)
        axes[1, i].imshow(def_img, origin="upper")
        axes[1, i].set_title(
            f"δ={result['delta']:.1f}px\nPerf={props['performance_index']:.4f}"
        )
        axes[1, i].axis("off")

    # Legend
    patches = [mpatches.Patch(color=colors[n], label=n) for n in colors]
    fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=12)

    # Suptitle
    fig.suptitle(f"Comparaison (F={force}) | Haut: Original, Bas: Déformé", fontsize=14)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.show()

    # Print table
    print("\n" + "=" * 55)
    print(f"{'Nom':<10} {'K_vert':>8} {'Masse':>8} {'δ (px)':>8} {'Performance':>12}")
    print("-" * 55)
    for r in results:
        print(
            f"{r['name']:<10} {r['K']:>8.1f} {r['m']:>8.0f} {r['delta']:>8.1f} {r['perf']:>12.5f}"
        )
    best = max(results, key=lambda x: x["perf"])
    print(f"\n→ Meilleure performance : {best['name']} (meilleur ratio K/masse)")
