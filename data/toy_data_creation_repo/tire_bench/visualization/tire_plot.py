"""Tire visualization functions - ported from notebook."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, Optional

from tire_bench.core.tire import Tire
from tire_bench.mechanics.rigid_ring import RigidRingModel


def create_overlay(
    tire: Tire,
    colors: Optional[Dict[str, list]] = None
) -> np.ndarray:
    """
    Create RGB overlay of tire components.

    Args:
        tire: Tire instance
        colors: Optional color mapping {component_name: [r, g, b]}

    Returns:
        RGB image array
    """
    if colors is None:
        colors = {
            "carcass": [0.9, 0.2, 0.2],  # Red
            "crown": [0.2, 0.5, 0.9],     # Blue
            "flanks": [0.2, 0.8, 0.3],    # Green
        }

    res = tire.resolution
    overlay = np.zeros((res, res, 3))

    for comp_name, component in tire.components.items():
        img = component.image
        # Find matching color
        color = [0.5, 0.5, 0.5]  # Default gray
        for key, col in colors.items():
            if key in comp_name.lower():
                color = col
                break

        for ch in range(3):
            overlay[:, :, ch] += (img > 0.5) * color[ch]

    return np.clip(overlay, 0, 1)


def visualize_tire(
    tire: Tire,
    title: str = "Tire Cross-Section",
    colors: Optional[Dict[str, list]] = None,
    show_legend: bool = True,
    figsize: tuple = (8, 8)
):
    """
    Visualize tire cross-section.

    Args:
        tire: Tire instance
        title: Plot title
        colors: Color mapping
        show_legend: Whether to show legend
        figsize: Figure size
    """
    if colors is None:
        colors = {
            "carcass": [0.9, 0.2, 0.2],
            "crown": [0.2, 0.5, 0.9],
            "flanks": [0.2, 0.8, 0.3],
        }

    overlay = create_overlay(tire, colors)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(overlay, origin="upper")
    ax.set_title(title, fontsize=14)
    ax.axis("off")

    if show_legend:
        patches = []
        for name in tire.component_names:
            color = [0.5, 0.5, 0.5]
            for key, col in colors.items():
                if key in name.lower():
                    color = col
                    break
            patches.append(mpatches.Patch(color=color, label=name))

        ax.legend(handles=patches, loc="upper right", fontsize=10)

    plt.tight_layout()
    plt.show()


def visualize_deformation(
    tire: Tire,
    force: float = 200,
    mechanics: Optional[RigidRingModel] = None,
    colors: Optional[Dict[str, list]] = None,
    y_rim: int = 56,
    figsize: tuple = (16, 5)
):
    """
    Visualize tire deformation (before/after comparison).

    Ported from notebook's visualize() function.

    Args:
        tire: Original tire instance
        force: Applied force
        mechanics: Mechanics model (default: RigidRingModel())
        colors: Color mapping
        y_rim: Y-coordinate of rim (for yellow line)
        figsize: Figure size
    """
    if mechanics is None:
        mechanics = RigidRingModel()

    if colors is None:
        colors = {
            "carcass": [0.9, 0.2, 0.2],
            "crown": [0.2, 0.5, 0.9],
            "flanks": [0.2, 0.8, 0.3],
        }

    # Compute properties and deformation
    props = mechanics.compute_properties(tire)
    result = mechanics.apply_load(tire, force)
    deformed_tire = result["deformed_tire"]

    # Create overlays
    before = create_overlay(tire, colors)
    after = create_overlay(deformed_tire, colors)

    # Create figure
    fig, axes = plt.subplots(1, 4, figsize=figsize)

    # Add rim line to first 3 plots
    for ax in axes[:3]:
        ax.axhline(y=y_rim, color="yellow", linestyle="--", linewidth=2)

    # Plot 1: AVANT (before)
    axes[0].imshow(before, origin="upper")
    axes[0].set_title("AVANT", fontsize=14)
    axes[0].axis("off")

    # Plot 2: APRÈS (after)
    axes[1].imshow(after, origin="upper")
    axes[1].set_title(f"APRÈS (F={force})", fontsize=14)
    axes[1].axis("off")

    # Plot 3: Superposition
    axes[2].imshow(0.4 * before + 0.6 * after, origin="upper")
    axes[2].set_title("Superposition", fontsize=14)
    axes[2].axis("off")

    # Plot 4: Displacement field
    im = axes[3].imshow(result["displacement"], origin="upper", cmap="Reds", vmin=0)
    plt.colorbar(im, ax=axes[3], label="δ (px)")
    axes[3].set_title("Déplacement vertical\n+ bulge latéral", fontsize=12)
    axes[3].axis("off")

    # Legend
    patches = [mpatches.Patch(color=colors[n], label=n) for n in colors]
    patches.append(mpatches.Patch(color="yellow", label="Base fixée (jante)"))
    fig.legend(handles=patches, loc="lower center", ncol=4, fontsize=11)

    # Suptitle
    fig.suptitle(
        f"Modèle ANNEAU RIGIDE  |  δ = {result['delta']:.1f} px  |  "
        f"K_vert = {props['K_vert']:.1f}  |  Perf = {props['performance_index']:.4f}",
        fontsize=12,
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.show()

    print(f"\n✓ Écrasement: δ={result['delta']:.1f}px, avec bulge latéral (ν≈0.49).")
