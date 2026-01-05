"""Vertical stiffness metric."""

import numpy as np
from tire_bench.metrics.base import BaseMetric
from tire_bench.metrics.registry import MetricRegistry
from tire_bench.core.tire import Tire


@MetricRegistry.register("vertical_stiffness")
class VerticalStiffnessMetric(BaseMetric):
    """
    Vertical stiffness K_vert using parallel spring model.

    For each column: k_x = E_eff * A_x / L_x
    Total: K_vert = sum(k_x)

    This metric uses the material properties defined in each component.
    """

    def __init__(
        self,
        E_carcass: float = 1.0,
        E_crown: float = 0.8,
        E_flanks: float = 0.5,
    ):
        """
        Initialize stiffness metric.

        Args:
            E_carcass: Default Young's modulus for carcass
            E_crown: Default Young's modulus for crown
            E_flanks: Default Young's modulus for flanks
        """
        super().__init__(
            name="K_vert",
            description="Vertical stiffness (parallel spring model)",
            unit="",
        )
        self.E_defaults = {
            "carcass": E_carcass,
            "crown": E_crown,
            "flanks": E_flanks,
        }

    def compute(self, tire: Tire, **kwargs) -> float:
        """
        Compute vertical stiffness.

        Args:
            tire: Tire instance
            **kwargs: Additional parameters (unused)

        Returns:
            Vertical stiffness value
        """
        K_vert = 0.0
        components = tire.components
        full_mask = tire.get_full_mask()

        if full_mask.sum() == 0:
            return 0.0

        # Find active columns
        cols = np.where(full_mask.any(axis=0))[0]

        for x in cols:
            col_mask = full_mask[:, x]
            if col_mask.sum() == 0:
                continue

            # Calculate effective modulus for this column
            A_x = col_mask.sum()
            active_rows = np.where(col_mask)[0]
            L_x = active_rows.max() - active_rows.min() + 1

            if L_x == 0:
                continue

            # Weighted average of material properties
            E_eff = 0.0
            for comp_name, component in components.items():
                comp_col = component.image[:, x]
                col_area = comp_col.sum()

                if col_area == 0:
                    continue

                # Get E from component material or use default
                if hasattr(component.material, "E"):
                    E_val = component.material.E
                else:
                    # Fallback to default based on component name
                    E_val = 1.0
                    for key in self.E_defaults:
                        if key in comp_name.lower():
                            E_val = self.E_defaults[key]
                            break

                E_eff += col_area * E_val

            E_eff /= max(A_x, 1)

            k_x = E_eff * A_x / L_x
            K_vert += k_x

        return float(K_vert)
