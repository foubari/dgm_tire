"""Rigid Ring deformation model - ported from notebook TireMechanics class."""

from typing import Dict, Any
import numpy as np
from scipy.ndimage import map_coordinates

from tire_bench.core.tire import Tire


class RigidRingModel:
    """
    Rigid Ring mechanical deformation model.

    This model treats the tire as a rigid ring that deforms uniformly:
    - The base (rim) is fixed
    - The crown descends uniformly by δ = F / K_vert
    - Lateral bulge occurs due to Poisson effect

    Physical justification:
    - Vertical stiffness K_vert calculated using parallel spring model
    - Quasi-static equilibrium: F ≈ K_vert * δ
    - Lateral coupling via Poisson effect (nearly incompressible rubber)

    Attributes:
        E_carcass: Young's modulus for carcass (relative stiffness)
        E_crown: Young's modulus for crown
        E_flanks: Young's modulus for flanks
        rho_carcass: Density for carcass (relative)
        rho_crown: Density for crown
        rho_flanks: Density for flanks
        max_strain: Maximum allowable compression (fraction of height)
        nu: Poisson ratio (for lateral bulge)
    """

    def __init__(
        self,
        E_carcass: float = 1.0,
        E_crown: float = 0.8,
        E_flanks: float = 0.5,
        rho_carcass: float = 1.0,
        rho_crown: float = 1.2,
        rho_flanks: float = 0.8,
        max_strain: float = 0.3,
        nu: float = 0.49,
    ):
        """
        Initialize rigid ring model.

        Args:
            E_carcass: Young's modulus for carcass
            E_crown: Young's modulus for crown
            E_flanks: Young's modulus for flanks
            rho_carcass: Density for carcass
            rho_crown: Density for crown
            rho_flanks: Density for flanks
            max_strain: Maximum compression (0.3 = 30% of height)
            nu: Poisson ratio (0.49 for nearly incompressible rubber)
        """
        self.E_carcass = E_carcass
        self.E_crown = E_crown
        self.E_flanks = E_flanks
        self.rho_carcass = rho_carcass
        self.rho_crown = rho_crown
        self.rho_flanks = rho_flanks
        self.max_strain = max_strain
        self.nu = nu

    def compute_properties(self, tire: Tire) -> Dict[str, Any]:
        """
        Compute mechanical properties using parallel spring model.

        For each vertical column:
        - k_x = E_eff * A_x / L_x (spring constant)
        - K_vert = sum(k_x) (total vertical stiffness)

        Args:
            tire: Tire instance

        Returns:
            Dictionary with:
                - K_vert: Vertical stiffness
                - mass_index: Total mass
                - performance_index: K_vert / mass ratio
                - height_px: Tire height in pixels
        """
        components = tire.components
        full_mask = tire.get_full_mask()
        res = tire.resolution

        # Default material properties mapping
        material_E = {
            "carcass": self.E_carcass,
            "crown": self.E_crown,
            "flanks": self.E_flanks,
        }
        material_rho = {
            "carcass": self.rho_carcass,
            "crown": self.rho_crown,
            "flanks": self.rho_flanks,
        }

        if full_mask.sum() == 0:
            return {
                "K_vert": 0.0,
                "mass_index": 0.0,
                "performance_index": 0.0,
                "height_px": 0.0,
            }

        rows = np.where(full_mask.any(axis=1))[0]
        cols = np.where(full_mask.any(axis=0))[0]
        height_px = rows.max() - rows.min() + 1

        # Compute vertical stiffness column by column
        K_vert = 0.0

        for x in range(cols.min(), cols.max() + 1):
            col_mask = full_mask[:, x]
            if col_mask.sum() == 0:
                continue

            A_x = col_mask.sum()  # Cross-sectional area
            active_rows = np.where(col_mask)[0]
            L_x = active_rows.max() - active_rows.min() + 1  # Column length

            # Calculate effective Young's modulus for this column
            E_eff = 0.0
            for comp_name, component in components.items():
                comp_col = component.image[:, x]
                # Get E value from material or use default based on name
                if hasattr(component.material, "E"):
                    E_val = component.material.E
                else:
                    # Fallback to default based on component name
                    for key in material_E:
                        if key in comp_name.lower():
                            E_val = material_E[key]
                            break
                    else:
                        E_val = 1.0

                E_eff += comp_col.sum() * E_val

            E_eff /= max(A_x, 1)

            # Column stiffness k_x = E_eff * A_x / L_x
            k_x = E_eff * A_x / L_x
            K_vert += k_x

        # Compute mass
        mass = 0.0
        for comp_name, component in components.items():
            volume = component.image.sum()

            # Get rho value from material or use default
            if hasattr(component.material, "rho"):
                rho_val = component.material.rho
            else:
                # Fallback to default based on component name
                for key in material_rho:
                    if key in comp_name.lower():
                        rho_val = material_rho[key]
                        break
                else:
                    rho_val = 1.0

            mass += volume * rho_val

        # Performance index = K / mass
        performance = K_vert / mass if mass > 0 else 0.0

        return {
            "K_vert": float(K_vert),
            "mass_index": float(mass),
            "performance_index": float(performance),
            "height_px": float(height_px),
        }

    def apply_load(self, tire: Tire, force: float) -> Dict[str, Any]:
        """
        Apply vertical load to tire using rigid ring model.

        Deformation field:
        - Vertical: disp_y = δ * t, where t = (y_bottom - y) / h
          (t=0 at rim, t=1 at crown)
        - Lateral: disp_x = ν * (δ/h) * bulge_weight * (x - cx)
          (Poisson effect, maximum at sidewalls)

        Args:
            tire: Tire instance
            force: Applied vertical force

        Returns:
            Dictionary with:
                - deformed_tire: Tire instance with deformed components
                - displacement: Vertical displacement field
                - displacement_x: Lateral displacement field
                - delta: Actual deflection in pixels
                - K_vert: Vertical stiffness
                - mass_index: Mass
                - performance_index: Performance ratio
        """
        props = self.compute_properties(tire)
        K_vert = props["K_vert"]
        res = tire.resolution

        if K_vert == 0:
            # No deformation
            return {
                "deformed_tire": tire,
                "displacement": np.zeros((res, res)),
                "displacement_x": np.zeros((res, res)),
                "delta": 0.0,
                **props,
            }

        full_mask = tire.get_full_mask()
        rows = np.where(full_mask.any(axis=1))[0]
        y_top, y_bottom = rows.min(), rows.max()
        h = y_bottom - y_top + 1

        # Calculate deflection: δ = F / K
        delta = min(force / K_vert, h * self.max_strain)

        # Create displacement fields
        disp_y = np.zeros((res, res), dtype=np.float32)
        bulge_w = np.zeros((res, res), dtype=np.float32)

        for y in range(y_top, y_bottom + 1):
            t = (y_bottom - y) / h  # t=0 at rim, t=1 at crown
            # Vertical displacement (rigid ring: linear profile)
            disp_y[y, :] = delta * t

            # Bulge weight: maximum at mid-height (sidewalls)
            # w(t) = 4*t*(1-t) peaks at t=0.5
            bulge_w[y, :] = 4.0 * t * (1.0 - t)

        # Create coordinate grids
        yc, xc = np.meshgrid(np.arange(res), np.arange(res), indexing="ij")
        x0 = (res - 1) / 2.0

        # Lateral displacement (Poisson effect)
        # ε_x ≈ ν * ε_y, where ε_y ≈ -δ/h
        eps_x = self.nu * (delta / max(h, 1))
        disp_x = eps_x * bulge_w * (xc - x0)

        # New coordinates after deformation
        new_y = yc - disp_y  # Crown moves down
        new_x = xc - disp_x  # Lateral bulge

        def warp(img):
            """Warp image using displacement field."""
            return (
                map_coordinates(img, [new_y, new_x], order=1, mode="constant", cval=0)
                > 0.5
            ).astype(np.float32)

        # Create deformed tire
        deformed_tire = Tire(resolution=res)

        for comp_name, component in tire.components.items():
            # Warp component
            warped_img = warp(component.image)

            # Create new component with same parameters but deformed image
            from tire_bench.core.component import ComponentBase

            class DeformedComponent(ComponentBase):
                def __init__(self, name, material, resolution, image):
                    super().__init__(name, material, resolution)
                    self._image = image

                def generate(self):
                    return self._image

            deformed_comp = DeformedComponent(
                name=comp_name,
                material=component.material,
                resolution=res,
                image=warped_img,
            )

            deformed_tire.add_component(comp_name, deformed_comp)

        # Re-impose partition (no overlap) after warping
        comp_names = list(deformed_tire.components.keys())
        if len(comp_names) >= 2:
            # Ensure components don't overlap (maintain partition)
            for i in range(1, len(comp_names)):
                current_comp = deformed_tire.components[comp_names[i]]
                mask = current_comp.image.copy()

                for j in range(i):
                    prev_comp = deformed_tire.components[comp_names[j]]
                    mask = mask * (1.0 - prev_comp.image)

                current_comp._image = mask

        return {
            "deformed_tire": deformed_tire,
            "displacement": disp_y,
            "displacement_x": disp_x,
            "delta": float(delta),
            **props,
        }
