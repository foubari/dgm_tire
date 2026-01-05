"""Tire component generators - ported from notebook."""

import numpy as np
from tire_bench.core.component import ComponentBase, ComponentRegistry
from tire_bench.core.materials import MaterialProperties


@ComponentRegistry.register("carcass")
class CarcassComponent(ComponentBase):
    """
    Carcass component - main structural layer of the tire.

    This is the primary load-bearing structure.
    """

    def __init__(
        self,
        name: str,
        material: MaterialProperties,
        resolution: int = 64,
        y_top: int = 8,
        y_bottom: int = 56,
        w_belly: float = 24,
        w_bottom: float = 14,
        belly_position: float = 0.40,
        thickness: int = 4,
        lip_rounding: int = 3,
    ):
        """
        Initialize carcass component.

        Args:
            name: Component name
            material: Material properties
            resolution: Grid resolution
            y_top: Top y-coordinate
            y_bottom: Bottom y-coordinate
            w_belly: Width at belly (maximum width)
            w_bottom: Width at bottom
            belly_position: Relative position of belly (0-1)
            thickness: Carcass thickness
            lip_rounding: Rounding at bottom lip
        """
        super().__init__(name, material, resolution)
        self.y_top = y_top
        self.y_bottom = y_bottom
        self.w_belly = w_belly
        self.w_bottom = w_bottom
        self.belly_position = belly_position
        self.thickness = thickness
        self.lip_rounding = lip_rounding

    def generate(self) -> np.ndarray:
        """Generate carcass geometry."""
        cx = self.resolution // 2
        y_belly = int(self.y_top + self.belly_position * (self.y_bottom - self.y_top))
        arc_height = y_belly - self.y_top
        flank_height = self.y_bottom - y_belly

        carcass = np.zeros((self.resolution, self.resolution), dtype=np.float32)

        # Ellipse parameters for upper part
        ellipse_rx_outer = self.w_belly
        ellipse_ry_outer = arc_height
        ellipse_rx_inner = max(0, self.w_belly - self.thickness)
        ellipse_ry_inner = max(0, arc_height - self.thickness)

        y_lip_start = self.y_bottom - self.lip_rounding

        for y in range(self.resolution):
            if y < self.y_top or y > self.y_bottom:
                continue

            if y <= y_belly:
                # Upper part - elliptical
                dy = y - y_belly
                ro = (dy / ellipse_ry_outer) ** 2 if ellipse_ry_outer > 0 else 0
                w_outer = (
                    ellipse_rx_outer * np.sqrt(max(0, 1 - ro)) if ro <= 1 else 0
                )

                ri = (dy / ellipse_ry_inner) ** 2 if ellipse_ry_inner > 0 else 0
                w_inner = (
                    ellipse_rx_inner * np.sqrt(max(0, 1 - ri))
                    if ri <= 1 and ellipse_rx_inner > 0
                    else 0
                )
            else:
                # Lower part - smooth transition
                t = (y - y_belly) / flank_height if flank_height > 0 else 0
                ts = 0.5 * (1 - np.cos(np.pi * t))  # Cosine interpolation
                w_outer = self.w_belly + ts * (self.w_bottom - self.w_belly)
                w_inner = max(0, w_outer - self.thickness)

                # Apply lip rounding
                if y >= y_lip_start and self.lip_rounding > 0:
                    dl = y - y_lip_start
                    if dl <= self.lip_rounding:
                        lc = self.lip_rounding * (
                            1 - np.sqrt(max(0, 1 - (dl / self.lip_rounding) ** 2))
                        )
                        w_outer -= lc * 0.5
                        w_inner += lc * 0.5

            # Fill the ring
            for x in range(self.resolution):
                if w_inner <= abs(x - cx) <= w_outer:
                    carcass[y, x] = 1

        return carcass


@ComponentRegistry.register("crown")
class CrownComponent(ComponentBase):
    """
    Crown component - reinforcement layer at the top.

    Provides additional stiffness in the contact patch region.
    """

    def __init__(
        self,
        name: str,
        material: MaterialProperties,
        resolution: int = 64,
        y_top: int = 8,
        y_bottom: int = 56,
        w_belly: float = 24,
        belly_position: float = 0.40,
        thickness: int = 5,
        thickness_carcass: int = 4,
    ):
        """
        Initialize crown component.

        Args:
            name: Component name
            material: Material properties
            resolution: Grid resolution
            y_top: Top y-coordinate
            y_bottom: Bottom y-coordinate
            w_belly: Width at belly
            belly_position: Relative position of belly
            thickness: Crown thickness
            thickness_carcass: Carcass thickness (for exclusion)
        """
        super().__init__(name, material, resolution)
        self.y_top = y_top
        self.y_bottom = y_bottom
        self.w_belly = w_belly
        self.belly_position = belly_position
        self.thickness = thickness
        self.thickness_carcass = thickness_carcass

    def generate(self) -> np.ndarray:
        """Generate crown geometry."""
        # Need to generate carcass for exclusion
        carcass_comp = CarcassComponent(
            name="temp_carcass",
            material=MaterialProperties(),
            resolution=self.resolution,
            y_top=self.y_top,
            y_bottom=self.y_bottom,
            w_belly=self.w_belly,
            belly_position=self.belly_position,
            thickness=self.thickness_carcass,
        )
        carcass = carcass_comp.generate()

        cx = self.resolution // 2
        y_belly = int(self.y_top + self.belly_position * (self.y_bottom - self.y_top))
        arc_height = y_belly - self.y_top

        crown = np.zeros((self.resolution, self.resolution), dtype=np.float32)

        # Crown ellipse parameters
        ellipse_rx_outer = self.w_belly
        ellipse_ry_outer = arc_height
        crown_rx = ellipse_rx_outer + self.thickness
        crown_ry = ellipse_ry_outer + self.thickness

        for y in range(self.resolution):
            if y > y_belly:
                continue

            dy = y - y_belly
            rcr = (dy / crown_ry) ** 2 if crown_ry > 0 else 0
            if rcr > 1:
                continue

            w_cr = crown_rx * np.sqrt(1 - rcr)

            ro = (dy / ellipse_ry_outer) ** 2 if ellipse_ry_outer > 0 else 0
            w_carc = ellipse_rx_outer * np.sqrt(max(0, 1 - ro)) if ro <= 1 else 0

            for x in range(self.resolution):
                d = abs(x - cx)
                if w_carc < d <= w_cr:
                    crown[y, x] = 1
                elif (
                    d <= w_cr
                    and carcass[y, x] == 0
                    and y < self.y_top + self.thickness
                ):
                    crown[y, x] = 1

        # Exclude carcass
        crown = crown * (1 - carcass)

        return crown


@ComponentRegistry.register("flanks")
class FlanksComponent(ComponentBase):
    """
    Flanks component - side reinforcement.

    Provides lateral support and flexibility in the sidewall region.
    """

    def __init__(
        self,
        name: str,
        material: MaterialProperties,
        resolution: int = 64,
        y_top: int = 8,
        y_bottom: int = 56,
        w_belly: float = 24,
        w_bottom: float = 14,
        belly_position: float = 0.40,
        thickness_top: int = 5,
        thickness_bottom: int = 2,
        thickness_carcass: int = 4,
        thickness_crown: int = 5,
        lip_rounding: int = 3,
    ):
        """
        Initialize flanks component.

        Args:
            name: Component name
            material: Material properties
            resolution: Grid resolution
            y_top: Top y-coordinate
            y_bottom: Bottom y-coordinate
            w_belly: Width at belly
            w_bottom: Width at bottom
            belly_position: Relative position of belly
            thickness_top: Flank thickness at top
            thickness_bottom: Flank thickness at bottom
            thickness_carcass: Carcass thickness (for exclusion)
            thickness_crown: Crown thickness (for exclusion)
            lip_rounding: Rounding at bottom lip
        """
        super().__init__(name, material, resolution)
        self.y_top = y_top
        self.y_bottom = y_bottom
        self.w_belly = w_belly
        self.w_bottom = w_bottom
        self.belly_position = belly_position
        self.thickness_top = thickness_top
        self.thickness_bottom = thickness_bottom
        self.thickness_carcass = thickness_carcass
        self.thickness_crown = thickness_crown
        self.lip_rounding = lip_rounding

    def generate(self) -> np.ndarray:
        """Generate flanks geometry."""
        # Generate carcass and crown for exclusion
        carcass_comp = CarcassComponent(
            name="temp_carcass",
            material=MaterialProperties(),
            resolution=self.resolution,
            y_top=self.y_top,
            y_bottom=self.y_bottom,
            w_belly=self.w_belly,
            w_bottom=self.w_bottom,
            belly_position=self.belly_position,
            thickness=self.thickness_carcass,
            lip_rounding=self.lip_rounding,
        )
        carcass = carcass_comp.generate()

        crown_comp = CrownComponent(
            name="temp_crown",
            material=MaterialProperties(),
            resolution=self.resolution,
            y_top=self.y_top,
            y_bottom=self.y_bottom,
            w_belly=self.w_belly,
            belly_position=self.belly_position,
            thickness=self.thickness_crown,
            thickness_carcass=self.thickness_carcass,
        )
        crown = crown_comp.generate()

        cx = self.resolution // 2
        y_belly = int(self.y_top + self.belly_position * (self.y_bottom - self.y_top))
        flank_height = self.y_bottom - y_belly

        flanks = np.zeros((self.resolution, self.resolution), dtype=np.float32)
        y_lip_start = self.y_bottom - self.lip_rounding

        for y in range(self.resolution):
            if y <= y_belly or y > self.y_bottom:
                continue

            t = (y - y_belly) / flank_height if flank_height > 0 else 0
            ts = 0.5 * (1 - np.cos(np.pi * t))
            w_carc = self.w_belly + ts * (self.w_bottom - self.w_belly)

            # Apply lip rounding
            if y >= y_lip_start and self.lip_rounding > 0:
                dl = y - y_lip_start
                if dl <= self.lip_rounding:
                    lc = self.lip_rounding * (
                        1 - np.sqrt(max(0, 1 - (dl / self.lip_rounding) ** 2))
                    )
                    w_carc -= lc * 0.5

            w_fl = w_carc + self.thickness_top + t * (
                self.thickness_bottom - self.thickness_top
            )

            for x in range(self.resolution):
                if w_carc < abs(x - cx) <= w_fl:
                    flanks[y, x] = 1

        # Exclude carcass and crown
        flanks = flanks * (1 - carcass) * (1 - crown)

        return flanks
