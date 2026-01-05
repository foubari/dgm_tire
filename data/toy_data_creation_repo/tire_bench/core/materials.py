"""Material properties for tire components."""

from dataclasses import dataclass


@dataclass
class MaterialProperties:
    """
    Material properties for a tire component.

    Attributes:
        E: Young's modulus (relative stiffness)
        rho: Density (relative)
        nu: Poisson ratio (for lateral bulge during compression)
        name: Material name for identification

    Examples:
        >>> rubber = MaterialProperties(E=1.0, rho=1.0, name="rubber_carcass")
        >>> steel = MaterialProperties(E=0.8, rho=1.2, name="steel_belt")
    """

    E: float = 1.0          # Young's modulus (relative)
    rho: float = 1.0        # Density (relative)
    nu: float = 0.49        # Poisson ratio (rubber is nearly incompressible)
    name: str = "generic"

    def __post_init__(self):
        """Validate material properties."""
        if self.E <= 0:
            raise ValueError(f"Young's modulus must be positive, got {self.E}")
        if self.rho <= 0:
            raise ValueError(f"Density must be positive, got {self.rho}")
        if not -1.0 <= self.nu <= 0.5:
            raise ValueError(f"Poisson ratio must be in [-1, 0.5], got {self.nu}")
