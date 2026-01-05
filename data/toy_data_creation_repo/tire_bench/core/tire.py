"""Tire composition class supporting variable number of components."""

from typing import Dict, Optional, List, Any
import numpy as np

from tire_bench.core.component import ComponentBase, ComponentRegistry


class Tire:
    """
    Tire composed of variable number of components.

    This class implements the Composition pattern to allow flexible
    tire architectures with any number and types of components.

    Examples:
        >>> tire = Tire(resolution=64)
        >>> tire.add_component('carcass', CarcassComponent(...))
        >>> tire.add_component('crown_1', CrownComponent(...))
        >>> tire.add_component('crown_2', CrownComponent(...))  # Multi-layer
        >>> tire.add_component('flanks', FlanksComponent(...))

        >>> # Create from config
        >>> config = {'resolution': 64, 'components': [...]}
        >>> tire = Tire.from_config(config)
    """

    def __init__(self, resolution: int = 64):
        """
        Initialize tire.

        Args:
            resolution: Grid resolution (default: 64x64)
        """
        self.resolution = resolution
        self._components: Dict[str, ComponentBase] = {}
        self._full_mask: Optional[np.ndarray] = None
        self._dirty = True  # Track if recomputation needed

    def add_component(self, name: str, component: ComponentBase):
        """
        Add a component to the tire.

        Args:
            name: Component identifier
            component: Component instance

        Raises:
            ValueError: If component name already exists
        """
        if name in self._components:
            raise ValueError(f"Component '{name}' already exists")
        self._components[name] = component
        self._dirty = True

    def remove_component(self, name: str):
        """
        Remove a component.

        Args:
            name: Component identifier

        Raises:
            KeyError: If component doesn't exist
        """
        del self._components[name]
        self._dirty = True

    def get_component(self, name: str) -> ComponentBase:
        """
        Get component by name.

        Args:
            name: Component identifier

        Returns:
            Component instance

        Raises:
            KeyError: If component doesn't exist
        """
        return self._components[name]

    @property
    def components(self) -> Dict[str, ComponentBase]:
        """
        Get all components.

        Returns:
            Dictionary of component name to component instance
        """
        return self._components

    @property
    def component_names(self) -> List[str]:
        """
        Get list of component names.

        Returns:
            List of component names
        """
        return list(self._components.keys())

    def get_full_mask(self, priority_order: Optional[List[str]] = None) -> np.ndarray:
        """
        Get combined mask of all components.

        Args:
            priority_order: Optional list of component names in priority order.
                          If None, uses insertion order. Later components can
                          overlap earlier ones.

        Returns:
            Binary mask of combined tire geometry
        """
        if not self._dirty and self._full_mask is not None and priority_order is None:
            return self._full_mask

        full_mask = np.zeros((self.resolution, self.resolution))

        # Determine component order
        if priority_order is not None:
            # Validate all names exist
            for name in priority_order:
                if name not in self._components:
                    raise ValueError(f"Component '{name}' not found")
            component_order = priority_order
        else:
            component_order = list(self._components.keys())

        # Apply components in order (later ones can overlap)
        for name in component_order:
            component = self._components[name]
            full_mask = np.maximum(full_mask, component.image)

        if priority_order is None:
            self._full_mask = full_mask
            self._dirty = False

        return full_mask

    def to_dict(self) -> Dict[str, np.ndarray]:
        """
        Export all component images as dict.

        Returns:
            Dictionary mapping component names to their binary masks
        """
        return {name: comp.image for name, comp in self._components.items()}

    def get_properties(self) -> Dict[str, Any]:
        """
        Get tire properties.

        Returns:
            Dictionary with tire metadata including all components
        """
        return {
            "resolution": self.resolution,
            "num_components": len(self._components),
            "components": {
                name: comp.get_properties() for name, comp in self._components.items()
            },
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Tire":
        """
        Create tire from configuration dict.

        Args:
            config: Configuration dictionary with structure:
                {
                    'resolution': int,
                    'components': [
                        {
                            'name': str,
                            'type': str,
                            'params': dict
                        },
                        ...
                    ]
                }

        Returns:
            Tire instance

        Examples:
            >>> config = {
            ...     'resolution': 64,
            ...     'components': [
            ...         {'name': 'carcass', 'type': 'carcass', 'params': {...}},
            ...         {'name': 'crown', 'type': 'crown', 'params': {...}}
            ...     ]
            ... }
            >>> tire = Tire.from_config(config)
        """
        tire = cls(resolution=config.get("resolution", 64))

        for comp_config in config.get("components", []):
            # Extract component parameters
            comp_name = comp_config["name"]
            comp_type = comp_config["type"]
            comp_params = comp_config.get("params", {})

            # Add resolution if not specified
            if "resolution" not in comp_params:
                comp_params["resolution"] = tire.resolution

            # Create component using registry
            component = ComponentRegistry.create(component_type=comp_type, **comp_params)

            tire.add_component(comp_name, component)

        return tire

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"Tire(resolution={self.resolution}, "
            f"components={list(self._components.keys())})"
        )
