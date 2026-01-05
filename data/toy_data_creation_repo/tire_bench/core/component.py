"""Base component class and registry for tire components."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np

from tire_bench.core.materials import MaterialProperties


class ComponentBase(ABC):
    """
    Abstract base class for tire components.

    Design: Open-Closed Principle - extend by creating new component types,
    not by modifying existing code.

    Attributes:
        name: Component name (e.g., "carcass", "crown_layer_1")
        material: Material properties for this component

    Examples:
        >>> @ComponentRegistry.register('custom')
        ... class CustomComponent(ComponentBase):
        ...     def generate(self):
        ...         return np.zeros((64, 64))
    """

    def __init__(self, name: str, material: MaterialProperties, resolution: int = 64):
        """
        Initialize component.

        Args:
            name: Component name
            material: Material properties
            resolution: Grid resolution (default: 64x64)
        """
        self.name = name
        self.material = material
        self.resolution = resolution
        self._image: Optional[np.ndarray] = None

    @property
    def image(self) -> np.ndarray:
        """
        Get component image (lazy evaluation).

        Returns:
            Binary mask of shape (resolution, resolution)
        """
        if self._image is None:
            self._image = self.generate()
        return self._image

    @abstractmethod
    def generate(self) -> np.ndarray:
        """
        Generate component geometry.

        Returns:
            Binary mask (0 or 1) of shape (resolution, resolution)
        """
        pass

    def get_properties(self) -> Dict[str, Any]:
        """
        Get component properties.

        Returns:
            Dictionary with component metadata
        """
        return {
            "name": self.name,
            "material": self.material.name,
            "E": self.material.E,
            "rho": self.material.rho,
            "nu": self.material.nu,
        }

    def invalidate_cache(self):
        """Invalidate cached image (force regeneration on next access)."""
        self._image = None


class ComponentRegistry:
    """
    Registry for component types - enables config-driven instantiation.

    This registry allows dynamic creation of components from configuration files.

    Examples:
        >>> @ComponentRegistry.register('my_component')
        ... class MyComponent(ComponentBase):
        ...     pass
        >>> comp = ComponentRegistry.create('my_component', name='test', material=mat)
    """

    _registry: Dict[str, type] = {}

    @classmethod
    def register(cls, component_type: str):
        """
        Decorator to register component types.

        Args:
            component_type: Type identifier for the component

        Returns:
            Decorator function

        Examples:
            >>> @ComponentRegistry.register('carcass')
            ... class CarcassComponent(ComponentBase):
            ...     pass
        """

        def decorator(component_class):
            if not issubclass(component_class, ComponentBase):
                raise TypeError(
                    f"{component_class.__name__} must inherit from ComponentBase"
                )
            cls._registry[component_type] = component_class
            return component_class

        return decorator

    @classmethod
    def create(cls, component_type: str, **kwargs) -> ComponentBase:
        """
        Factory method to create components from config.

        Args:
            component_type: Type of component to create
            **kwargs: Arguments to pass to component constructor

        Returns:
            Component instance

        Raises:
            ValueError: If component type is not registered
        """
        if component_type not in cls._registry:
            raise ValueError(
                f"Unknown component type: {component_type}. "
                f"Available types: {list(cls._registry.keys())}"
            )
        return cls._registry[component_type](**kwargs)

    @classmethod
    def list_types(cls) -> list:
        """
        List all registered component types.

        Returns:
            List of registered component type names
        """
        return list(cls._registry.keys())
