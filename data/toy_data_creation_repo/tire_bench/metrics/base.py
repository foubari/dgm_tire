"""Base class for all performance metrics."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from tire_bench.core.tire import Tire


class BaseMetric(ABC):
    """
    Abstract base class for all performance metrics.

    Design: Open-Closed Principle
    - Add new metrics by inheriting from BaseMetric
    - No need to modify existing code
    - Each metric can have its own visualization

    Examples:
        >>> @MetricRegistry.register('custom_metric')
        ... class CustomMetric(BaseMetric):
        ...     def __init__(self):
        ...         super().__init__('custom', 'Custom description')
        ...
        ...     def compute(self, tire, **kwargs):
        ...         return 42.0
    """

    def __init__(self, name: str, description: str = "", unit: str = ""):
        """
        Initialize metric.

        Args:
            name: Metric name (e.g., 'K_vert', 'mass_index')
            description: Human-readable description
            unit: Physical unit (if applicable)
        """
        self.name = name
        self.description = description
        self.unit = unit

    @abstractmethod
    def compute(self, tire: "Tire", **kwargs) -> float:
        """
        Compute the metric value.

        Args:
            tire: Tire object with components
            **kwargs: Additional parameters (e.g., force for deformation metrics)

        Returns:
            Metric value (float)
        """
        pass

    def visualize(self, tire: "Tire", value: float, **kwargs) -> Optional[Any]:
        """
        Optional: Visualize this metric.

        Override this method to provide custom visualizations for the metric.

        Args:
            tire: Tire object
            value: Computed metric value
            **kwargs: Additional visualization parameters

        Returns:
            matplotlib Figure or None
        """
        return None

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metric metadata.

        Returns:
            Dictionary with metric information
        """
        return {
            "name": self.name,
            "description": self.description,
            "unit": self.unit,
            "type": self.__class__.__name__,
        }

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(name='{self.name}')"
