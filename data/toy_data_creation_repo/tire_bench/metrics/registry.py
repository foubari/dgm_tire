"""Registry for performance metrics."""

from typing import Dict
from tire_bench.metrics.base import BaseMetric


class MetricRegistry:
    """
    Registry for metrics - enables config-driven metric selection.

    This registry allows dynamic creation of metrics from configuration files.

    Examples:
        >>> @MetricRegistry.register('my_metric')
        ... class MyMetric(BaseMetric):
        ...     pass
        >>> metric = MetricRegistry.create('my_metric')
    """

    _registry: Dict[str, type] = {}

    @classmethod
    def register(cls, metric_name: str):
        """
        Decorator to register metric classes.

        Args:
            metric_name: Metric identifier

        Returns:
            Decorator function

        Examples:
            >>> @MetricRegistry.register('vertical_stiffness')
            ... class VerticalStiffnessMetric(BaseMetric):
            ...     pass
        """

        def decorator(metric_class):
            if not issubclass(metric_class, BaseMetric):
                raise TypeError(f"{metric_class.__name__} must inherit from BaseMetric")
            cls._registry[metric_name] = metric_class
            return metric_class

        return decorator

    @classmethod
    def create(cls, metric_name: str, **kwargs) -> BaseMetric:
        """
        Factory method to create metrics from config.

        Args:
            metric_name: Name of metric to create
            **kwargs: Additional arguments for metric constructor

        Returns:
            Metric instance

        Raises:
            ValueError: If metric name is not registered
        """
        if metric_name not in cls._registry:
            raise ValueError(
                f"Unknown metric: {metric_name}. "
                f"Available metrics: {list(cls._registry.keys())}"
            )
        return cls._registry[metric_name](**kwargs)

    @classmethod
    def list_metrics(cls) -> list:
        """
        List all registered metrics.

        Returns:
            List of registered metric names
        """
        return list(cls._registry.keys())

    @classmethod
    def get_all_metrics(cls) -> Dict[str, BaseMetric]:
        """
        Get instances of all registered metrics.

        Returns:
            Dictionary mapping metric names to metric instances
        """
        return {name: cls.create(name) for name in cls._registry.keys()}
