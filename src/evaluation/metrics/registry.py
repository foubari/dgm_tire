"""
Metric registry for automatic discovery and instantiation.

Provides a plugin-like architecture for adding/removing metrics easily.
"""

from typing import Dict, Type, List
from .base_metric import CacheableMetric


# Global metric registry
_METRIC_REGISTRY: Dict[str, Type[CacheableMetric]] = {}


def register_metric(name: str):
    """
    Decorator to register a metric class.

    Usage:
        @register_metric('my_metric')
        class MyMetric(CacheableMetric):
            ...
    """
    def decorator(cls):
        _METRIC_REGISTRY[name] = cls
        return cls
    return decorator


def get_metric_class(name: str) -> Type[CacheableMetric]:
    """
    Get metric class by name.

    Args:
        name: Metric name (e.g., 'fid', 'iou_dice')

    Returns:
        Metric class, or None if not found
    """
    return _METRIC_REGISTRY.get(name)


def list_available_metrics() -> List[str]:
    """
    List all registered metric names.

    Returns:
        List of metric names
    """
    return list(_METRIC_REGISTRY.keys())


# ============================================================================
# Auto-register all metrics
# ============================================================================

from .fid import FIDMetric
from .iou_dice import IoUDiceMetric
from .rce import RCEMetric
from .com import CoMMetric

# Register each metric
register_metric('fid')(FIDMetric)
register_metric('iou_dice')(IoUDiceMetric)
register_metric('rce')(RCEMetric)
register_metric('com')(CoMMetric)
