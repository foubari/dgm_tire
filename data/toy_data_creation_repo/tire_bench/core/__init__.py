"""Core tire component abstractions."""

from tire_bench.core.component import ComponentBase, ComponentRegistry
from tire_bench.core.materials import MaterialProperties
from tire_bench.core.tire import Tire

__all__ = ["ComponentBase", "ComponentRegistry", "MaterialProperties", "Tire"]
