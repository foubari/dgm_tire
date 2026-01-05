"""
Tire Deformation Benchmark

A modular, extensible benchmark for tire geometry generation
and mechanical deformation simulation.
"""

__version__ = "0.1.0"

from tire_bench.core.tire import Tire
from tire_bench.core.materials import MaterialProperties
from tire_bench.core.component import ComponentBase, ComponentRegistry

__all__ = [
    "Tire",
    "MaterialProperties",
    "ComponentBase",
    "ComponentRegistry",
]
