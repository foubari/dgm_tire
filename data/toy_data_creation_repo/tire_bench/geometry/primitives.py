"""Geometric primitives for tire generation."""

import numpy as np
from typing import Tuple


def create_ellipse_mask(
    resolution: int,
    cx: float,
    cy: float,
    rx: float,
    ry: float
) -> np.ndarray:
    """
    Create ellipse mask.

    Args:
        resolution: Grid resolution
        cx, cy: Center coordinates
        rx, ry: Radii (x and y)

    Returns:
        Binary mask with 1 inside ellipse, 0 outside
    """
    y, x = np.ogrid[:resolution, :resolution]
    dist = ((x - cx) / max(rx, 1e-6)) ** 2 + ((y - cy) / max(ry, 1e-6)) ** 2
    return (dist <= 1.0).astype(np.float32)


def create_ring_mask(
    resolution: int,
    cx: float,
    cy: float,
    r_outer: float,
    r_inner: float
) -> np.ndarray:
    """
    Create ring (annulus) mask.

    Args:
        resolution: Grid resolution
        cx, cy: Center coordinates
        r_outer, r_inner: Outer and inner radii

    Returns:
        Binary mask with 1 in ring, 0 elsewhere
    """
    y, x = np.ogrid[:resolution, :resolution]
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    return ((dist >= r_inner) & (dist <= r_outer)).astype(np.float32)


def apply_smooth_transition(
    value_start: float,
    value_end: float,
    t: float,
    mode: str = "cosine"
) -> float:
    """
    Apply smooth transition between two values.

    Args:
        value_start: Starting value
        value_end: Ending value
        t: Interpolation parameter [0, 1]
        mode: Transition mode ('linear', 'cosine', 'cubic')

    Returns:
        Interpolated value
    """
    t = np.clip(t, 0.0, 1.0)

    if mode == "linear":
        ts = t
    elif mode == "cosine":
        ts = 0.5 * (1.0 - np.cos(np.pi * t))
    elif mode == "cubic":
        ts = 3 * t ** 2 - 2 * t ** 3
    else:
        raise ValueError(f"Unknown transition mode: {mode}")

    return value_start + ts * (value_end - value_start)
