"""Angle wrapping shared by the simulators.

``robot.normalize_angle`` (radians, documented as ``[-pi, pi)`` but returning
``(-pi, pi]``) and ``datagen._wrap_heading`` (radians, ``[0, 2*pi)``) disagreed
about both the range and the convention. Everything that wraps an angle now
goes through these two functions, which share one convention: the result lies
in ``[center - half_period, center + half_period)``.
"""

import numpy as np

__all__ = ["TAU", "wrap_angle", "wrap_degrees"]

TAU = 2.0 * np.pi


def wrap_angle(angle, *, center: float = 0.0):
    """Wrap an angle in radians into ``[center - pi, center + pi)``.

    Works on scalars and arrays alike. ``pi`` and ``-pi`` both map to ``-pi``,
    matching the half-open interval rather than distinguishing the two ends.
    """

    return (np.asarray(angle) - center + np.pi) % TAU - np.pi + center


def wrap_degrees(angle, *, center: float = 0.0):
    """Wrap an angle in degrees into ``[center - 180, center + 180)``."""

    return (np.asarray(angle) - center + 180.0) % 360.0 - 180.0 + center
