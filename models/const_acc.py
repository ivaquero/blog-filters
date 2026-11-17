"""Constant-acceleration model exports.

The implementation moved to :mod:`models.ssmodel_linear` — a constant
acceleration plant is linear, and its closed-form matrices belong next to the
constant-velocity ones. This module keeps the historical import path and the
``HCA``/``FCA``/``KFCA`` aliases working.
"""

from .ssmodel_linear import (
    FCA,
    HCA as H,
    KFCA,
    ConstantAcceleration,
    FxCA,
    KFCA3d,
    constant_acceleration_F,
    constant_acceleration_H,
    constant_acceleration_L,
    constant_acceleration_filter,
    constant_acceleration_filter_3d,
    constant_acceleration_fx,
)

__all__ = [
    "FCA",
    "KFCA",
    "ConstantAcceleration",
    "FxCA",
    "H",
    "KFCA3d",
    "constant_acceleration_F",
    "constant_acceleration_H",
    "constant_acceleration_L",
    "constant_acceleration_filter",
    "constant_acceleration_filter_3d",
    "constant_acceleration_fx",
]
