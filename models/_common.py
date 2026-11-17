"""Shared numeric helpers for the state-space model package.

Everything in here was previously copy-pasted into both ``ssmodel_linear`` and
``ssmodel_nonlinear`` (five identical private functions). Keeping one copy makes
the dimension bookkeeping auditable in a single place: a state vector of
``dim_x`` entries grouped as ``state_dim_per_axis`` values per spatial axis must
divide evenly, and every covariance the factories accept is validated here
rather than silently reshaped.
"""

from typing import Any

import numpy as np

from shared import Matrix, Vector

__all__ = [
    "as_covariance",
    "as_state_batch",
    "as_state_vector",
    "axis_count",
    "block_diag_axis",
    "numerical_jacobian",
    "position_measurement_matrix",
]


def axis_count(dim_x: int, state_dim_per_axis: int) -> int:
    """Return the number of spatial axes encoded in ``dim_x``.

    Raises `ValueError` when ``dim_x`` is not a positive multiple of
    ``state_dim_per_axis``, which is the only way a mis-specified state
    ordering can be caught before it produces a plausible but wrong matrix.
    """

    if state_dim_per_axis < 1:
        raise ValueError("state_dim_per_axis must be at least 1")
    if dim_x < state_dim_per_axis or dim_x % state_dim_per_axis != 0:
        raise ValueError(
            f"dim_x must be a positive multiple of {state_dim_per_axis}, got {dim_x}"
        )
    return dim_x // state_dim_per_axis


def as_state_vector(x: Any) -> Vector:
    """Coerce ``x`` to a 1-D float64 array without copying when possible."""

    vector = np.asarray(x, dtype=float)
    return vector.reshape(-1) if vector.ndim > 1 else vector


def as_state_batch(x: Any, dim_x: int | None = None) -> Matrix:
    """Coerce one state or a stack of states to a ``(n, dim_x)`` batch.

    A 1-D input becomes a single row so callers can pass either shape without
    branching. A 2-D input is used as-is. Anything else is rejected: silently
    flattening a ``(n, dim_x, k)`` tensor used to hide shape bugs deep inside a
    particle filter. ``dim_x``, when given, validates the column count.
    """

    batch = np.asarray(x, dtype=float)
    if batch.ndim == 1:
        batch = batch.reshape(1, -1)
    if batch.ndim != 2:
        raise ValueError(f"state must be 1-D or 2-D, got shape {batch.shape}")
    if dim_x is not None and batch.shape[1] != dim_x:
        raise ValueError(
            f"state batch must have {dim_x} columns, got shape {batch.shape}"
        )
    return batch


def as_covariance(value: Any, dim: int, name: str = "P") -> Matrix:
    """Expand a scalar variance to ``value * I`` or validate a full matrix.

    A scalar is broadcast onto the diagonal. An array must already be a
    symmetric ``(dim, dim)`` matrix: a 1-D input used to be quietly promoted to
    a single-row ``(1, dim)`` matrix by ``np.atleast_2d``, which then produced
    broadcast-shaped garbage further down the pipeline.
    """

    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        return np.eye(dim) * float(array)
    if array.shape != (dim, dim):
        raise ValueError(
            f"{name} must be a scalar or a ({dim}, {dim}) matrix, "
            f"got shape {array.shape}"
        )
    if not np.allclose(array, array.T, rtol=1e-10, atol=1e-12):
        raise ValueError(f"{name} must be symmetric")
    return array


def block_diag_axis(axis_matrix: Matrix, axes: int) -> Matrix:
    """Tile a per-axis block matrix into a block-diagonal state matrix."""

    return np.asarray(np.kron(np.eye(axes), axis_matrix), dtype=float)


def position_measurement_matrix(
    dim_x: int, state_dim_per_axis: int, dim_z: int | None = None
) -> Matrix:
    """Build a matrix that measures the first state of each axis.

    ``dim_z=None`` measures position on every axis; ``dim_z == dim_x`` returns
    the identity for full-state measurements.
    """

    axes = axis_count(dim_x, state_dim_per_axis)
    dim_z = axes if dim_z is None else dim_z

    if dim_z == dim_x and dim_z > 0:
        return np.eye(dim_x)
    if dim_z < 1:
        raise ValueError(f"dim_z must be at least 1, got {dim_z}")
    if dim_z > axes:
        raise ValueError(
            f"dim_z must be <= {axes} (one per axis) or equal dim_x ({dim_x}), "
            f"got {dim_z}"
        )

    matrix = np.zeros((dim_z, dim_x))
    matrix[np.arange(dim_z), np.arange(dim_z) * state_dim_per_axis] = 1.0
    return matrix


def numerical_jacobian(
    fn: Any, x: Any, step: float | None = None, eps: float = 1e-6
) -> Matrix:
    """Central-difference Jacobian of ``fn`` at ``x``.

    Gives nonlinear models a working ``df/dx`` without hand-derived algebra, so
    a new model only has to implement ``fx``/``hx`` to be usable by an EKF.
    """

    vector = as_state_vector(x)
    scale = max(1.0, float(np.max(np.abs(vector)))) if vector.size else 1.0
    delta = eps * scale if step is None else step

    columns = []
    for index in range(vector.size):
        perturbation = np.zeros_like(vector)
        perturbation[index] = delta
        columns.append(
            (fn(vector + perturbation) - fn(vector - perturbation)) / (2.0 * delta)
        )
    if not columns:
        return np.zeros((0, 0))
    return np.column_stack([as_state_vector(column) for column in columns])
