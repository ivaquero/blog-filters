"""Process- and measurement-noise covariance helpers.

Both generators return a per-axis covariance block, optionally tiled across
spatial axes. The original implementations spelled out every matrix entry as a
literal; each of those literals is the outer product ``v @ v.T`` of a *noise
input vector* ``v`` that maps the driving white noise onto the state. Writing
``v`` instead of the entries keeps the ``dt`` powers and the factorial
normalisations in one line each, and makes the physical assumption visible:

======  ============================  ==================
``dim``  ``v``                         driving noise
======  ============================  ==================
2        ``[dt**2/2, dt]``             white acceleration
3        ``[dt**2/2, dt, 1]``          white acceleration
4        ``[dt**3/6, dt**2/2, dt, 1]``  white jerk
======  ============================  ==================

``dim=2`` and ``dim=3`` share the top-left block and both assume the noise
drives acceleration; ``dim=4`` assumes it drives jerk. That inconsistency is
inherited from filterpy, so it is preserved rather than "fixed" — pick the
generator by the physical noise source, not by the state count.
"""

import math
from typing import Any

import numpy as np

from filters._typing import Matrix

__all__ = ["order_by_derivative", "white_noise_continuous", "white_noise_discrete"]

#: ``(dt power, denominator)`` pairs per block dimension, highest state first.
_NOISE_INPUT_SPEC: dict[int, tuple[tuple[int, float], ...]] = {
    2: ((2, 2.0), (1, 1.0)),
    3: ((2, 2.0), (1, 1.0), (0, 1.0)),
    4: ((3, 6.0), (2, 2.0), (1, 1.0), (0, 1.0)),
}


def _tile(block: Matrix, block_size: int, *, order_by_dim: bool) -> Matrix:
    """Repeat ``block`` either per axis or per derivative.

    ``order_by_dim`` matches a state ordered ``[x, vx, y, vy, ...]``;
    ``order_by_derivative`` matches ``[x, y, ..., vx, vy, ...]``.
    """

    if block_size < 1:
        raise ValueError(f"block_size must be at least 1, got {block_size}")
    if order_by_dim:
        return np.asarray(np.kron(np.eye(block_size), block), dtype=float)
    return np.asarray(np.kron(block, np.eye(block_size)), dtype=float)


def order_by_derivative(Q: Any, dim: int, block_size: int) -> Matrix:
    """Reorder a per-derivative ``Q`` block into the per-axis state ordering.

    For ``dim=2``, ``block_size=2`` and states ``[x, y, vx, vy]`` the
    ``[position, velocity]`` coupling of each axis lands in the off-diagonal
    blocks. Equivalent to ``np.kron(Q, np.eye(block_size))``; the loop this
    replaced built one ``np.eye`` per matrix entry.
    """

    block = np.asarray(Q, dtype=float)
    if block.ndim != 2 or block.shape[0] != dim:
        raise ValueError(f"Q must be a ({dim}, {dim}) matrix, got {block.shape}")
    return np.kron(block, np.eye(block_size))


def white_noise_discrete(
    dim: int,
    dt: float = 1.0,
    var: float = 1.0,
    block_size: int = 1,
    *,
    order_by_dim: bool = True,
) -> Matrix:
    """Return the process-noise covariance of a discretely sampled plant.

    Parameters
    ----------
    dim : int
        States per axis: 2 for position/velocity, 3 for position/velocity/
        acceleration, 4 for the same plus jerk.
    dt : float, optional
        Sampling interval, by default 1.0.
    var : float, optional
        Variance of the driving white noise, by default 1.0.
    block_size : int, optional
        Number of spatial axes to tile, by default 1.
    order_by_dim : bool, optional
        Tile per axis (``[x, vx, y, vy]``) when True, per derivative
        (``[x, y, vx, vy]``) when False, by default True.

    Returns
    -------
    Matrix
        ``(dim * block_size, dim * block_size)`` covariance matrix.

    Raises
    ------
    ValueError
        If ``dim`` is not 2, 3 or 4, or ``block_size`` is below 1.
    """

    spec = _NOISE_INPUT_SPEC.get(dim)
    if spec is None:
        raise ValueError(f"dim must be 2, 3 or 4, got {dim}")

    input_vector = np.array([dt**power / denom for power, denom in spec])
    block = np.outer(input_vector, input_vector)
    return _tile(block, block_size, order_by_dim=order_by_dim) * var


def white_noise_continuous(
    dim: int,
    dt: float = 1.0,
    spectral_density: float = 1.0,
    block_size: int = 1,
    *,
    order_by_dim: bool = True,
) -> Matrix:
    """Return the process-noise covariance of a continuously driven plant.

    The block is the exact discretisation
    ``Q_ij = dt**(2m-i-j+1) / ((2m-i-j+1) (m-i)! (m-j)!)`` with ``m = dim - 1``,
    which reproduces the textbook fourth-, sixth- and eighth-order blocks.

    Parameters
    ----------
    dim : int
        States per axis, at least 2.
    dt : float, optional
        Sampling interval, by default 1.0.
    spectral_density : float, optional
        Spectral density of the driving white noise, by default 1.0.
    block_size : int, optional
        Number of spatial axes to tile, by default 1.
    order_by_dim : bool, optional
        Tile per axis when True, per derivative when False, by default True.

    Returns
    -------
    Matrix
        ``(dim * block_size, dim * block_size)`` covariance matrix.

    Raises
    ------
    ValueError
        If ``dim`` is below 2 or ``block_size`` is below 1.
    """

    if dim < 2:
        raise ValueError(f"dim must be at least 2, got {dim}")

    order = dim - 1
    block = np.zeros((dim, dim))
    for row in range(dim):
        for col in range(dim):
            power = 2 * order - row - col + 1
            block[row, col] = dt**power / (
                power * math.factorial(order - row) * math.factorial(order - col)
            )
    return _tile(block, block_size, order_by_dim=order_by_dim) * spectral_density
