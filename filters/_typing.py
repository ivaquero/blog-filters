"""Shared type vocabulary for the Kalman-family estimator stack.

Single source of truth for:

* Numeric aliases (`Matrix`, `Vector`, `Measurement`) pinned to
  `npt.NDArray[np.float64]` so `ty`/`mypy`/`pyright` all agree.
* RNG helpers — `resolve_rng` centralises the "caller-supplied Generator or
  a fresh one" pattern that used to be scattered across every simulator.
* Protocols — `Filter`, `SigmaPointGenerator`, `Resampler` describe the
  runtime duck-typed contracts so IMM/MMAE and downstream helpers can accept
  conforming objects (including test doubles) without inheritance.
* NamedTuple return shapes — `BatchOutput`, `RtsOutput`, `SmootherOutput`,
  `IMMOutput`. They unpack exactly like the historical tuples so existing
  notebooks keep working (`means, cov, _, _ = kf.batch_filter(zs)`), but
  `result.means` also type-checks.

Uses Python 3.14 native PEP 695 `type` aliases and PEP 649 deferred
annotations — no `from __future__ import annotations` needed.
"""

from typing import NamedTuple, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

# ---------------------------------------------------------------------------
# Numeric aliases
# ---------------------------------------------------------------------------

type Matrix = npt.NDArray[np.float64]
type Vector = npt.NDArray[np.float64]
type Measurement = npt.NDArray[np.float64]
# Any caller-supplied measurement — ndarray, list, tuple, or scalar — before
# `reshape_z` normalises it into a `Measurement`.
type MeasurementLike = npt.ArrayLike

type FloatLike = float | np.floating
type MatrixLike = Matrix | FloatLike

type StateLayout = npt.NDArray[np.float64]
type CovarianceLayout = npt.NDArray[np.float64]

__all__ = [
    "Batch2Output",
    "BatchOutput",
    "CovarianceLayout",
    "Filter",
    "FloatLike",
    "IMMOutput",
    "Matrix",
    "MatrixLike",
    "Measurement",
    "MeasurementLike",
    "Resampler",
    "RtsOutput",
    "SigmaPointGenerator",
    "SmootherOutput",
    "StateLayout",
    "UkfRtsOutput",
    "Vector",
    "resolve_rng",
]


# ---------------------------------------------------------------------------
# RNG helper
# ---------------------------------------------------------------------------


def resolve_rng(rng: np.random.Generator | None = None) -> np.random.Generator:
    """Return `rng` when provided, otherwise a fresh unseeded generator.

    Every simulator, resampler and plotting helper that touches randomness
    accepts a `numpy.random.Generator`, so a caller can pin a stream, share
    it across several objects, or thread it through a test fixture. Passing
    `None` matches `np.random.default_rng()` semantics.
    """

    return np.random.default_rng() if rng is None else rng


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class Filter(Protocol):
    """Structural contract shared by every Kalman-family filter.

    Anything that keeps `x`/`P` state, exposes `dim_x`/`dim_z`, and
    implements `predict`/`update` satisfies the protocol. IMM, MMAE, savers,
    and downstream plotting helpers accept `Filter` rather than a concrete
    base class, so test doubles and third-party filters compose freely.
    """

    dim_x: int
    dim_z: int

    x: Vector
    P: Matrix
    Q: Matrix
    R: Matrix
    F: Matrix
    H: Matrix
    y: Vector
    z: Vector
    K: Matrix

    def predict(self, *args: object, **kwargs: object) -> None: ...

    def update(
        self, z: Measurement | None, *args: object, **kwargs: object
    ) -> None: ...


class SigmaPointGenerator(Protocol):
    """Turns (mean, covariance) into sigma points and their weights."""

    def num_sigmas(self) -> int: ...

    def sigma_points(self, x: Vector, P: Matrix) -> tuple[Vector, Vector, Vector]: ...


class Resampler(Protocol):
    """Maps a weight vector to a resampled index vector."""

    def __call__(self, weights: Vector, rng: np.random.Generator) -> Vector: ...


# ---------------------------------------------------------------------------
# NamedTuple return shapes
# ---------------------------------------------------------------------------


class BatchOutput(NamedTuple):
    """Return of `KalmanFilter.batch_filter` and similar APIs."""

    means: StateLayout
    covs: CovarianceLayout
    means_prior: StateLayout
    covs_prior: CovarianceLayout


class Batch2Output(NamedTuple):
    """Return of UKF/Hinf `batch_filter` — posterior estimates only."""

    means: StateLayout
    covs: CovarianceLayout


class RtsOutput(NamedTuple):
    """Return of `KalmanFilter.rts_smoother` / `smoothers.rts_smoother`."""

    x: StateLayout
    P: CovarianceLayout
    K: CovarianceLayout
    P_prior: CovarianceLayout


class UkfRtsOutput(NamedTuple):
    """Return of `UnscentedKalmanFilter.rts_smoother` (3-tuple)."""

    x: StateLayout
    P: CovarianceLayout
    K: CovarianceLayout


class SmootherOutput(NamedTuple):
    """Fixed-lag smoother output."""

    x_smoothed: StateLayout
    P_smoothed: CovarianceLayout


class IMMOutput(NamedTuple):
    """IMM / MMAE estimator snapshot after an update or predict step."""

    x: Vector
    P: Matrix
    mu: Vector
