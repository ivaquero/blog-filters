"""Shared type vocabulary for the Kalman-family estimator stack.

Single source of truth for:

* Numeric aliases (`Matrix`, `Vector`, `Measurement`) pinned to
  `npt.NDArray[np.float64]` so `ty`/`mypy`/`pyright` all agree.
* RNG helpers — `resolve_rng` centralises the "caller-supplied Generator or
  a fresh one" pattern that used to be scattered across every simulator.
* Protocols — `Filter`, `SigmaPointGenerator`, `Resampler` describe the
  runtime duck-typed contracts so IMM/MMAE and downstream helpers can accept
  conforming objects (including test doubles) without inheritance.
* NamedTuple return shapes — `BatchOutput`, `BatchPosteriorOutput`, `RtsOutput`,
  `UkfRtsOutput`. They unpack exactly like the historical tuples so existing
  notebooks keep working (`means, cov, _, _ = kf.batch_filter(zs)`), but
  `result.means` also type-checks.

Uses Python 3.14 native PEP 695 `type` aliases and PEP 649 deferred
annotations — no `from __future__ import annotations` needed.
"""

from typing import Any, NamedTuple, Protocol, runtime_checkable

import numpy as np

from shared import (
    CovarianceLayout,
    FloatLike,
    Matrix,
    MatrixLike,
    Measurement,
    MeasurementLike,
    StateLayout,
    Vector,
    resolve_rng,
)

# ---------------------------------------------------------------------------
# Numeric aliases — re-exported from the package-neutral ``shared`` module so
# models can import them without depending on ``filters``. Existing imports
# from this module stay source-compatible.
# ---------------------------------------------------------------------------

__all__ = [
    "Batch2Output",
    "BatchOutput",
    "BatchPosteriorOutput",
    "CovarianceLayout",
    "Filter",
    "FloatLike",
    "Matrix",
    "MatrixLike",
    "Measurement",
    "MeasurementLike",
    "Resampler",
    "RtsOutput",
    "SigmaPointGenerator",
    "StateLayout",
    "StateSpaceModelLike",
    "UkfRtsOutput",
    "Vector",
    "resolve_rng",
]


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

    def sigma_points(self, x: Vector, P: Matrix) -> Matrix: ...


class Resampler(Protocol):
    """Maps a weight vector to a resampled index vector."""

    def __call__(self, weights: Vector, rng: np.random.Generator) -> Vector: ...


@runtime_checkable
class StateSpaceModelLike(Protocol):
    """Structural contract for the motion models the estimators accept.

    ``ExtendedKalmanFilter.from_model`` and
    ``UnscentedKalmanFilter.from_model`` take anything with this shape, so
    ``models.ConstantVelocity()``, a user's own class, or a ``SimpleNamespace``
    of bound methods all work — no inheritance from the ``models`` package, and
    therefore no import from ``filters`` into ``models``.

    The signatures mirror the callables the filters already used:
    ``fx(x, dt)`` for propagation and ``hx(x)`` for measurement prediction, with
    ``jacobian_*`` supplying the linearisations an EKF needs.
    """

    @property
    def dim_x(self) -> int: ...

    @property
    def dim_z(self) -> int: ...

    def fx(
        self,
        x: Any,
        dt: float | None = None,
        *,
        t: float = 0.0,
        u: Any = None,
        w: Any = None,
    ) -> Vector: ...

    def hx(self, x: Any, v: Any = None) -> Measurement: ...

    def jacobian_x(self, x: Any) -> Matrix: ...

    def jacobian_h(self, x: Any) -> Matrix: ...

    def process_cov(self) -> Matrix: ...

    def measurement_cov(self) -> Matrix: ...


# ---------------------------------------------------------------------------
# NamedTuple return shapes
# ---------------------------------------------------------------------------


class BatchOutput(NamedTuple):
    """Return of `KalmanFilter.batch_filter` and similar APIs."""

    means: StateLayout
    covs: CovarianceLayout
    means_prior: StateLayout
    covs_prior: CovarianceLayout


class BatchPosteriorOutput(NamedTuple):
    """Return of UKF/H∞ `batch_filter` — posterior estimates only."""

    means: StateLayout
    covs: CovarianceLayout


# Historical name: the "2" meant two fields, not two filters.
Batch2Output = BatchPosteriorOutput


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
