"""Package-neutral numeric vocabulary shared by ``filters`` and ``models``.

These aliases and the RNG helper used to live in ``filters._typing``, which made
``models`` — the layer ``filters`` consumes — import from ``filters`` in turn.
Both packages now depend on this leaf module instead, so no filter needs to
import a model (or vice versa) merely to share a type name.

``filters._typing`` re-exports everything here, so all pre-existing imports such
as ``from filters._typing import Matrix`` keep working unchanged.

Uses Python 3.14 native PEP 695 ``type`` aliases — no ``from __future__`` needed.
"""

import numpy as np
import numpy.typing as npt

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
    "CovarianceLayout",
    "FloatLike",
    "Matrix",
    "MatrixLike",
    "Measurement",
    "MeasurementLike",
    "StateLayout",
    "Vector",
    "resolve_rng",
]


def resolve_rng(rng: np.random.Generator | None = None) -> np.random.Generator:
    """Return `rng` when provided, otherwise a fresh unseeded generator.

    Every simulator, resampler and plotting helper that touches randomness
    accepts a `numpy.random.Generator`, so a caller can pin a stream, share
    it across several objects, or thread it through a test fixture. Passing
    `None` matches `np.random.default_rng()` semantics.
    """

    return np.random.default_rng() if rng is None else rng
