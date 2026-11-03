"""Regression tests for :mod:`filters.imm`.

Locks in the P0 fixes from Phase 1:

- ``MMAEFilterBank.update`` must invoke the sub-filter with the correct
  keyword arguments (``f.update(z, H=H, R=R)``), not the MATLAB-style
  positional ``(z, R, H)`` that silently swapped the two.
- ``IMMEstimator`` requires at least two filters (guards against a
  degenerate 1-filter configuration).
"""

from __future__ import annotations

import numpy as np
import pytest

from filters.imm import IMMEstimator, MMAEFilterBank
from filters.kalman import KalmanFilter


def _cv_filter(Q: float) -> KalmanFilter:
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.F = np.array([[1.0, 1.0], [0.0, 1.0]])
    kf.H = np.array([[1.0, 0.0]])
    kf.Q = np.eye(2) * Q
    kf.R = np.array([[1.0]])
    kf.x = np.array([0.0, 1.0])
    kf.P = np.eye(2)
    return kf


def test_imm_estimator_runs_two_model_bank(linear_cv_truth):
    _, zs = linear_cv_truth
    filters = [_cv_filter(0.01), _cv_filter(0.5)]
    mu = np.array([0.5, 0.5])
    M = np.array([[0.9, 0.1], [0.1, 0.9]])
    imm = IMMEstimator(filters, mu, M)

    for z in zs:
        imm.predict()
        imm.update([z])

    assert np.isfinite(imm.x).all()
    assert np.isclose(imm.mu.sum(), 1.0)
    assert np.allclose(imm.P, imm.P.T)


def test_imm_requires_two_filters():
    with pytest.raises(ValueError, match="at least two"):
        IMMEstimator([_cv_filter(0.1)], mu=[1.0], M=np.array([[1.0]]))


def test_mmae_update_routes_H_and_R_by_keyword(monkeypatch):
    """Prior to Phase 1, ``f.update(z, R, H)`` was passed positionally.

    That call bound ``R`` to ``H`` and ``H`` to ``R`` on the underlying LKF,
    silently corrupting every MMAE estimate. The fix threads them as
    keywords — assert on the recorded call rather than on numeric output so
    the test stays stable across future tuning.
    """
    calls = []

    class SpyFilter:
        def __init__(self):
            self.x = np.array([0.0, 1.0])
            self.P = np.eye(2)
            self.z = np.array([0.0])
            self.likelihood = 1.0

        def update(self, z, H=None, R=None):
            calls.append({"z": z, "H": H, "R": R})
            self.x = np.array([1.0, 1.0])

    spy_a, spy_b = SpyFilter(), SpyFilter()
    bank = MMAEFilterBank([spy_a, spy_b], p=[0.5, 0.5], dim_x=2)

    H_override = np.array([[1.0, 0.0]])
    R_override = np.array([[0.25]])
    z = np.array([1.2])
    bank.update(z, H=H_override, R=R_override)

    assert len(calls) == 2, "each sub-filter should have received one update"
    for record in calls:
        # H and R must arrive as their respective keyword args, not swapped.
        assert np.array_equal(record["H"], H_override)
        assert np.array_equal(record["R"], R_override)
        np.testing.assert_array_equal(record["z"], z)


def test_mmae_rejects_mismatched_filter_count():
    with pytest.raises(ValueError, match="filters and p"):
        MMAEFilterBank([_cv_filter(0.01)], p=[0.5, 0.5], dim_x=2)
