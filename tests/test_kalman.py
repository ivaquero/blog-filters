"""Regression tests for the linear Kalman filter (:mod:`filters.kalman`).

These lock in three properties that downstream notebooks rely on:

1. A zero-noise filter reproduces the analytic constant-velocity solution.
2. ``update`` with R → 0 collapses the posterior covariance toward zero and
   snaps the state to the measurement.
3. ``batch_filter`` and the equivalent predict/update loop agree bit-for-bit
   when the observation stream is fixed.
"""

from __future__ import annotations

import numpy as np
import pytest

from filters.kalman import KalmanFilter


@pytest.fixture
def cv_filter() -> KalmanFilter:
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.F = np.array([[1.0, 1.0], [0.0, 1.0]])
    kf.H = np.array([[1.0, 0.0]])
    kf.Q = np.zeros((2, 2))
    kf.R = np.array([[0.0]])
    kf.x = np.array([0.0, 1.0])
    kf.P = np.eye(2)
    return kf


def test_predict_propagates_state_with_unit_velocity(cv_filter):
    """Zero-noise CV prediction after N steps should reach x = N."""
    for _ in range(5):
        cv_filter.predict()
    assert np.allclose(cv_filter.x, [5.0, 1.0])
    assert cv_filter.P.shape == (2, 2)
    assert np.all(np.linalg.eigvalsh(cv_filter.P) >= -1e-12)


def test_update_with_zero_R_snaps_to_measurement(cv_filter):
    cv_filter.update(z=[3.0])
    assert np.isclose(cv_filter.x[0], 3.0, atol=1e-9)
    assert cv_filter.P[0, 0] < 1e-9


def test_batch_filter_matches_manual_loop(linear_cv_truth):
    xs, zs = linear_cv_truth
    dim_x, dim_z = 2, 1
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = np.array([[0.05, 0.05], [0.05, 0.05]])
    R = np.array([[0.25]])

    kf_a = KalmanFilter(dim_x=dim_x, dim_z=dim_z)
    kf_a.x = np.array([0.0, 1.0])
    kf_a.F, kf_a.H, kf_a.Q, kf_a.R = F, H, Q, R
    kf_a.P = np.eye(dim_x)
    means_a, covs_a, _, _ = kf_a.batch_filter(list(zs))

    kf_b = KalmanFilter(dim_x=dim_x, dim_z=dim_z)
    kf_b.x = np.array([0.0, 1.0])
    kf_b.F, kf_b.H, kf_b.Q, kf_b.R = F, H, Q, R
    kf_b.P = np.eye(dim_x)
    means_b, covs_b = [], []
    for z in zs:
        kf_b.predict()
        kf_b.update([z])
        means_b.append(kf_b.x.copy())
        covs_b.append(kf_b.P.copy())

    assert np.allclose(means_a, np.array(means_b), atol=1e-9)
    assert np.allclose(covs_a, np.array(covs_b), atol=1e-9)
    assert np.all(np.isfinite(means_a))
    assert xs.shape == (10, 2)


def test_rts_smoother_output_shape(cv_filter, linear_cv_truth):
    _, zs = linear_cv_truth
    cv_filter.R = np.array([[0.25]])
    cv_filter.Q = np.array([[0.05, 0.05], [0.05, 0.05]])
    means, covs, _, _ = cv_filter.batch_filter(list(zs))
    smoothed_x, smoothed_P, _, _ = cv_filter.rts_smoother(means, covs)

    assert smoothed_x.shape == means.shape
    assert smoothed_P.shape == covs.shape
    assert np.all(np.isfinite(smoothed_x))
    for k in range(len(covs)):
        assert np.linalg.det(smoothed_P[k]) <= np.linalg.det(covs[k]) + 1e-9
