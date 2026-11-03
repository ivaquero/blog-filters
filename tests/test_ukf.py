"""Regression tests for :class:`filters.kalman_ukf.UnscentedKalmanFilter`.

The UKF is only as good as its sigma-point machinery, so we assert two things:
(1) the weighted mean/covariance of the raw sigma points reproduces the input
``(x, P)`` exactly, and (2) a one-step prediction followed by an update on a
1-D linear measurement reduces the posterior variance.
"""

from __future__ import annotations

import numpy as np
import pytest

from filters.kalman_ukf import UnscentedKalmanFilter, unscented_transform
from filters.sigma_points import MerweScaledSigmas


@pytest.fixture
def points_2d() -> MerweScaledSigmas:
    return MerweScaledSigmas(n=2, kappa=3 - 2, alpha=1e-3, beta=2.0)


def test_weights_sum_to_one(points_2d):
    """Merwe's mean weights sum to one; the covariance weights do not (the
    β correction breaks that identity on purpose)."""
    assert np.isclose(points_2d.Wm.sum(), 1.0)
    assert points_2d.Wc.shape == points_2d.Wm.shape


def test_unscented_transform_recovers_mean_and_covariance(points_2d):
    x = np.array([1.0, -2.0])
    P = np.array([[2.0, 0.3], [0.3, 1.5]])
    sigmas = points_2d.sigma_points(x, P)
    mean, cov = unscented_transform(sigmas, points_2d.Wm, points_2d.Wc)
    assert np.allclose(mean, x, atol=1e-9)
    assert np.allclose(cov, P, atol=1e-9)


def test_ukf_predict_update_cycle_shrinks_variance():
    def fx(x, *_args, **_kwargs):
        # constant-velocity step with dt = 1
        return np.array([x[0] + x[1], x[1]])

    def hx(x, *_args, **_kwargs):
        return np.array([x[0]])

    points = MerweScaledSigmas(n=2, kappa=3 - 2, alpha=1.0, beta=0.0)
    ukf = UnscentedKalmanFilter(2, 1, dt=1.0, hx=hx, fx=fx, points=points)
    ukf.x = np.array([0.0, 1.0])
    ukf.P = np.eye(2) * 5.0
    ukf.Q = np.eye(2) * 0.01
    ukf.R = np.array([[0.25]])

    prior_var = None
    for z in [1.0, 2.1, 3.2, 4.0, 5.1]:
        ukf.predict()
        posterior_var_before = ukf.P[0, 0]
        ukf.update(np.array([z]))
        posterior_var_after = ukf.P[0, 0]
        assert posterior_var_after <= posterior_var_before + 1e-12
        prior_var = posterior_var_after

    # After a handful of measurements with a well-specified model the
    # variance in the observed coordinate should have collapsed substantially.
    assert prior_var < 1.0
