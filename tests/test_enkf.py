"""Regression tests for :mod:`filters.kalman_enkf` (ensemble Kalman filter).

``update`` used to keep the analytic covariance ``P - K S K'`` while replacing
``x`` with the ensemble mean, leaving ``x``, ``P`` and ``sigmas`` mutually
inconsistent — the filter reported a covariance its own ensemble did not have.
"""

from __future__ import annotations

import numpy as np
import pytest

from filters.kalman_enkf import EnsembleKalmanFilter, outer_product_sum


def _identity_model(seed: int = 0, N: int = 400) -> EnsembleKalmanFilter:
    def fx(x, _dt):
        return np.asarray(x)

    def hx(x):
        return np.asarray(x)[:1]

    return EnsembleKalmanFilter(
        x=np.zeros(2),
        P=np.eye(2),
        dim_z=1,
        dt=1.0,
        N=N,
        hx=hx,
        fx=fx,
        rng=np.random.default_rng(seed),
    )


def test_update_reports_the_ensemble_covariance():
    enkf = _identity_model()
    enkf.update(np.array([1.0]))

    assert np.allclose(enkf.P, np.cov(enkf.sigmas.T))


def test_update_leaves_the_state_at_the_ensemble_mean():
    enkf = _identity_model()
    enkf.update(np.array([1.0]))

    assert np.allclose(enkf.x, enkf.sigmas.mean(axis=0))


def test_predict_reports_the_ensemble_covariance():
    enkf = _identity_model()
    enkf.predict()

    assert np.allclose(enkf.P, np.cov(enkf.sigmas.T))


def test_measurement_pulls_the_estimate_towards_the_observation():
    enkf = _identity_model()
    before = enkf.x.copy()

    enkf.update(np.array([5.0]))

    assert enkf.x[0] > before[0]


def test_update_with_none_is_a_no_op():
    enkf = _identity_model()
    x_before, P_before = enkf.x.copy(), enkf.P.copy()

    enkf.update(None)

    assert np.array_equal(enkf.x, x_before)
    assert np.array_equal(enkf.P, P_before)


def test_same_seed_gives_identical_runs():
    a, b = _identity_model(seed=11), _identity_model(seed=11)

    for z in (0.5, 1.0, 1.5):
        a.predict()
        a.update(np.array([z]))
        b.predict()
        b.update(np.array([z]))

    assert np.array_equal(a.sigmas, b.sigmas)


def test_rejects_degenerate_construction():
    with pytest.raises(ValueError, match="dim_z"):
        EnsembleKalmanFilter(
            np.zeros(2), np.eye(2), 0, 1.0, 10, lambda x: x, lambda x, _: x
        )
    with pytest.raises(ValueError, match="N must"):
        EnsembleKalmanFilter(
            np.zeros(2), np.eye(2), 1, 1.0, 0, lambda x: x[:1], lambda x, _: x
        )


def test_outer_product_sum_matches_the_gram_matrix():
    A = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    assert np.allclose(outer_product_sum(A), A.T @ A)
