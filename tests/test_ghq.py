"""Regression tests for :mod:`filters.ghq` (Gauss-Hermite quadrature).

Unreferenced module, and the one with the deepest defects:

* ``gh_nd_scaled`` used the upper Cholesky factor as if it were lower, so the
  sigma points implied a covariance that was a rotation of ``P``.
* ``gh_packed_pc`` indexed a ``(d, 1)`` mean elementwise and raised.
* ``gh_kf_predict`` defaulted ``Q`` to a 1-D array and added it to a matrix.
* ``gh_kf_update`` returned 5 or 6 values depending on ``h_param``.
"""

from __future__ import annotations

import numpy as np

from filters.ghq import (
    gh_kf_predict,
    gh_kf_update,
    gh_nd_scaled,
    gh_rts_smooth,
    gh_transform,
    hermite_polynomial,
)

MEAN = np.array([[1.0], [2.0]])
COV = np.array([[0.5, 0.1], [0.1, 0.3]])


def test_hermite_polynomial_matches_the_classic_recursion():
    # H_3(x) = 8x³ - 12x for the physicists' convention
    assert np.array_equal(hermite_polynomial(3), np.array([8, 0, -12, 0]))
    assert np.array_equal(hermite_polynomial(2), np.array([4, 0, -2]))


def test_quadrature_weights_sum_to_pi_for_two_dimensions():
    _, SX, _x, weights, _F = gh_nd_scaled(np.eye(2), 3, MEAN, COV)

    assert SX.shape == (2, 9)
    assert np.isclose(weights.sum(), np.pi)  # sqrt(pi)**d for d = 2


def test_identity_transform_reproduces_mean_and_covariance():
    """The upper/lower Cholesky mix-up rotated P by a few percent."""
    mean, cov, cross, *_ = gh_transform(MEAN, COV, np.eye(2))

    assert np.allclose(mean.ravel(), MEAN.ravel())
    assert np.allclose(cov, COV)
    assert cross.shape == (2, 2)


def test_identity_transform_accepts_a_flat_mean():
    mean, cov, *_ = gh_transform(MEAN.ravel(), COV, np.eye(2))

    assert np.allclose(mean.ravel(), MEAN.ravel())
    assert np.allclose(cov, COV)


def test_quadrature_integrates_a_quadratic_exactly():
    # E[x²] under N(m, P) is m² + P for the scalar case.
    m, P = np.array([[0.5]]), np.array([[2.0]])

    integral, *_ = gh_nd_scaled(lambda x: x**2, 3, m, P)

    assert np.isclose(integral.ravel()[0], 0.5**2 + 2.0)


def test_predict_with_default_process_noise_is_a_no_op():
    """``Q`` used to default to a 1-D zero array and be added to a matrix."""
    _mean, cov = gh_kf_predict(MEAN, COV, f=np.eye(2))

    assert np.all(np.isfinite(cov))
    assert np.allclose(cov, COV)


def test_predict_adds_process_noise():
    Q = np.eye(2) * 0.1
    _, cov = gh_kf_predict(MEAN, COV, f=np.eye(2), Q=Q)

    assert np.allclose(cov, COV + Q)


def test_update_gain_matches_the_linear_kalman_gain():
    Y = np.array([[1.2]])
    H = np.array([[1.0, 0.0]])
    R = np.array([[0.2]])

    _, _, K, _, S, _ = gh_kf_update(MEAN, COV, Y, H, R=R)

    expected_S = H @ COV @ H.T + R
    expected_K = COV @ H.T @ np.linalg.inv(expected_S)

    assert np.allclose(S, expected_S)
    assert np.allclose(K, expected_K)


def test_update_always_returns_six_values():
    """The arity depended on whether ``h_param`` was supplied."""
    Y = np.array([[1.2]])
    H = np.array([[1.0, 0.0]])
    R = np.array([[0.2]])

    with_param = gh_kf_update(MEAN, COV, Y, H, R=R, h_param=(0,))
    without_param = gh_kf_update(MEAN, COV, Y, H, R=R)

    assert len(with_param) == 6
    assert len(without_param) == 6


def test_update_without_measurement_noise_stays_finite():
    Y = np.array([[1.2]])
    H = np.array([[1.0, 0.0]])

    result = gh_kf_update(MEAN, COV, Y, H)

    assert len(result) == 6
    assert np.all(np.isfinite(result[4]))


def test_smoother_on_a_static_signal_returns_the_input_means():
    steps = 5
    means = np.tile(MEAN.ravel(), (steps, 1))
    covs = np.tile(COV, (steps, 1, 1))

    smoothed, smoothed_cov, gains = gh_rts_smooth(
        means, covs, np.eye(2), Q=np.zeros((2, 2)), p=3
    )

    assert smoothed.shape == means.shape
    assert smoothed_cov.shape == covs.shape
    assert gains.shape == (steps, 2, 2)
    assert np.allclose(smoothed, means)


def test_smoother_does_not_mutate_its_inputs():
    means = np.tile(MEAN.ravel(), (4, 1))
    covs = np.tile(COV, (4, 1, 1))
    means_before, covs_before = means.copy(), covs.copy()

    gh_rts_smooth(means, covs, np.eye(2), Q=np.eye(2) * 0.01, p=3)

    assert np.array_equal(means, means_before)
    assert np.array_equal(covs, covs_before)


def test_smoother_responds_to_process_noise():
    means = np.tile(MEAN.ravel(), (4, 1))
    covs = np.tile(COV, (4, 1, 1))

    *_, quiet = gh_rts_smooth(means, covs, np.eye(2), Q=np.zeros((2, 2)), p=3)
    *_, noisy = gh_rts_smooth(means, covs, np.eye(2), Q=np.eye(2), p=3)

    assert not np.array_equal(quiet, noisy)


def test_scalar_state_is_supported():
    m, P = np.array([[3.0]]), np.array([[0.25]])

    integral, SX, *_ = gh_nd_scaled(np.eye(1), 3, m, P)

    assert np.isclose(integral.ravel()[0], 3.0)
    assert SX.shape == (1, 3)
