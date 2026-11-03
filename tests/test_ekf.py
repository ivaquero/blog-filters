"""Regression tests for :class:`filters.kalman_ekf.ExtendedKalmanFilter`.

The linear-system limit of the EKF must match the closed-form Kalman filter
within float64 tolerance; that identity is what this file asserts. A second
test exercises the nonlinear range-bearing measurement path so we do not
silently regress on the Jacobian wiring.
"""

from __future__ import annotations

import math

import numpy as np

from filters.kalman import KalmanFilter
from filters.kalman_ekf import ExtendedKalmanFilter


def _HJac_linear(state, *_args):
    return np.array([[1.0, 0.0]])


def _Hx_linear(state, *_args):
    # EKF expects Hx to return a column vector shaped (dim_z, 1) or (1, dim_z).
    return np.array([[state[0, 0]]])


def test_ekf_matches_lkf_on_linear_system(linear_cv_truth):
    """With H(x) = H·x and a matching Jacobian, EKF ≡ LKF."""
    _, zs = linear_cv_truth
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    Q = np.array([[0.05, 0.05], [0.05, 0.05]])
    R = np.array([[0.25]])
    H = np.array([[1.0, 0.0]])

    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x = np.array([0.0, 1.0])
    kf.F, kf.H, kf.Q, kf.R = F, H, Q, R
    kf.P = np.eye(2)

    ekf = ExtendedKalmanFilter(dim_x=2, dim_z=1)
    ekf.x = np.array([[0.0], [1.0]])
    ekf.F = F
    ekf.Q = Q
    ekf.R = R
    ekf.P = np.eye(2)

    for z in zs:
        kf.predict()
        kf.update([z])

        ekf.predict()
        ekf.update(np.array([[z]]), _HJac_linear, _Hx_linear, R=R)

    assert np.allclose(kf.x.reshape(2), ekf.x.reshape(2), atol=1e-8)
    assert np.allclose(kf.P, ekf.P, atol=1e-8)


def test_ekf_range_bearing_update_reduces_uncertainty():
    """A single range/bearing update should shrink P's trace."""
    ekf = ExtendedKalmanFilter(dim_x=4, dim_z=2)
    ekf.x = np.array([[10.0], [1.0], [10.0], [0.0]])
    ekf.P = np.diag([5.0, 1.0, 5.0, 1.0])
    ekf.F = np.array(
        [[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]], dtype=float
    )
    ekf.Q = np.eye(4) * 0.01

    def Hx(state, *_args):
        x, _, y, _ = state.reshape(-1)
        rng = math.hypot(x, y)
        brg = math.atan2(y, x)
        return np.array([[rng], [brg]])

    def HJac(state, *_args):
        x, _, y, _ = state.reshape(-1)
        d = x * x + y * y
        return np.array(
            [[x / math.sqrt(d), 0, y / math.sqrt(d), 0], [-y / d, 0, x / d, 0]]
        )

    before = np.trace(ekf.P)
    ekf.update(np.array([[14.0], [math.pi / 4]]), HJac, Hx, R=np.diag([0.5, 0.01]))
    assert np.trace(ekf.P) < before
