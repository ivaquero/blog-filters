"""Multi-sensor fusion examples built on the Kalman and UKF filters."""

from __future__ import annotations

import numpy as np

from models.const_vel import constant_velocity_transition_matrix

from ._typing import resolve_rng
from .helpers import KFSaver
from .kalman import KalmanFilter
from .kalman_ukf import UnscentedKalmanFilter
from .sigma_points import JulierSigmas


def fusion_kf2d(
    sensor1_sigma: float,
    sensor2_sigma: float,
    dt: float = 0.1,
    P: float = 100,
    rng: np.random.Generator | None = None,
) -> KFSaver:
    """Two-sensor 1-D fusion demo; returns a KFSaver with the filter history."""
    generator = resolve_rng(rng)

    kf = KalmanFilter(dim_x=2, dim_z=2)
    kf.x = np.array([[0.0], [1.0]])
    kf.F = constant_velocity_transition_matrix(2, dt)
    kf.H = np.array([[1.0, 0.0], [1.0, 0.0]])
    kf.P *= P
    kf.Q *= np.array([[(dt**3) / 3, (dt**2) / 2], [(dt**2) / 2, dt]]) * 0.02
    kf.R = np.array([[sensor1_sigma**2, 0], [0, sensor2_sigma**2]])
    saver = KFSaver(kf)

    for i in range(1, 100):
        m0 = i + generator.standard_normal() * sensor1_sigma
        m1 = i + generator.standard_normal() * sensor2_sigma
        kf.predict()
        kf.update(np.array([[m0], [m1]]))
        saver.save()
    saver.to_array()
    print(f"fusion std: {np.std(saver.y[:, 0]):.3f}")

    return saver


def fusion_ukf2d(hx, fx) -> UnscentedKalmanFilter:
    """UKF variant of the 2-D fusion setup with a large initial covariance."""
    # create unscented Kalman filter with large initial uncertainty
    points = JulierSigmas(2, kappa=2)
    ukf = UnscentedKalmanFilter(2, 2, 0.1, hx=hx, fx=fx, points=points)
    ukf.x = np.array([100, 100])
    ukf.P *= 40
    return ukf
