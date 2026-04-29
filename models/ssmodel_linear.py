from __future__ import annotations

import numpy as np

from filters.kalman import KalmanFilter

from .noise import white_noise_discrete
from .ssmodel import StateSpaceModel


class LinearStateSpaceModel(StateSpaceModel):
    """Linear time varying system.

    State space model of the plant.
    x[t+1] = F[t]*x[t] + G[t]*u[t] + L[t]*w[t]
    z[t] = H[t]*x[t] + M[t]*v[t]

    x: state
    z: output
    u: control input
    w: system noise
    v: observation noise
    """

    def Ft(self, t):
        """x[t+1] = F[t]*x[t] + G[t]*u[t] + L[t]*w[t].

        return F[t].
        """
        raise NotImplementedError

    def Gt(self, t):
        """x[t+1] = F[t]*x[t] + G[t]*u[t] + L[t]*w[t].

        return G[t].
        """
        raise NotImplementedError

    def Lt(self, t):
        """x[t+1] = F[t]*x[t] + G[t]*u[t] + L[t]*w[t].

        return L[t].
        """
        raise NotImplementedError

    def Ht(self, t):
        """y[t] = H[t]*x[t] + M[t]*v[t].

        return H[t].
        """
        raise NotImplementedError

    def Mt(self, t):
        """y[t] = H[t]*x[t] + M[t]*v[t].

        return M[t].
        """
        raise NotImplementedError


CV_STATE_DIM_PER_AXIS = 2


def _cv_axis_count(dim_x: int) -> int:
    if dim_x < CV_STATE_DIM_PER_AXIS or dim_x % CV_STATE_DIM_PER_AXIS != 0:
        raise ValueError("dim_x must be a positive multiple of 2")
    return dim_x // CV_STATE_DIM_PER_AXIS


def _as_state_vector(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def constant_velocity_H(dim_x: int, dim_z: int | None = None) -> np.ndarray:
    """Return a measurement matrix for constant-velocity states.

    State order is ``[x, vx, y, vy, ...]``. By default, the matrix measures
    position only for every spatial axis. Passing ``dim_z == dim_x`` returns
    an identity matrix for full-state measurements.
    """
    axes = _cv_axis_count(dim_x)
    dim_z = axes if dim_z is None else dim_z

    if dim_z < 1:
        raise ValueError("dim_z must be at least 1")
    if dim_z == dim_x:
        return np.eye(dim_x)
    if dim_z > axes:
        raise ValueError("dim_z must be <= number of axes, or equal to dim_x")

    H = np.zeros((dim_z, dim_x))
    H[np.arange(dim_z), np.arange(dim_z) * CV_STATE_DIM_PER_AXIS] = 1.0
    return H


def constant_velocity_F(dim: int, dt: float) -> np.ndarray:
    """Return the transition matrix for a constant-velocity model.

    Supports any state dimension that is a multiple of 2.
    """
    axes = _cv_axis_count(dim)
    F_axis = np.array([[1, dt], [0, 1]], dtype=float)
    return np.kron(np.eye(axes), F_axis)


def constant_velocity_fx(x, dt: float) -> np.ndarray:
    """Apply a constant-velocity transition to state ``x``."""
    x = _as_state_vector(x)
    return constant_velocity_F(len(x), dt) @ x


def constant_velocity_filter(
    P, R, Q=0, dt: float = 1, x=(0, 0), dim_z: int | None = None
):
    """Create a Kalman filter configured for constant velocity.

    The state is ordered per axis as ``[position, velocity]``. For example,
    a 2D state is ``[x, vx, y, vy]``.
    """
    x = _as_state_vector(x)
    dim_x = len(x)
    axes = _cv_axis_count(dim_x)
    dim_z = axes if dim_z is None else dim_z

    kf_cv = KalmanFilter(dim_x=dim_x, dim_z=dim_z)
    kf_cv.x = x
    kf_cv.F = constant_velocity_F(dim_x, dt)
    kf_cv.H = constant_velocity_H(dim_x, dim_z)
    kf_cv.P = np.eye(dim_x) * P if np.isscalar(P) else np.atleast_2d(P)
    kf_cv.R = np.eye(dim_z) * R if np.isscalar(R) else np.atleast_2d(R)
    kf_cv.Q = (
        white_noise_discrete(dim=CV_STATE_DIM_PER_AXIS, dt=dt, var=Q, block_size=axes)
        if np.isscalar(Q)
        else np.atleast_2d(Q)
    )
    return kf_cv


def constant_velocity_filter_1d(P, R, Q=0, dt=1, x=(0,)):
    """Backward-compatible 1D filter factory.

    The historical default is a single-state constant-value model. Passing a
    two-element state returns the 1-axis constant-velocity model.
    """
    x = _as_state_vector(x)
    if len(x) != 1:
        return constant_velocity_filter(P=P, R=R, Q=Q, dt=dt, x=x, dim_z=1)

    kf_cv = KalmanFilter(dim_x=1, dim_z=1)
    kf_cv.x = x
    kf_cv.F = np.eye(1)
    kf_cv.H = np.eye(1)
    kf_cv.P = np.eye(1) * P if np.isscalar(P) else np.atleast_2d(P)
    kf_cv.R = np.eye(1) * R if np.isscalar(R) else np.atleast_2d(R)
    kf_cv.Q = np.eye(1) * Q if np.isscalar(Q) else np.atleast_2d(Q)
    return kf_cv


def constant_velocity_filter_2d(P, R, Q=0, dt=1, x=(0, 0)):
    """Backward-compatible constant-velocity filter factory.

    Historically this function initialized a 1-axis ``[x, vx]`` filter. It now
    also works for any even-length state while keeping the original default.
    """
    return constant_velocity_filter(P=P, R=R, Q=Q, dt=dt, x=x, dim_z=1)


class ConstantVelocity(LinearStateSpaceModel):
    """Linear 2D constant-velocity state-space model."""

    NDIM = {"x": 4, "z": 2, "u": 0, "w": 2, "v": 2}

    def __init__(self, dt=0.1):
        self.F = constant_velocity_F(4, dt)
        self.L = np.array([[0.5 * (dt**2), 0], [dt, 0], [0, 0.5 * (dt**2)], [0, dt]])
        self.H = constant_velocity_H(4)
        self.M = np.eye(2)

    def state_equation(self, t, x, u=0, w=None):
        """Return ``x[t+1] = F[t] * x[t] + L[t] * w[t]``."""
        if w is None:
            w = np.zeros(self.NDIM["w"])
        return self.F @ x + self.L @ w

    def observation_equation(self, t, x, v=None):
        """Return ``z[t] = H[t] * x[t] + M[t] * v[t]``."""
        if v is None:
            v = np.zeros(self.NDIM["v"])
        return self.H @ x + self.M @ v

    def Ft(self, t):
        """Return the transition matrix at time ``t``."""
        return self.F

    def Lt(self, t):
        """Return the process-noise input matrix at time ``t``."""
        return self.L

    def Ht(self, t):
        """Return the measurement matrix at time ``t``."""
        return self.H

    def Mt(self, t):
        """Return the measurement-noise matrix at time ``t``."""
        return self.M


HCV = constant_velocity_H
FCV = constant_velocity_F
FxCV = constant_velocity_fx
KFCV = constant_velocity_filter
KFCV1d = constant_velocity_filter_1d
KFCV2d = constant_velocity_filter_2d
