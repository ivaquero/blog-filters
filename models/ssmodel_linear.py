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

    def transition_matrix(self, t):
        """x[t+1] = F[t]*x[t] + G[t]*u[t] + L[t]*w[t].

        return F[t].
        """
        raise NotImplementedError

    def control_input_matrix(self, t):
        """x[t+1] = F[t]*x[t] + G[t]*u[t] + L[t]*w[t].

        return G[t].
        """
        raise NotImplementedError

    def process_noise_input(self, t):
        """x[t+1] = F[t]*x[t] + G[t]*u[t] + L[t]*w[t].

        return L[t].
        """
        raise NotImplementedError

    def measurement_matrix(self, t):
        """y[t] = H[t]*x[t] + M[t]*v[t].

        return H[t].
        """
        raise NotImplementedError

    def measurement_noise_matrix(self, t):
        """y[t] = H[t]*x[t] + M[t]*v[t].

        return M[t].
        """
        raise NotImplementedError


CV_STATE_DIM_PER_AXIS = 2


def _axis_count(dim_x: int, state_dim_per_axis: int) -> int:
    if dim_x < state_dim_per_axis or dim_x % state_dim_per_axis != 0:
        raise ValueError(f"dim_x must be a positive multiple of {state_dim_per_axis}")
    return dim_x // state_dim_per_axis


def _as_state_vector(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def _as_covariance(value, dim: int) -> np.ndarray:
    return np.eye(dim) * value if np.isscalar(value) else np.atleast_2d(value)


def _position_measurement_matrix(
    dim_x: int, state_dim_per_axis: int, dim_z: int | None = None
) -> np.ndarray:
    axes = _axis_count(dim_x, state_dim_per_axis)
    dim_z = axes if dim_z is None else dim_z

    if dim_z < 1:
        raise ValueError("dim_z must be at least 1")
    if dim_z == dim_x:
        return np.eye(dim_x)
    if dim_z > axes:
        raise ValueError("dim_z must be <= number of axes, or equal to dim_x")

    H = np.zeros((dim_z, dim_x))
    H[np.arange(dim_z), np.arange(dim_z) * state_dim_per_axis] = 1.0
    return H


def _block_diag_axis(axis_matrix: np.ndarray, axes: int) -> np.ndarray:
    return np.kron(np.eye(axes), axis_matrix)


def _cv_axis_count(dim_x: int) -> int:
    return _axis_count(dim_x, CV_STATE_DIM_PER_AXIS)


def constant_velocity_measurement_matrix(
    dim_x: int, dim_z: int | None = None
) -> np.ndarray:
    """Return a measurement matrix for constant-velocity states.

    State order is ``[x, vx, y, vy, ...]``. By default, the matrix measures
    position only for every spatial axis. Passing ``dim_z == dim_x`` returns
    an identity matrix for full-state measurements.
    """
    return _position_measurement_matrix(dim_x, CV_STATE_DIM_PER_AXIS, dim_z)


def constant_velocity_transition_matrix(dim_x: int, dt: float) -> np.ndarray:
    """Return the transition matrix for a constant-velocity model.

    Supports any state dimension that is a multiple of 2.
    """
    axes = _cv_axis_count(dim_x)
    F_axis = np.array([[1, dt], [0, 1]], dtype=float)
    return _block_diag_axis(F_axis, axes)


def constant_velocity_process_noise_input(dim_x: int, dt: float) -> np.ndarray:
    """Return the process-noise input matrix for constant-velocity states.

    The process noise is one acceleration term per spatial axis.
    """
    axes = _cv_axis_count(dim_x)
    L_axis = np.array([[0.5 * (dt**2)], [dt]], dtype=float)
    return _block_diag_axis(L_axis, axes)


def constant_velocity_measurement_noise_matrix(dim_z: int) -> np.ndarray:
    """Return an additive measurement-noise matrix."""
    if dim_z < 1:
        raise ValueError("dim_z must be at least 1")
    return np.eye(dim_z)


def constant_velocity_transition_function(x, dt: float) -> np.ndarray:
    """Apply a constant-velocity transition to state ``x``."""
    x = _as_state_vector(x)
    return constant_velocity_transition_matrix(len(x), dt) @ x


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
    kf_cv.F = constant_velocity_transition_matrix(dim_x, dt)
    kf_cv.H = constant_velocity_measurement_matrix(dim_x, dim_z)
    kf_cv.P = _as_covariance(P, dim_x)
    kf_cv.R = _as_covariance(R, dim_z)
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
    kf_cv.P = _as_covariance(P, 1)
    kf_cv.R = _as_covariance(R, 1)
    kf_cv.Q = _as_covariance(Q, 1)
    return kf_cv


def constant_velocity_filter_2d(P, R, Q=0, dt=1, x=(0, 0)):
    """Backward-compatible constant-velocity filter factory.

    Historically this function initialized a 1-axis ``[x, vx]`` filter. It now
    also works for any even-length state while keeping the original default.
    """
    return constant_velocity_filter(P=P, R=R, Q=Q, dt=dt, x=x, dim_z=1)


class ConstantVelocity(LinearStateSpaceModel):
    """Linear constant-velocity state-space model.

    State order is ``[x, vx, y, vy, ...]``. The historical default remains the
    2D model with state ``[x, vx, y, vy]`` and measurement ``[x, y]``.
    """

    NDIM = {"x": 4, "z": 2, "u": 0, "w": 2, "v": 2}

    def __init__(self, dt=0.1, axes: int = 2, dim_z: int | None = None):
        if axes < 1:
            raise ValueError("axes must be at least 1")

        dim_x = axes * CV_STATE_DIM_PER_AXIS
        dim_z = axes if dim_z is None else dim_z

        self.dt = dt
        self.axes = axes
        self.dim_z = dim_z
        self.ndim = {"x": dim_x, "z": dim_z, "u": 0, "w": axes, "v": dim_z}
        self.F = constant_velocity_transition_matrix(dim_x, dt)
        self.L = constant_velocity_process_noise_input(dim_x, dt)
        self.H = constant_velocity_measurement_matrix(dim_x, dim_z)
        self.M = constant_velocity_measurement_noise_matrix(dim_z)

    def state_equation(self, t, x, u=0, w=None):
        """Return ``x[t+1] = F[t] * x[t] + L[t] * w[t]``."""
        if w is None:
            w = np.zeros(self.ndim["w"])
        return self.F @ x + self.L @ w

    def observation_equation(self, t, x, v=None):
        """Return ``z[t] = H[t] * x[t] + M[t] * v[t]``."""
        if v is None:
            v = np.zeros(self.ndim["v"])
        return self.H @ x + self.M @ v

    def transition_matrix(self, t):
        """Return the transition matrix at time ``t``."""
        return self.F

    def control_input_matrix(self, t):
        """Return the control-input matrix at time ``t``."""
        return np.zeros((self.ndim["x"], self.ndim["u"]))

    def process_noise_input(self, t):
        """Return the process-noise input matrix at time ``t``."""
        return self.L

    def measurement_matrix(self, t):
        """Return the measurement matrix at time ``t``."""
        return self.H

    def measurement_noise_matrix(self, t):
        """Return the measurement-noise matrix at time ``t``."""
        return self.M
