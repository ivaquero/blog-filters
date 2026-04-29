from __future__ import annotations

import math

import numpy as np

from filters.kalman import KalmanFilter

from .noise import white_noise_discrete
from .ssmodel import StateSpaceModel


class NonlinearStateSpaceModel(StateSpaceModel):
    """Non-Linear time varying system.

    State space model of the plant to be estimated.
    x[t+1] = f(t, x[t], u[t], w[t])
    z[t] = h(t, x[t], v[t])

    f: state equation
    h: observation equation
    x: state
    z: output
    u: control input
    w: system noise
    v: observation noise
    """

    def Jfx(self, t, x):
        """The Jacobian of the system model.

        x[t+1] = f(x[t], u[t], w[t], t),
        return (df/dx)(x).
        """
        raise NotImplementedError

    def Jfw(self, t, x):
        """The Jacobian of the state equation.

        x[t+1] = f(x[t], u[t], w[t], t),
        return (df/dw)(x).
        """
        raise NotImplementedError

    def Jhx(self, t, x):
        """The Jacobian of the observation equation.

        z[t] = h(x[t], v[t], t),
        return (dh/dx)(x).
        """
        raise NotImplementedError

    def Jhv(self, t, x):
        """The Jacobian of the observation equation.

        z[t] = h(x[t], v[t], t),
        return (dh/dv)(x).
        """
        raise NotImplementedError

    def Lt(self, t):
        """x[t+1] = f(x[t], u[t], t) + L[t] * w[t]

        In case of system noise is additive, return L[t].
        """
        raise NotImplementedError

    def Mt(self, t):
        """z[t] = h(x[t], t) + M[t] * v[t]

        In case of observatoin noise is additive, return M[t].
        """
        raise NotImplementedError


CA_STATE_DIM_PER_AXIS = 3
CT_STATE_DIM = 5
CT_PROCESS_NOISE_DIM = 3
CT_MEASUREMENT_DIM = 2


def _axis_count(dim_x: int, state_dim_per_axis: int) -> int:
    if dim_x < state_dim_per_axis or dim_x % state_dim_per_axis != 0:
        raise ValueError(f"dim_x must be a positive multiple of {state_dim_per_axis}")
    return dim_x // state_dim_per_axis


def _as_state_vector(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def _as_covariance(value, dim: int) -> np.ndarray:
    return np.eye(dim) * value if np.isscalar(value) else np.atleast_2d(value)


def _position_measurement_H(
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


def _ca_axis_count(dim_x: int) -> int:
    return _axis_count(dim_x, CA_STATE_DIM_PER_AXIS)


def constant_acceleration_H(dim_x: int, dim_z: int | None = None) -> np.ndarray:
    """Return a measurement matrix for constant-acceleration states.

    State order is ``[x, vx, ax, y, vy, ay, ...]``. By default, the matrix
    measures position only for every spatial axis. Passing ``dim_z == dim_x``
    returns an identity matrix for full-state measurements.
    """
    return _position_measurement_H(dim_x, CA_STATE_DIM_PER_AXIS, dim_z)


def constant_acceleration_F(dim: int, dt: float) -> np.ndarray:
    """Return the transition matrix for a constant-acceleration model.

    Supports any state dimension that is a multiple of 3.
    """
    axes = _ca_axis_count(dim)
    F_axis = np.array([[1, dt, 0.5 * dt**2], [0, 1, dt], [0, 0, 1]], dtype=float)
    return _block_diag_axis(F_axis, axes)


def constant_acceleration_fx(x, dt: float) -> np.ndarray:
    """Apply a constant-acceleration transition to state ``x``."""
    x = _as_state_vector(x)
    return constant_acceleration_F(len(x), dt) @ x


def constant_acceleration_filter(
    P, R, Q=0, dt: float = 1, x=(0, 0, 0), dim_z: int | None = None
):
    """Create a Kalman filter configured for constant acceleration.

    The state is ordered per axis as ``[position, velocity, acceleration]``.
    For example, a 2D state is ``[x, vx, ax, y, vy, ay]``.
    """
    x = _as_state_vector(x)
    dim_x = len(x)
    axes = _ca_axis_count(dim_x)
    dim_z = axes if dim_z is None else dim_z

    kf_ca = KalmanFilter(dim_x=dim_x, dim_z=dim_z)
    kf_ca.x = x
    kf_ca.F = constant_acceleration_F(dim_x, dt)
    kf_ca.H = constant_acceleration_H(dim_x, dim_z)
    kf_ca.P = _as_covariance(P, dim_x)
    kf_ca.R = _as_covariance(R, dim_z)
    kf_ca.Q = (
        white_noise_discrete(dim=CA_STATE_DIM_PER_AXIS, dt=dt, var=Q, block_size=axes)
        if np.isscalar(Q)
        else np.atleast_2d(Q)
    )
    return kf_ca


def constant_acceleration_filter_3d(P, R, Q=0, dt=1, x=(0, 0, 0)):
    """Backward-compatible 1-axis constant-acceleration filter factory."""
    return constant_acceleration_filter(P=P, R=R, Q=Q, dt=dt, x=x, dim_z=1)


HCA = constant_acceleration_H
FCA = constant_acceleration_F
FxCA = constant_acceleration_fx
KFCA = constant_acceleration_filter
KFCA3d = constant_acceleration_filter_3d


def coordinated_turn_F(
    dim: int, dt: float, turn_rate: float = math.pi / 180, eps: float = 1e-14
) -> np.ndarray:
    """Return the transition matrix for a 2D coordinated-turn model."""
    if dim != CT_STATE_DIM:
        raise ValueError("dim must be 5")

    if abs(turn_rate) < eps:
        return np.array(
            [
                [1.0, dt, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, dt, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        )

    theta = turn_rate * dt
    sin_theta = math.sin(theta)
    cos_theta = math.cos(theta)
    return np.array(
        [
            [1.0, sin_theta / turn_rate, 0.0, -(1 - cos_theta) / turn_rate, 0.0],
            [0.0, cos_theta, 0.0, -sin_theta, 0.0],
            [0.0, (1 - cos_theta) / turn_rate, 1.0, sin_theta / turn_rate, 0.0],
            [0.0, sin_theta, 0.0, cos_theta, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )


def coordinated_turn_H(dim_z: int = CT_MEASUREMENT_DIM) -> np.ndarray:
    """Return the coordinated-turn measurement matrix.

    ``dim_z=2`` measures position ``[x, y]``. ``dim_z=1`` measures ``x`` only,
    and ``dim_z=5`` returns full-state measurements.
    """
    if dim_z == CT_STATE_DIM:
        return np.eye(CT_STATE_DIM)
    if dim_z < 1 or dim_z > CT_MEASUREMENT_DIM:
        raise ValueError("dim_z must be 1, 2, or 5")

    H = np.zeros((dim_z, CT_STATE_DIM))
    H[np.arange(dim_z), np.arange(dim_z) * 2] = 1.0
    return H


def coordinated_turn_L(dt: float) -> np.ndarray:
    """Return the additive process-noise input matrix for the turn model."""
    return np.array(
        [
            [0.5 * (dt**2), 0.0, 0.0],
            [dt, 0.0, 0.0],
            [0.0, 0.5 * (dt**2), 0.0],
            [0.0, dt, 0.0],
            [0.0, 0.0, dt],
        ]
    )


def coordinated_turn_M(dim_z: int = CT_MEASUREMENT_DIM) -> np.ndarray:
    """Return the additive observation-noise matrix for the turn model."""
    if dim_z != CT_STATE_DIM and (dim_z < 1 or dim_z > CT_MEASUREMENT_DIM):
        raise ValueError("dim_z must be 1, 2, or 5")
    return np.eye(dim_z)


def coordinated_turn_fx(x, dt: float, eps: float = 1e-14) -> np.ndarray:
    """Apply a coordinated-turn transition to state ``x``."""
    x = _as_state_vector(x)
    return coordinated_turn_F(len(x), dt, turn_rate=x[4], eps=eps) @ x


def _coordinated_turn_zero_jacobian(dt: float, vx: float, vy: float) -> np.ndarray:
    return np.array(
        [
            [1.0, dt, 0.0, 0.0, -0.5 * (dt**2) * vy],
            [0.0, 1.0, 0.0, 0.0, -dt * vy],
            [0.0, 0.0, 1.0, dt, 0.5 * (dt**2) * vx],
            [0.0, 0.0, 0.0, 1.0, dt * vx],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )


def coordinated_turn_Jfx(x, dt: float, eps: float = 1e-14) -> np.ndarray:
    """Return ``df/dx`` for the coordinated-turn state equation."""
    x = _as_state_vector(x)
    vx, vy, omega = x[1], x[3], x[4]

    if abs(omega) < eps:
        return _coordinated_turn_zero_jacobian(dt, vx, vy)

    theta = omega * dt
    sin_theta = math.sin(theta)
    cos_theta = math.cos(theta)
    omega_sq = omega**2

    J = np.zeros((CT_STATE_DIM, CT_STATE_DIM))
    J[0, 0] = 1
    J[0, 1] = sin_theta / omega
    J[0, 3] = -(1 - cos_theta) / omega
    J[0, 4] = (
        cos_theta * dt * vx / omega
        - sin_theta * vx / omega_sq
        - sin_theta * dt * vy / omega
        - (-1 + cos_theta) * vy / omega_sq
    )
    J[1, 1] = cos_theta
    J[1, 3] = -sin_theta
    J[1, 4] = -sin_theta * dt * vx - cos_theta * dt * vy
    J[2, 1] = (1 - cos_theta) / omega
    J[2, 2] = 1
    J[2, 3] = sin_theta / omega
    J[2, 4] = (
        sin_theta * dt * vx / omega
        - (1 - cos_theta) * vx / omega_sq
        + cos_theta * dt * vy / omega
        - sin_theta * vy / omega_sq
    )
    J[3, 1] = sin_theta
    J[3, 3] = cos_theta
    J[3, 4] = cos_theta * dt * vx - sin_theta * dt * vy
    J[4, 4] = 1
    return J


class CoordinatedTurn(NonlinearStateSpaceModel):
    """2D coordinated-turn state-space model.

    State order is ``[x, vx, y, vy, omega]`` and measurement order is
    ``[x, y]``.
    """

    NDIM = {
        "x": CT_STATE_DIM,
        "z": CT_MEASUREMENT_DIM,
        "u": 0,
        "w": CT_PROCESS_NOISE_DIM,
        "v": CT_MEASUREMENT_DIM,
    }

    def __init__(self, dt=0.1, eps=1e-14, dim_z: int = CT_MEASUREMENT_DIM):
        self.dt = dt
        self.eps = eps
        self.dim_z = dim_z
        self.ndim = {**self.NDIM, "z": dim_z, "v": dim_z}
        self.H = coordinated_turn_H(dim_z)
        self.M = coordinated_turn_M(dim_z)

    def compute_F(self, omega, dt):
        return coordinated_turn_F(CT_STATE_DIM, dt, turn_rate=omega, eps=self.eps)

    def state_equation(self, t, x, u=0, w=None):
        """Return ``x[t+1] = f(t, x[t], u[t], w[t])``."""
        if w is None:
            w = np.zeros(self.ndim["w"])
        return self.compute_F(x[4], self.dt) @ x + self.Lt(t) @ w

    def observation_equation(self, t, x, v=None):
        """Return ``z[t] = h(t, x[t], v[t])``."""
        if v is None:
            v = np.zeros(self.ndim["v"])
        return self.H @ x + self.M @ v

    def Jfx(self, t, x):
        """Return ``df/dx`` for the coordinated-turn state equation."""
        return coordinated_turn_Jfx(x, self.dt, eps=self.eps)

    def Jfw(self, t, x):
        """Return ``df/dw`` for additive system noise."""
        return self.Lt(t)

    def Jhx(self, t, x):
        """Return ``dh/dx`` for position-only measurements."""
        return self.H

    def Jhv(self, t, x):
        """Return ``dh/dv`` for additive observation noise."""
        return self.Mt(t)

    def Lt(self, t):
        """Return additive system-noise input matrix ``L[t]``."""
        return coordinated_turn_L(self.dt)

    def Mt(self, t):
        """Return additive observation-noise matrix ``M[t]``."""
        return self.M


FCT = coordinated_turn_F
HCT = coordinated_turn_H
FxCT = coordinated_turn_fx
