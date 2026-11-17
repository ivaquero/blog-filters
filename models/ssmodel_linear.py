"""Linear (possibly time-varying) Gaussian state-space models.

Two layers live here:

``LinearStateSpaceModel``
    Abstract machinery. A subclass only supplies the five matrix accessors
    (``F``, ``G``, ``L``, ``H``, ``M``) and inherits ``fx``/``hx``, both
    Jacobians, and the noise covariances.
``LinearModel``
    Concrete model configured straight from matrices, so an arbitrary linear
    plant needs no new class::

        model = LinearModel(
            F=[[1, dt], [0, 1]], H=[[1, 0]], L=[[dt**2 / 2], [dt]]
        )

``ConstantVelocity`` and ``ConstantAcceleration`` are the two parameterised
special cases the notebooks use; both are plain instances of the machinery
above rather than hand-written duplicates of it.
"""

from typing import TYPE_CHECKING, Any

import numpy as np

from shared import Matrix, Measurement, Vector

from ._common import (
    as_covariance,
    as_state_batch,
    as_state_vector,
    axis_count,
    block_diag_axis,
    position_measurement_matrix,
)
from .noise import white_noise_discrete
from .ssmodel import StateSpaceModel

if TYPE_CHECKING:
    from filters.kalman import KalmanFilter

__all__ = [
    "CA_STATE_DIM_PER_AXIS",
    "CV_STATE_DIM_PER_AXIS",
    "ConstantAcceleration",
    "ConstantVelocity",
    "LinearModel",
    "LinearStateSpaceModel",
    "constant_acceleration_F",
    "constant_acceleration_H",
    "constant_acceleration_filter",
    "constant_acceleration_filter_3d",
    "constant_acceleration_fx",
    "constant_velocity_filter",
    "constant_velocity_filter_1d",
    "constant_velocity_filter_2d",
    "constant_velocity_measurement_matrix",
    "constant_velocity_measurement_noise_matrix",
    "constant_velocity_process_noise_input",
    "constant_velocity_transition_function",
    "constant_velocity_transition_matrix",
]

CV_STATE_DIM_PER_AXIS = 2
CA_STATE_DIM_PER_AXIS = 3


def _as_matrix(value: Any, name: str) -> Matrix:
    matrix = np.asarray(value, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a 2-D matrix, got shape {matrix.shape}")
    return matrix


class LinearStateSpaceModel(StateSpaceModel):
    """Linear time-varying plant with additive Gaussian noise.

    ``x[t+1] = F[t] x[t] + G[t] u[t] + L[t] w[t]``
    ``z[t]   = H[t] x[t] + M[t] v[t]``

    The five accessors are the only required surface. Everything the estimators
    consume — ``fx``, ``hx``, the Jacobians, ``Q``, ``R`` — is derived from them,
    so a linear model can no longer disagree between its "equation" methods and
    its matrices.
    """

    def transition_matrix(self, t: float) -> Matrix:
        """Return ``F[t]``."""

        raise NotImplementedError

    def control_input_matrix(self, t: float) -> Matrix:
        """Return ``G[t]``."""

        raise NotImplementedError

    def process_noise_input(self, t: float) -> Matrix:
        """Return ``L[t]``."""

        raise NotImplementedError

    def measurement_matrix(self, t: float) -> Matrix:
        """Return ``H[t]``."""

        raise NotImplementedError

    def measurement_noise_matrix(self, t: float) -> Matrix:
        """Return ``M[t]``."""

        raise NotImplementedError

    def fx(
        self,
        x: Any,
        dt: float | None = None,
        *,
        t: float = 0.0,
        u: Any = None,
        w: Any = None,
    ) -> Vector:
        state = self.transition_matrix(t) @ as_state_vector(x)
        if u is not None:
            state = state + self.control_input_matrix(t) @ as_state_vector(u)
        if w is not None:
            state = state + self.process_noise_input(t) @ as_state_vector(w)
        return state

    def hx(self, x: Any, v: Any = None) -> Measurement:
        measurement = self.measurement_matrix(0.0) @ as_state_vector(x)
        if v is not None:
            measurement = measurement + self.measurement_noise_matrix(0.0) @ (
                as_state_vector(v)
            )
        return measurement

    def fx_many(
        self,
        x: Any,
        dt: float | None = None,
        *,
        t: float = 0.0,
        u: Any = None,
        w: Any = None,
    ) -> Matrix:
        """Propagate a batch in one matrix product (no per-particle Python loop).

        Control and noise terms, when present, are broadcast from ``u``/``w``
        across the batch by NumPy, preserving ``fx``'s semantics exactly.
        """

        batch = as_state_batch(x, self.dim_x)
        propagated = batch @ self.transition_matrix(t).T
        if u is not None:
            propagated = propagated + as_state_batch(u, self.dim_u) @ (
                self.control_input_matrix(t).T
            )
        if w is not None:
            propagated = propagated + as_state_batch(w, self.dim_w) @ (
                self.process_noise_input(t).T
            )
        return propagated

    def hx_many(self, x: Any, v: Any = None) -> Matrix:
        batch = as_state_batch(x, self.dim_x)
        measurements = batch @ self.measurement_matrix(0.0).T
        if v is not None:
            measurements = measurements + as_state_batch(v, self.dim_v) @ (
                self.measurement_noise_matrix(0.0).T
            )
        return measurements

    def jacobian_x(self, x: Any) -> Matrix:
        return self.transition_matrix(0.0)

    def jacobian_h(self, x: Any) -> Matrix:
        return self.measurement_matrix(0.0)

    def process_cov(self) -> Matrix:
        """Return ``L Lᵀ`` — the covariance for unit-variance process noise."""

        noise_input = self.process_noise_input(0.0)
        return noise_input @ noise_input.T

    def measurement_cov(self) -> Matrix:
        """Return ``M Mᵀ`` — the covariance for unit-variance sensor noise."""

        noise_matrix = self.measurement_noise_matrix(0.0)
        return noise_matrix @ noise_matrix.T


class LinearModel(LinearStateSpaceModel):
    """Linear model built directly from matrices.

    Parameters
    ----------
    F : array_like
        ``(dim_x, dim_x)`` transition matrix.
    H : array_like
        ``(dim_z, dim_x)`` measurement matrix.
    G : array_like, optional
        ``(dim_x, dim_u)`` control-input matrix.
    L : array_like, optional
        ``(dim_x, dim_w)`` process-noise input matrix.
    M : array_like, optional
        ``(dim_z, dim_v)`` measurement-noise matrix; identity when omitted.
    Q, R : array_like, optional
        Full covariances. When omitted the model uses ``L Lᵀ`` and ``M Mᵀ``.
    dt : float, optional
        Nominal step length, exposed for the UKF constructor.
    """

    def __init__(
        self,
        F: Any,
        H: Any,
        *,
        G: Any = None,
        L: Any = None,
        M: Any = None,
        Q: Any = None,
        R: Any = None,
        dt: float = 1.0,
    ) -> None:
        transition = _as_matrix(F, "F")
        if transition.shape[0] != transition.shape[1]:
            raise ValueError(f"F must be square, got shape {transition.shape}")

        measurement = _as_matrix(H, "H")
        if measurement.shape[1] != transition.shape[0]:
            raise ValueError(
                f"H must have {transition.shape[0]} columns, got {measurement.shape[1]}"
            )

        self.dt = dt
        self.F = transition
        self.H = measurement
        self.G = None if G is None else _as_matrix(G, "G")
        self.L = None if L is None else _as_matrix(L, "L")
        self.M = np.eye(measurement.shape[0]) if M is None else _as_matrix(M, "M")
        self._Q = None if Q is None else as_covariance(Q, transition.shape[0], "Q")
        self._R = None if R is None else as_covariance(R, measurement.shape[0], "R")

        if self.G is not None and self.G.shape[0] != transition.shape[0]:
            raise ValueError(f"G must have {transition.shape[0]} rows")
        if self.L is not None and self.L.shape[0] != transition.shape[0]:
            raise ValueError(f"L must have {transition.shape[0]} rows")

    @property
    def dim_x(self) -> int:
        return self.F.shape[0]

    @property
    def dim_z(self) -> int:
        return self.H.shape[0]

    @property
    def dim_u(self) -> int:
        return 0 if self.G is None else self.G.shape[1]

    @property
    def dim_w(self) -> int:
        return 0 if self.L is None else self.L.shape[1]

    def transition_matrix(self, t: float) -> Matrix:
        return self.F

    def control_input_matrix(self, t: float) -> Matrix:
        if self.G is None:
            return np.zeros((self.dim_x, 0))
        return self.G

    def process_noise_input(self, t: float) -> Matrix:
        if self.L is None:
            return np.zeros((self.dim_x, 0))
        return self.L

    def measurement_matrix(self, t: float) -> Matrix:
        return self.H

    def measurement_noise_matrix(self, t: float) -> Matrix:
        return self.M

    def process_cov(self) -> Matrix:
        if self._Q is not None:
            return self._Q
        return super().process_cov()

    def measurement_cov(self) -> Matrix:
        if self._R is not None:
            return self._R
        return super().measurement_cov()


# ---------------------------------------------------------------------------
# Constant velocity
# ---------------------------------------------------------------------------


def constant_velocity_measurement_matrix(
    dim_x: int, dim_z: int | None = None
) -> Matrix:
    """Return a measurement matrix for constant-velocity states.

    State order is ``[x, vx, y, vy, ...]``. By default the matrix measures
    position on every axis; ``dim_z == dim_x`` returns the identity.
    """

    return position_measurement_matrix(dim_x, CV_STATE_DIM_PER_AXIS, dim_z)


def constant_velocity_transition_matrix(dim_x: int, dt: float) -> Matrix:
    """Return ``F`` for constant-velocity states (``dim_x`` a multiple of 2)."""

    axes = axis_count(dim_x, CV_STATE_DIM_PER_AXIS)
    return block_diag_axis(np.array([[1.0, dt], [0.0, 1.0]]), axes)


def constant_velocity_process_noise_input(dim_x: int, dt: float) -> Matrix:
    """Return ``L`` mapping one acceleration noise term per axis."""

    axes = axis_count(dim_x, CV_STATE_DIM_PER_AXIS)
    return block_diag_axis(np.array([[0.5 * dt**2], [dt]]), axes)


def constant_velocity_measurement_noise_matrix(dim_z: int) -> Matrix:
    """Return the unit measurement-noise matrix ``M = I``."""

    if dim_z < 1:
        raise ValueError(f"dim_z must be at least 1, got {dim_z}")
    return np.eye(dim_z)


def constant_velocity_transition_function(x: Any, dt: float) -> Vector | Matrix:
    """Apply the constant-velocity transition to ``x`` without building ``F``.

    Accepts a single state ``(dim_x,)`` or a batch ``(n, dim_x)``; the output
    has the same shape as the input. Equivalent to multiplying by
    ``constant_velocity_transition_matrix(dim_x, dt)`` but ~1000x faster per
    call, which matters when a particle filter evaluates it once per particle.
    """

    state = np.array(x, dtype=float, copy=True)
    if state.ndim == 1:
        state[0::2] += dt * state[1::2]
    elif state.ndim == 2:
        state[:, 0::2] += dt * state[:, 1::2]
    else:
        raise ValueError(f"state must be 1-D or 2-D, got shape {state.shape}")
    return state


class ConstantVelocity(LinearStateSpaceModel):
    """Linear constant-velocity model.

    State order is ``[x, vx, y, vy, ...]``; the measurement vector is the first
    state of each axis, so the default 2-axis model has ``dim_x=4, dim_z=2``.
    """

    def __init__(
        self,
        dt: float = 0.1,
        axes: int = 2,
        dim_z: int | None = None,
        *,
        process_var: float = 1.0,
        measurement_var: float = 1.0,
    ) -> None:
        if axes < 1:
            raise ValueError(f"axes must be at least 1, got {axes}")
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")

        dim_x = axes * CV_STATE_DIM_PER_AXIS
        dim_z = axes if dim_z is None else dim_z

        self.dt = dt
        self.axes = axes
        self.process_var = process_var
        self.measurement_var = measurement_var
        self.F = constant_velocity_transition_matrix(dim_x, dt)
        self.L = constant_velocity_process_noise_input(dim_x, dt)
        self.H = constant_velocity_measurement_matrix(dim_x, dim_z)
        self.M = constant_velocity_measurement_noise_matrix(dim_z)
        self._dim_z = dim_z

    @property
    def dim_x(self) -> int:
        return self.axes * CV_STATE_DIM_PER_AXIS

    @property
    def dim_z(self) -> int:
        return self._dim_z

    @property
    def dim_w(self) -> int:
        return self.axes

    def process_cov(self) -> Matrix:
        """Return ``Q`` as ``L Lᵀ * process_var``."""

        return self.L @ self.L.T * self.process_var

    def measurement_cov(self) -> Matrix:
        """Return ``R`` as ``M Mᵀ * measurement_var``."""

        return self.M @ self.M.T * self.measurement_var

    def transition_matrix(self, t: float) -> Matrix:
        return self.F

    def control_input_matrix(self, t: float) -> Matrix:
        return np.zeros((self.dim_x, 0))

    def process_noise_input(self, t: float) -> Matrix:
        return self.L

    def measurement_matrix(self, t: float) -> Matrix:
        return self.H

    def measurement_noise_matrix(self, t: float) -> Matrix:
        return self.M


def constant_velocity_filter(
    P: Any, R: Any, Q: Any = 0, dt: float = 1, x: Any = (0, 0), dim_z: int | None = None
) -> KalmanFilter:
    """Create a Kalman filter configured for constant velocity.

    The state is ordered per axis as ``[position, velocity]``; ``dim_x`` is
    inferred from ``x`` and ``dim_z`` defaults to one measurement per axis.
    """

    from filters.kalman import KalmanFilter  # deferred: keeps models filter-free

    state = as_state_vector(x)
    dim_x = state.size
    axes = axis_count(dim_x, CV_STATE_DIM_PER_AXIS)
    dim_z = axes if dim_z is None else dim_z

    kf = KalmanFilter(dim_x=dim_x, dim_z=dim_z)
    kf.x = state
    kf.F = constant_velocity_transition_matrix(dim_x, dt)
    kf.H = constant_velocity_measurement_matrix(dim_x, dim_z)
    kf.P = as_covariance(P, dim_x, "P")
    kf.R = as_covariance(R, dim_z, "R")
    kf.Q = (
        white_noise_discrete(dim=CV_STATE_DIM_PER_AXIS, dt=dt, var=Q, block_size=axes)
        if np.ndim(Q) == 0
        else as_covariance(Q, dim_x, "Q")
    )
    return kf


def constant_velocity_filter_1d(
    P: Any, R: Any, Q: Any = 0, dt: float = 1, x: Any = (0,)
) -> KalmanFilter:
    """Create a 1-D filter: constant-value model, or constant velocity.

    A one-element ``x`` keeps the historical 0th-order (constant value) model,
    where ``Q`` is the state's own random-walk variance. Any longer ``x`` falls
    through to :func:`constant_velocity_filter` with a single measured axis.
    """

    state = as_state_vector(x)
    if state.size != 1:
        return constant_velocity_filter(P=P, R=R, Q=Q, dt=dt, x=state, dim_z=1)

    from filters.kalman import KalmanFilter  # deferred: keeps models filter-free

    kf = KalmanFilter(dim_x=1, dim_z=1)
    kf.x = state
    kf.F = np.eye(1)
    kf.H = np.eye(1)
    kf.P = as_covariance(P, 1, "P")
    kf.R = as_covariance(R, 1, "R")
    kf.Q = as_covariance(Q, 1, "Q")
    return kf


def constant_velocity_filter_2d(
    P: Any, R: Any, Q: Any = 0, dt: float = 1, x: Any = (0, 0)
) -> KalmanFilter:
    """Create a constant-velocity filter for an unconstrained state length.

    ``dim_z`` follows the number of spatial axes encoded in ``x`` rather than
    being pinned to 1, so a four-element state gets both positions measured
    instead of silently tracking only ``x``.
    """

    return constant_velocity_filter(P=P, R=R, Q=Q, dt=dt, x=x)


# ---------------------------------------------------------------------------
# Constant acceleration
# ---------------------------------------------------------------------------


def constant_acceleration_H(dim_x: int, dim_z: int | None = None) -> Matrix:
    """Return a measurement matrix for constant-acceleration states.

    State order is ``[x, vx, ax, y, vy, ay, ...]``; position on every axis by
    default, identity when ``dim_z == dim_x``.
    """

    return position_measurement_matrix(dim_x, CA_STATE_DIM_PER_AXIS, dim_z)


def constant_acceleration_F(dim_x: int, dt: float) -> Matrix:
    """Return ``F`` for constant-acceleration states (multiple of 3)."""

    axes = axis_count(dim_x, CA_STATE_DIM_PER_AXIS)
    axis_block = np.array([[1.0, dt, 0.5 * dt**2], [0.0, 1.0, dt], [0.0, 0.0, 1.0]])
    return block_diag_axis(axis_block, axes)


def constant_acceleration_L(dim_x: int, dt: float) -> Matrix:
    """Return ``L`` mapping one jerk noise term per axis."""

    axes = axis_count(dim_x, CA_STATE_DIM_PER_AXIS)
    return block_diag_axis(np.array([[0.5 * dt**2], [dt], [1.0]]), axes)


def constant_acceleration_fx(x: Any, dt: float) -> Vector | Matrix:
    """Apply the constant-acceleration transition without building ``F``.

    Accepts a single state ``(dim_x,)`` or a batch ``(n, dim_x)``.
    """

    state = np.array(x, dtype=float, copy=True)
    if state.ndim == 1:
        state[0::3] += dt * state[1::3] + 0.5 * dt**2 * state[2::3]
        state[1::3] += dt * state[2::3]
    elif state.ndim == 2:
        state[:, 0::3] += dt * state[:, 1::3] + 0.5 * dt**2 * state[:, 2::3]
        state[:, 1::3] += dt * state[:, 2::3]
    else:
        raise ValueError(f"state must be 1-D or 2-D, got shape {state.shape}")
    return state


class ConstantAcceleration(LinearStateSpaceModel):
    """Linear constant-acceleration model.

    State order is ``[x, vx, ax, y, vy, ay, ...]``; position is measured on
    every axis by default.
    """

    def __init__(
        self,
        dt: float = 0.1,
        axes: int = 2,
        dim_z: int | None = None,
        *,
        process_var: float = 1.0,
        measurement_var: float = 1.0,
    ) -> None:
        if axes < 1:
            raise ValueError(f"axes must be at least 1, got {axes}")
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")

        dim_x = axes * CA_STATE_DIM_PER_AXIS
        dim_z = axes if dim_z is None else dim_z

        self.dt = dt
        self.axes = axes
        self.process_var = process_var
        self.measurement_var = measurement_var
        self.F = constant_acceleration_F(dim_x, dt)
        self.L = constant_acceleration_L(dim_x, dt)
        self.H = constant_acceleration_H(dim_x, dim_z)
        self.M = constant_velocity_measurement_noise_matrix(dim_z)
        self._dim_z = dim_z

    @property
    def dim_x(self) -> int:
        return self.axes * CA_STATE_DIM_PER_AXIS

    @property
    def dim_z(self) -> int:
        return self._dim_z

    @property
    def dim_w(self) -> int:
        return self.axes

    def process_cov(self) -> Matrix:
        return self.L @ self.L.T * self.process_var

    def measurement_cov(self) -> Matrix:
        return self.M @ self.M.T * self.measurement_var

    def transition_matrix(self, t: float) -> Matrix:
        return self.F

    def control_input_matrix(self, t: float) -> Matrix:
        return np.zeros((self.dim_x, 0))

    def process_noise_input(self, t: float) -> Matrix:
        return self.L

    def measurement_matrix(self, t: float) -> Matrix:
        return self.H

    def measurement_noise_matrix(self, t: float) -> Matrix:
        return self.M


def constant_acceleration_filter(
    P: Any,
    R: Any,
    Q: Any = 0,
    dt: float = 1,
    x: Any = (0, 0, 0),
    dim_z: int | None = None,
) -> KalmanFilter:
    """Create a Kalman filter configured for constant acceleration."""

    from filters.kalman import KalmanFilter  # deferred: keeps models filter-free

    state = as_state_vector(x)
    dim_x = state.size
    axes = axis_count(dim_x, CA_STATE_DIM_PER_AXIS)
    dim_z = axes if dim_z is None else dim_z

    kf = KalmanFilter(dim_x=dim_x, dim_z=dim_z)
    kf.x = state
    kf.F = constant_acceleration_F(dim_x, dt)
    kf.H = constant_acceleration_H(dim_x, dim_z)
    kf.P = as_covariance(P, dim_x, "P")
    kf.R = as_covariance(R, dim_z, "R")
    kf.Q = (
        white_noise_discrete(dim=CA_STATE_DIM_PER_AXIS, dt=dt, var=Q, block_size=axes)
        if np.ndim(Q) == 0
        else as_covariance(Q, dim_x, "Q")
    )
    return kf


def constant_acceleration_filter_3d(
    P: Any, R: Any, Q: Any = 0, dt: float = 1, x: Any = (0, 0, 0)
) -> KalmanFilter:
    """Create a single-axis constant-acceleration filter (1-D measurement)."""

    return constant_acceleration_filter(P=P, R=R, Q=Q, dt=dt, x=x, dim_z=1)


# Legacy one-letter aliases, kept for the pre-0.2 notebooks.
HCA = constant_acceleration_H
FCA = constant_acceleration_F
FxCA = constant_acceleration_fx
KFCA = constant_acceleration_filter
KFCA3d = constant_acceleration_filter_3d
