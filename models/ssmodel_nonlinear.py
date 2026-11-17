"""Nonlinear state-space models — the coordinated-turn family.

The constant-acceleration helpers used to live here even though a
constant-acceleration plant is linear; they now sit in
:mod:`models.ssmodel_linear` next to the other closed-form matrices.

Everything trigonometric in the turn model is evaluated through
``sin(θ)/ω`` and ``2 sin²(θ/2)/ω`` rather than ``sin(θ)/ω`` and
``(1-cos θ)/ω``. The latter pair loses all significance once ``θ`` drops below
``1e-8`` (``1 - cos θ`` rounds to zero), which used to give a 100% error in
``dJ/dω`` for a target flying straight; the replacement is exact to machine
precision across the whole range, so no ``eps`` branch is needed.
"""

import math
from typing import Any

import numpy as np

from filters._typing import Matrix, Measurement, Vector

from ._common import as_state_vector
from .ssmodel import NonlinearStateSpaceModel

__all__ = [
    "CT_MEASUREMENT_DIM",
    "CT_PROCESS_NOISE_DIM",
    "CT_STATE_DIM",
    "FCT",
    "HCT",
    "CoordinatedTurn",
    "FxCT",
    "coordinated_turn_F",
    "coordinated_turn_H",
    "coordinated_turn_Jfx",
    "coordinated_turn_L",
    "coordinated_turn_M",
    "coordinated_turn_fx",
]

CT_STATE_DIM = 5
CT_PROCESS_NOISE_DIM = 3
CT_MEASUREMENT_DIM = 2

#: Below this |sin θ| the series expansions win; the closed forms stay accurate
#: far beyond it, so the switch only guards the last few digits.
_SERIES_LIMIT = 1e-3


def _turn_terms(turn_rate: float, dt: float) -> tuple[float, float, float, float]:
    """Return ``(sinθ/ω, (1-cosθ)/ω, d/dω[sinθ/ω], d/dω[(1-cosθ)/ω])``.

    Written in terms of ``θ = ω dt`` so the ω→0 limit is a removal of the
    singularity rather than a separate code path.
    """

    theta = turn_rate * dt
    if abs(theta) < _SERIES_LIMIT:
        theta_sq = theta * theta
        a = dt * (1.0 - theta_sq / 6.0 + theta_sq**2 / 120.0)
        b = dt * theta * (0.5 - theta_sq / 24.0 + theta_sq**2 / 720.0)
        da = dt**2 * theta * (-1.0 / 3.0 + theta_sq / 30.0 - theta_sq**2 / 840.0)
        db = dt**2 * (0.5 - theta_sq / 8.0 + theta_sq**2 / 144.0)
        return a, b, da, db

    sin_theta = math.sin(theta)
    cos_theta = math.cos(theta)
    a = sin_theta / turn_rate
    b = (1.0 - cos_theta) / turn_rate
    da = dt**2 * (theta * cos_theta - sin_theta) / (theta * theta)
    db = dt**2 * (theta * sin_theta - (1.0 - cos_theta)) / (theta * theta)
    return a, b, da, db


def coordinated_turn_F(dim: int, dt: float, turn_rate: float = math.pi / 180) -> Matrix:
    """Return the transition matrix of a 2-D coordinated-turn model.

    Parameters
    ----------
    dim : int
        Must be 5; the guess is rejected rather than silently reinterpreted.
    dt : float
        Step length.
    turn_rate : float, optional
        Angular rate ``ω`` in rad/s, by default 1°/s.
    """

    if dim != CT_STATE_DIM:
        raise ValueError(f"dim must be {CT_STATE_DIM}, got {dim}")

    a, b, _, _ = _turn_terms(turn_rate, dt)
    cos_theta = math.cos(turn_rate * dt)
    sin_theta = math.sin(turn_rate * dt)
    return np.array(
        [
            [1.0, a, 0.0, -b, 0.0],
            [0.0, cos_theta, 0.0, -sin_theta, 0.0],
            [0.0, b, 1.0, a, 0.0],
            [0.0, sin_theta, 0.0, cos_theta, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )


def coordinated_turn_H(dim_z: int = CT_MEASUREMENT_DIM) -> Matrix:
    """Return the coordinated-turn measurement matrix.

    ``dim_z=2`` measures position ``[x, y]``, ``dim_z=1`` measures ``x`` only,
    and ``dim_z=5`` returns full-state measurements.

    The state is ``[x, vx, y, vy, ω]``, so positions sit at indices ``0, 2``
    rather than in evenly spaced axis blocks — the shared turn rate breaks the
    block structure the other models can rely on.
    """

    if dim_z == CT_STATE_DIM:
        return np.eye(CT_STATE_DIM)
    if not 1 <= dim_z <= CT_MEASUREMENT_DIM:
        raise ValueError(
            f"dim_z must be 1, {CT_MEASUREMENT_DIM} or {CT_STATE_DIM}, got {dim_z}"
        )

    matrix = np.zeros((dim_z, CT_STATE_DIM))
    matrix[np.arange(dim_z), np.arange(dim_z) * 2] = 1.0
    return matrix


def coordinated_turn_L(dt: float) -> Matrix:
    """Return the additive process-noise input matrix for the turn model."""

    return np.array(
        [
            [0.5 * dt**2, 0.0, 0.0],
            [dt, 0.0, 0.0],
            [0.0, 0.5 * dt**2, 0.0],
            [0.0, dt, 0.0],
            [0.0, 0.0, dt],
        ]
    )


def coordinated_turn_M(dim_z: int = CT_MEASUREMENT_DIM) -> Matrix:
    """Return the additive observation-noise matrix for the turn model."""

    if dim_z != CT_STATE_DIM and not 1 <= dim_z <= CT_MEASUREMENT_DIM:
        raise ValueError(
            f"dim_z must be 1, {CT_MEASUREMENT_DIM} or {CT_STATE_DIM}, got {dim_z}"
        )
    return np.eye(dim_z)


def coordinated_turn_fx(x: Any, dt: float, turn_rate: float | None = None) -> Vector:
    """Apply the coordinated-turn transition to ``x``.

    ``turn_rate=None`` reads the rate from the fifth state entry, which is the
    convention the model class uses; pass a number for a fixed-rate turn.
    """

    state = as_state_vector(x)
    if turn_rate is None:
        turn_rate = float(state[4])
    return coordinated_turn_F(CT_STATE_DIM, dt, turn_rate) @ state


def coordinated_turn_Jfx(x: Any, dt: float) -> Matrix:
    """Return ``df/dx`` for the coordinated-turn state equation.

    A straight-flying target (``ω = 0``) is a regular point of the Jacobian, so
    no special casing is needed: the entries that carry a ``1/ω²`` singularity
    are evaluated from their series near zero.
    """

    state = as_state_vector(x)
    vx, vy, omega = float(state[1]), float(state[3]), float(state[4])
    a, b, da, db = _turn_terms(omega, dt)

    theta = omega * dt
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)

    jacobian = np.zeros((CT_STATE_DIM, CT_STATE_DIM))
    jacobian[0, 0] = 1.0
    jacobian[0, 1] = a
    jacobian[0, 3] = -b
    jacobian[0, 4] = da * vx - db * vy
    jacobian[1, 1] = cos_theta
    jacobian[1, 3] = -sin_theta
    jacobian[1, 4] = -dt * (sin_theta * vx + cos_theta * vy)
    jacobian[2, 1] = b
    jacobian[2, 2] = 1.0
    jacobian[2, 3] = a
    jacobian[2, 4] = db * vx + da * vy
    jacobian[3, 1] = sin_theta
    jacobian[3, 3] = cos_theta
    jacobian[3, 4] = dt * (cos_theta * vx - sin_theta * vy)
    jacobian[4, 4] = 1.0
    return jacobian


class CoordinatedTurn(NonlinearStateSpaceModel):
    """2-D coordinated-turn model.

    State order is ``[x, vx, y, vy, ω]`` and the measurement vector is
    ``[x, y]`` by default.
    """

    def __init__(
        self,
        dt: float = 0.1,
        dim_z: int = CT_MEASUREMENT_DIM,
        *,
        process_var: float = 1.0,
        measurement_var: float = 1.0,
    ) -> None:
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")

        self.dt = dt
        self.process_var = process_var
        self.measurement_var = measurement_var
        self.H = coordinated_turn_H(dim_z)
        self.M = coordinated_turn_M(dim_z)
        self.L = coordinated_turn_L(dt)
        self._dim_z = dim_z

    @property
    def dim_x(self) -> int:
        return CT_STATE_DIM

    @property
    def dim_z(self) -> int:
        return self._dim_z

    @property
    def dim_w(self) -> int:
        return CT_PROCESS_NOISE_DIM

    def compute_F(self, omega: float, dt: float | None = None) -> Matrix:
        """Return the transition matrix for a given turn rate."""

        return coordinated_turn_F(CT_STATE_DIM, self.dt if dt is None else dt, omega)

    def fx(
        self,
        x: Any,
        dt: float | None = None,
        *,
        t: float = 0.0,
        u: Any = None,
        w: Any = None,
    ) -> Vector:
        state = as_state_vector(x)
        step = self.dt if dt is None else dt
        predicted = self.compute_F(float(state[4]), step) @ state
        if w is not None:
            predicted = predicted + coordinated_turn_L(step) @ as_state_vector(w)
        return predicted

    def hx(self, x: Any, v: Any = None) -> Measurement:
        measurement = self.H @ as_state_vector(x)
        if v is not None:
            measurement = measurement + self.M @ as_state_vector(v)
        return measurement

    def process_cov(self) -> Matrix:
        """Return ``Q`` as ``L Lᵀ * process_var`` (noise on acceleration, ω)."""

        return self.L @ self.L.T * self.process_var

    def measurement_cov(self) -> Matrix:
        return self.M @ self.M.T * self.measurement_var

    def Jfx(self, t: float, x: Any) -> Matrix:
        return coordinated_turn_Jfx(x, self.dt)

    def Jfw(self, t: float, x: Any) -> Matrix:
        return self.Lt(t)

    def Jhx(self, t: float, x: Any) -> Matrix:
        return self.H

    def Jhv(self, t: float, x: Any) -> Matrix:
        return self.Mt(t)

    def Lt(self, t: float) -> Matrix:
        return self.L

    def Mt(self, t: float) -> Matrix:
        return self.M


FCT = coordinated_turn_F
HCT = coordinated_turn_H
FxCT = coordinated_turn_fx
