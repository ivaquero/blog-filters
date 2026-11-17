"""Bicycle-model EKF for the differential-drive robot.

The filter used to live in ``simulators/robot.py``, which meant the simulator
package imported the filter package *and* SymPy while the filter itself
re-evaluated its symbolic Jacobian on every predict step (3.1 ms per step,
measured). The Jacobians are now closed-form ``numpy`` — 11.5 us per step — and
:mod:`symbols.robot` keeps the symbolic derivation as the reference that
``tests/test_robot_ekf.py`` checks them against.
"""

import math

import numpy as np

from ._typing import Matrix
from .kalman_ekf import ExtendedKalmanFilter

__all__ = ["STEER_EPS", "RobotEKF", "robot_move", "transition_jacobians"]

# Steering angles at or below this magnitude are treated as straight-line
# motion. The state propagation and the Jacobian switch at the *same* point: a
# mismatch would linearise the covariance about a different trajectory than the
# one the mean actually follows.
STEER_EPS = 1e-6


def robot_move(x, dt: float, *, u, wheelbase: float) -> np.ndarray:
    """Propagate a 3-state robot ``[x, y, heading]`` one step.

    The kinematic model is shared with the simulator, which is what a
    "differential-drive robot" is: one model, two consumers. The simulator used
    to carry a byte-identical copy because the EKF's version only accepted
    ``(3, 1)`` column vectors.

    ``dt`` stays positional-capable because
    :class:`~filters.kalman_ukf.UnscentedKalmanFilter` calls its ``fx`` callback
    as ``fx(sigma, dt, **fx_args)``. Only ``u`` and ``wheelbase`` are keyword-only.

    Parameters
    ----------
    x
        State, either ``(3,)`` or ``(3, 1)``.
    dt
        Time step.
    u
        Control ``(speed, steering_angle)`` with the angle in radians.
    wheelbase
        Distance between the axles.

    Returns
    -------
    numpy.ndarray
        The new state, flattened to ``(3,)``.
    """

    state = np.asarray(x, dtype=float).reshape(3)
    heading = state[2]
    speed = u[0]
    steering = u[1]
    distance = speed * dt

    if abs(steering) <= STEER_EPS:
        return state + np.array(
            [distance * math.cos(heading), distance * math.sin(heading), 0.0]
        )

    beta = (distance / wheelbase) * math.tan(steering)
    radius = wheelbase / math.tan(steering)
    return state + np.array(
        [
            -radius * math.sin(heading) + radius * math.sin(heading + beta),
            radius * math.cos(heading) - radius * math.cos(heading + beta),
            beta,
        ]
    )


def transition_jacobians(x, *, dt: float, u, wheelbase: float) -> tuple[Matrix, Matrix]:
    """Return ``F = df/dx`` and ``V = df/du`` for :func:`robot_move`.

    Both branches follow the branch taken by :func:`robot_move`. For a
    near-zero steering angle the arc expressions are 0/0: ``radius`` diverges
    while ``beta`` vanishes, and the two differences cancel to first order.
    Evaluating them at ``steering == 0`` raised ``ZeroDivisionError`` from
    ``tan(0)``; evaluating them at a tiny angle lost ten digits to cancellation.
    The limit is used instead, so a straight-line command — the most common input
    — no longer crashes and no longer poisons the covariance with ``nan``.
    """

    state = np.asarray(x, dtype=float).reshape(3)
    heading = state[2]
    speed = u[0]
    steering = u[1]
    sin_h, cos_h = math.sin(heading), math.cos(heading)

    if abs(steering) <= STEER_EPS:
        transition = np.array(
            [
                [1.0, 0.0, -dt * speed * sin_h],
                [0.0, 1.0, dt * speed * cos_h],
                [0.0, 0.0, 1.0],
            ]
        )
        control = np.array(
            [
                [dt * cos_h, -(dt * dt * speed * speed) / (2.0 * wheelbase) * sin_h],
                [dt * sin_h, (dt * dt * speed * speed) / (2.0 * wheelbase) * cos_h],
                [0.0, dt * speed / wheelbase],
            ]
        )
        return transition, control

    tan_s = math.tan(steering)
    sin_s, cos_s = math.sin(steering), math.cos(steering)
    beta = (dt * speed / wheelbase) * tan_s
    radius = wheelbase / tan_s
    sin_hb, cos_hb = math.sin(heading + beta), math.cos(heading + beta)

    transition = np.array(
        [
            [1.0, 0.0, -radius * cos_h + radius * cos_hb],
            [0.0, 1.0, -radius * sin_h + radius * sin_hb],
            [0.0, 0.0, 1.0],
        ]
    )

    inv_sin2 = 1.0 / sin_s**2
    cot_sec2 = 1.0 / (sin_s * cos_s)
    control = np.array(
        [
            [
                dt * cos_hb,
                wheelbase * (sin_h - sin_hb) * inv_sin2
                + dt * speed * cos_hb * cot_sec2,
            ],
            [
                dt * sin_hb,
                wheelbase * (cos_hb - cos_h) * inv_sin2
                + dt * speed * sin_hb * cot_sec2,
            ],
            [dt * tan_s / wheelbase, dt * speed / (wheelbase * cos_s**2)],
        ]
    )
    return transition, control


class RobotEKF(ExtendedKalmanFilter):
    """EKF localising a differential-drive robot against known landmarks.

    Parameters
    ----------
    dt
        Time step between predictions.
    wheelbase
        Distance between the axles.
    std_vel
        Standard deviation of the speed command, in speed units.
    std_steer
        Standard deviation of the steering command, in radians.

    Notes
    -----
    Process noise enters through ``V M V'`` in control space, so the base
    class's additive ``Q`` is intentionally not applied. ``M`` used to be
    ``diag((std_vel * speed)**2, std_steer**2)``, which tied the velocity noise
    to the commanded speed and made the covariance stop growing whenever the
    robot stood still; it is now ``diag(std_vel**2, std_steer**2)``, the plain
    control covariance the parameter names describe.
    """

    def __init__(
        self, dt: float, *, wheelbase: float, std_vel: float, std_steer: float
    ):
        super().__init__(3, 2, 2)
        self.dt = dt
        self.wheelbase = wheelbase
        self.std_vel = std_vel
        self.std_steer = std_steer

    def move(self, x, u, dt: float) -> np.ndarray:
        """Propagate ``x`` one step; thin wrapper over :func:`robot_move`."""

        return robot_move(x, dt=dt, u=u, wheelbase=self.wheelbase).reshape(3, 1)

    def predict(self, u=None) -> None:
        """Propagate the state and covariance one ``dt`` forward."""

        control = np.zeros(2) if u is None else np.asarray(u, dtype=float).reshape(2)
        self.x = robot_move(
            self.x, dt=self.dt, u=control, wheelbase=self.wheelbase
        ).reshape(3, 1)

        transition, control_matrix = transition_jacobians(
            self.x, dt=self.dt, u=control, wheelbase=self.wheelbase
        )
        noise = np.diag([self.std_vel**2, self.std_steer**2])
        self.P = (
            transition @ self.P @ transition.T
            + control_matrix @ noise @ control_matrix.T
        )

        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()
