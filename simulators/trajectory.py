"""Projectile and ballistic simulators.

Both classes integrate analytically-in-``t`` motion. :class:`BallTrajectory2D`
used to push its state through ``filters.solvers.rk4`` with a velocity callable
that was linear in time — a Runge-Kutta step that is *exactly* equivalent to the
midpoint rule for a linear integrand, at eight Python calls per step. The closed
form below reproduces those numbers and drops a helper class, a solver import
and a mutable callable.
"""

import math

import numpy as np

from filters._typing import resolve_rng

from ._types import Tracks
from ._validation import require_count, require_nonnegative

__all__ = ["BallTrajectory2D", "BaseballPath", "drag_force"]

# Drag model for a baseball, from the standard aerodynamic fit. These were
# literals inside ``BaseballPath.drag_force``; naming them makes the model
# tunable and the notebook copy-paste obvious.
DRAG_B_0 = 0.0039
DRAG_B_1 = 0.0058
DRAG_SPEED_REF = 35.0
DRAG_SPEED_SCALE = 5.0

GRAVITY_BASEBALL = 9.81


def drag_force(speed: float) -> float:
    """Return the air-drag force per unit mass of a baseball, in SI units."""

    b_m = DRAG_B_0 + DRAG_B_1 / (
        1.0 + math.exp((speed - DRAG_SPEED_REF) / DRAG_SPEED_SCALE)
    )
    return b_m * speed


class BallTrajectory2D:
    """2-D projectile under gravity, optionally sensed with noise.

    Parameters
    ----------
    x0, y0
        Launch position.
    speed
        Launch speed.
    launch_angle_deg
        Launch angle above the horizontal, in degrees.
    gravity
        Gravitational acceleration (positive magnitude).
    meas_std
        ``(x, y)`` standard deviations of the position readings.
    rng
        Random generator; a fresh unseeded one is built when omitted.
    """

    def __init__(
        self,
        *,
        x0: float,
        y0: float,
        speed: float,
        launch_angle_deg: float = 0.0,
        gravity: float = 9.8,
        meas_std: tuple[float, float] = (0.0, 0.0),
        rng: np.random.Generator | None = None,
    ):
        require_nonnegative("speed", speed)
        require_nonnegative("gravity", gravity)
        _require_std_pair("meas_std", meas_std)

        theta = math.radians(launch_angle_deg)
        self.x = x0
        self.y = y0
        self.t = 0.0
        self.vx = math.cos(theta) * speed
        self.vy0 = math.sin(theta) * speed
        self.gravity = gravity
        self.meas_std = meas_std
        self.rng = resolve_rng(rng)

    def step(self, *, dt: float) -> tuple[float, float]:
        """Advance ``dt`` and return the noisy ``(x, y)`` reading.

        The vertical term is ``dt * (vy0 - g * (t + dt/2))`` — the midpoint
        rule, which is what the retired RK4 call reduced to for a velocity that
        is linear in ``t``.
        """

        self.x += self.vx * dt
        self.y += dt * (self.vy0 - self.gravity * (self.t + 0.5 * dt))
        self.t += dt
        return (
            self.x + self.rng.standard_normal() * self.meas_std[0],
            self.y + self.rng.standard_normal() * self.meas_std[1],
        )

    def simulate(self, *, num: int, dt: float) -> Tracks:
        """Run ``num`` steps; ``x`` and ``z`` are both ``(N, 2)``."""

        require_count(num)
        xs = np.empty((num, 2))
        zs = np.empty((num, 2))
        for i in range(num):
            zs[i] = self.step(dt=dt)
            xs[i] = (self.x, self.y)
        return Tracks(t=(np.arange(num) + 1.0) * dt, x=xs, z=zs)


class BaseballPath:
    """2-D baseball path with quadratic drag, integrated by semi-implicit Euler.

    Parameters
    ----------
    x0, y0
        Launch position.
    speed
        Launch speed in m/s.
    launch_angle_deg
        Launch angle above the horizontal, in degrees.
    meas_std
        ``(x, y)`` standard deviations of the position readings. The default
        used to be ``(1.0, 1.0)`` here and ``(0.0, 0.0)`` one class above, so
        the same "no noise" spelling produced different results; both default to
        silence now.
    rng
        Random generator; a fresh unseeded one is built when omitted.
    """

    def __init__(
        self,
        *,
        x0: float,
        y0: float,
        speed: float,
        launch_angle_deg: float,
        meas_std: tuple[float, float] = (0.0, 0.0),
        rng: np.random.Generator | None = None,
    ):
        require_nonnegative("speed", speed)
        _require_std_pair("meas_std", meas_std)

        omega = math.radians(launch_angle_deg)
        self.v_x = speed * math.cos(omega)
        self.v_y = speed * math.sin(omega)
        self.x = x0
        self.y = y0
        self.meas_std = meas_std
        self.rng = resolve_rng(rng)

    def update(self, *, dt: float, wind_speed: float = 0.0) -> tuple[float, float]:
        """Advance ``dt`` and return the noisy ``(x, y)`` reading.

        Positions use the velocity at the start of the step, then the velocity
        is corrected for drag and gravity — semi-implicit Euler, unchanged from
        the original model.
        """

        self.x += self.v_x * dt
        self.y += self.v_y * dt
        v_x_wind = self.v_x - wind_speed
        speed = np.sqrt(v_x_wind**2 + self.v_y**2)
        force = drag_force(speed)
        self.v_x = self.v_x - force * v_x_wind * dt
        self.v_y = self.v_y - GRAVITY_BASEBALL * dt - force * self.v_y * dt
        return (
            self.x + self.rng.standard_normal() * self.meas_std[0],
            self.y + self.rng.standard_normal() * self.meas_std[1],
        )

    def simulate(self, *, num: int, dt: float, wind_speed: float = 0.0) -> Tracks:
        """Run ``num`` steps; ``x`` and ``z`` are both ``(N, 2)``."""

        require_count(num)
        xs = np.empty((num, 2))
        zs = np.empty((num, 2))
        for i in range(num):
            zs[i] = self.update(dt=dt, wind_speed=wind_speed)
            xs[i] = (self.x, self.y)
        return Tracks(t=(np.arange(num) + 1.0) * dt, x=xs, z=zs)


def _require_std_pair(name: str, value: tuple[float, float]) -> None:
    if len(value) != 2:
        msg = f"{name} must hold two entries, got {value!r}"
        raise ValueError(msg)
    for entry in value:
        require_nonnegative(name, entry)
