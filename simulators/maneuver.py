"""Planar maneuvering-target simulator.

Heading is a **compass** angle: ``heading_deg=0`` points along ``+y``, and the
angle grows clockwise, so ``90`` points along ``+x``. :func:`simulate_turning`
is the exception — it works in Cartesian ``[x, vx, z, vz]`` space because it
feeds a linear acceleration model, and its docstring says so.
"""

import math

import numpy as np

from filters._typing import resolve_rng
from models.const_vel import constant_velocity_transition_matrix

from ._angles import wrap_degrees
from ._types import Tracks
from ._validation import require_count, require_nonnegative

__all__ = [
    "ManeuveringTarget",
    "heading_delta",
    "simulate_maneuver",
    "simulate_turning",
]


def heading_delta(target_deg: float, current_deg: float) -> float:
    """Signed shortest turn from ``current_deg`` to ``target_deg``, in degrees.

    Returns a value in ``[-180, 180)``: positive is clockwise, the direction the
    compass heading grows. The retired ``angle_between(x, y)`` returned
    ``y - x`` over a window of only one turn, so it was both sign-inverted
    relative to its call site — commanding 310 degrees from 0 turned the target
    to 50 degrees — and wrong once the heading had accumulated past 360.
    """

    return wrap_degrees(target_deg - current_deg)


class ManeuveringTarget:
    """Target flying at ``speed`` along ``heading_deg``, with commanded turns.

    A turn is spread over a number of steps: :meth:`set_commanded_heading` only
    schedules the per-step increment, and each :meth:`update` applies one
    increment *after* moving, so the new heading takes effect from the next
    step.

    Parameters
    ----------
    x0, y0
        Initial position.
    speed
        Initial speed, position units per second.
    heading_deg
        Initial compass heading in degrees.
    dt
        Default time step, used when :meth:`update` is called without one.
    """

    def __init__(
        self,
        *,
        x0: float = 0.0,
        y0: float = 0.0,
        speed: float = 0.3,
        heading_deg: float = 0.0,
        dt: float = 1.0,
    ):
        self.x = x0
        self.y = y0
        self.vel = speed
        self.hdg = float(wrap_degrees(heading_deg))
        self.dt = dt
        self.cmd_vel = speed
        self.cmd_hdg = self.hdg
        self.vel_step = 0
        self.hdg_step = 0
        self.vel_delta = 0.0
        self.hdg_delta = 0.0

    def update(self, *, dt: float | None = None) -> tuple[float, float]:
        """Advance one step and return the new ``(x, y)``.

        The heading is wrapped back into ``[-180, 180)`` after every step. It
        used to accumulate without bound, which is what made the shortest-turn
        arithmetic report a 360-degree correction for a target that had flown
        two full circles.
        """

        step = self.dt if dt is None else dt
        angle = math.radians(90.0 - self.hdg)
        self.x += self.vel * math.cos(angle) * step
        self.y += self.vel * math.sin(angle) * step

        if self.hdg_step > 0:
            self.hdg_step -= 1
            self.hdg += self.hdg_delta
            self.hdg = float(wrap_degrees(self.hdg))

        if self.vel_step > 0:
            self.vel_step -= 1
            self.vel += self.vel_delta
        return (self.x, self.y)

    def set_commanded_heading(self, heading_deg: float, steps: int) -> None:
        """Turn to ``heading_deg`` over ``steps`` steps.

        Raises ``ValueError`` for a step count below one, which the retired
        implementation answered with ``ZeroDivisionError``.
        """

        self.cmd_hdg = float(wrap_degrees(heading_deg))
        delta = heading_delta(self.cmd_hdg, self.hdg)
        if steps < 1:
            msg = f"steps must be at least 1, got {steps}"
            raise ValueError(msg)
        self.hdg_delta = delta / steps
        self.hdg_step = steps if abs(self.hdg_delta) > 0 else 0

    def set_commanded_speed(self, speed: float, steps: int) -> None:
        """Change speed to ``speed`` over ``steps`` steps."""

        require_nonnegative("speed", speed)
        self.cmd_vel = speed
        if steps < 1:
            msg = f"steps must be at least 1, got {steps}"
            raise ValueError(msg)
        self.vel_delta = (self.cmd_vel - self.vel) / steps
        self.vel_step = steps if abs(self.vel_delta) > 0 else 0


def simulate_maneuver(
    *,
    steady_count: int,
    meas_std: float,
    dt: float = 1.0,
    warmup: int = 30,
    initial_speed: float = 0.3,
    turn_heading_deg: float = 310.0,
    turn_steps: int = 25,
    target_speed: float = 1.0,
    speed_steps: int = 15,
    rng: np.random.Generator | None = None,
) -> Tracks:
    """Fly straight, then command a turn and a speed change.

    Parameters
    ----------
    steady_count
        Steps flown after the maneuver commands are issued.
    meas_std
        Standard deviation of the position measurement noise.
    dt
        Time step.
    warmup
        Steps flown straight before the maneuver. This used to be the literal
        ``30``, which forced callers to subtract it by hand (one notebook ran
        ``simulate_maneuver(180 - 30, ...)`` to compensate).
    initial_speed, turn_heading_deg, turn_steps, target_speed, speed_steps
        Maneuver profile. These were literals inside the function body.
    rng
        Random generator; a fresh unseeded one is built when omitted.

    Returns
    -------
    Tracks
        ``x`` is the true ``(x, y)`` track, ``z`` the noisy reading of it.
    """

    require_count(steady_count)
    require_nonnegative("meas_std", meas_std)
    require_nonnegative("dt", dt)
    generator = resolve_rng(rng)

    target = ManeuveringTarget(
        x0=0.0, y0=0.0, speed=initial_speed, heading_deg=0.0, dt=dt
    )
    xs: list[float] = []
    ys: list[float] = []
    for _ in range(warmup):
        x, y = target.update()
        xs.append(x)
        ys.append(y)

    target.set_commanded_heading(turn_heading_deg, turn_steps)
    target.set_commanded_speed(target_speed, speed_steps)

    for _ in range(steady_count):
        x, y = target.update()
        xs.append(x)
        ys.append(y)

    pos = np.column_stack((xs, ys))
    zs = pos + generator.standard_normal(pos.shape) * meas_std
    return Tracks(t=(np.arange(len(xs)) + 1.0) * dt, x=pos, z=zs)


def simulate_turning(
    *,
    num: int = 600,
    turn_start: int = 400,
    dt: float = 1.0,
    accel: tuple[float, float] = (0.075, 0.075),
) -> Tracks:
    """Cartesian CV trajectory with a constant acceleration applied mid-run.

    Unlike :class:`ManeuveringTarget` this works in ``[x, vx, z, vz]`` state
    space and has no sensor, so ``Tracks.z`` is ``None``. The retired version
    returned an ``(N, 4, 1)`` stack that every caller had to ``reshape``.

    Parameters
    ----------
    num
        Number of steps.
    turn_start
        Step index from which the acceleration is applied.
    dt
        Time step.
    accel
        Constant ``(ax, az)`` acceleration applied from ``turn_start`` on.
    """

    require_count(num)
    transition = constant_velocity_transition_matrix(4, dt)
    accel_matrix = np.array([[dt**2 / 2, 0.0], [dt, 0.0], [0.0, dt**2 / 2], [0.0, dt]])
    accel_vector = np.asarray(accel, dtype=float)[:, None]

    state = np.array([[2000.0], [0.0], [10000.0], [-15.0]])
    states = np.empty((num, 4))
    for i in range(num):
        state = transition @ state
        if i >= turn_start:
            state = state + accel_matrix @ accel_vector
        states[i] = state.ravel()

    return Tracks(t=(np.arange(num) + 1.0) * dt, x=states)
