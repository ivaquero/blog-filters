"""Reusable synthetic-measurement generators.

Every function that touches randomness accepts an explicit
`numpy.random.Generator` (`rng=`), so callers own the stream and can pin
reproducibility by building a fresh `default_rng(seed)` before the call.
The previous `seed=` parameter has been retired: pass
`rng=np.random.default_rng(seed)` at the call site instead.
"""

from collections.abc import Sequence

import numpy as np

from filters._typing import resolve_rng

from ._angles import wrap_angle
from ._types import Tracks
from ._validation import require_count, require_nonnegative

__all__ = [
    "add_jitter",
    "gen_async_sensor_data",
    "gen_cvca",
    "gen_jittered_vel",
    "gen_particles_gaussian",
    "gen_particles_uniform",
]

Range = Sequence[float]
Vector3 = Sequence[float]


def gen_cvca(
    *,
    num: int,
    x0: float,
    vel: float,
    acc: float = 0.0,
    dt: float = 1.0,
    meas_var: float = 1.0,
    rng: np.random.Generator | None = None,
) -> Tracks:
    """Constant-velocity / constant-acceleration truth plus noisy readings.

    Integrates ``x += v*dt + a*dt**2/2`` and ``v += a*dt``, then reads the
    resulting position.

    The retired version mixed the acceleration term into the measurement
    channel instead of the state — ``x += ddx * i**2 / 2 + dx * i`` ran after
    the sample had been appended — so the readings drifted away from the truth
    at a rate that grew with ``i``. With ``meas_var=0`` the two channels used to
    disagree by ``0, 1.15, 2.60, 4.35, 6.40`` over five steps; they now agree
    exactly.

    Parameters
    ----------
    num
        Number of samples.
    x0
        Initial position.
    vel
        Initial velocity.
    acc
        Constant acceleration.
    dt
        Time step.
    meas_var
        Variance of the additive position noise.
    rng
        Random generator; a fresh unseeded one is built when omitted.

    Returns
    -------
    Tracks
        ``x`` and ``z`` are both ``(N, 1)``.
    """

    require_count(num)
    require_nonnegative("meas_var", meas_var)
    require_nonnegative("dt", dt)
    generator = resolve_rng(rng)
    meas_std = np.sqrt(meas_var)

    position = x0
    velocity = vel
    xs = np.empty(num)
    zs = np.empty(num)
    for i in range(num):
        position += velocity * dt + 0.5 * acc * dt**2
        velocity += acc * dt
        xs[i] = position
        zs[i] = position + generator.standard_normal() * meas_std
    return Tracks(t=(np.arange(num) + 1.0) * dt, x=xs[:, None], z=zs[:, None])


def gen_jittered_vel(
    *,
    num: int,
    x0: float,
    vel: float,
    dt: float = 1.0,
    vel_var: float = 0.0,
    meas_var: float = 1.0,
    rng: np.random.Generator | None = None,
) -> Tracks:
    """Constant-velocity truth with process noise on the velocity term.

    Parameters
    ----------
    num
        Number of samples.
    x0
        Initial position.
    vel
        Initial velocity.
    dt
        Time step.
    vel_var
        Variance of the per-step velocity jitter.
    meas_var
        Variance of the additive position noise.
    rng
        Random generator; a fresh unseeded one is built when omitted.
    """

    require_count(num)
    require_nonnegative("vel_var", vel_var)
    require_nonnegative("meas_var", meas_var)
    generator = resolve_rng(rng)
    process_std = np.sqrt(vel_var)
    meas_std = np.sqrt(meas_var)

    position = x0
    xs = np.empty(num)
    zs = np.empty(num)
    for i in range(num):
        position += (vel + generator.standard_normal() * process_std) * dt
        xs[i] = position
        zs[i] = position + generator.standard_normal() * meas_std
    return Tracks(t=(np.arange(num) + 1.0) * dt, x=xs[:, None], z=zs[:, None])


def add_jitter(
    center: float | np.ndarray, *, std: float, rng: np.random.Generator | None = None
) -> float | np.ndarray:
    """Add zero-mean normal jitter of standard deviation ``std`` to ``center``.

    Named ``jitterfy`` before; a verb coined from a noun read as a joke in a
    signature that also validated its ``std`` through a helper called
    ``_validate_variance``.
    """

    require_nonnegative("std", std)
    return center + (resolve_rng(rng).standard_normal() * std)


def gen_async_sensor_data(
    *,
    duration: float,
    pos_std: float,
    vel_std: float,
    pos_rate: float = 3.0,
    vel_rate: float = 7.0,
    pos_time_std: float = 0.01,
    vel_time_std: float = 0.006,
    vel_offset: float = 1.0,
    rng: np.random.Generator | None = None,
) -> tuple[list[list[float]], list[list[float]]]:
    """Two sensors sampling at different rates, with jittered timestamps.

    Returns ``([t, z] * n_pos, [t, z] * n_vel)`` as plain lists so a caller can
    merge the streams by popping whichever timestamp comes first.

    Parameters
    ----------
    duration
        Length of the recording in seconds. The retired name was ``time``,
        which read like a timestamp but was a count of seconds.
    pos_std, vel_std
        Standard deviations of the two sensors' value noise.
    pos_rate, vel_rate
        Sample rates in Hz. The rates used to be baked into ``duration * 3``
        and ``duration * 7`` literals inside the loops.
    pos_time_std, vel_time_std
        Standard deviations of the timestamp jitter.
    vel_offset
        Constant reported by the velocity sensor.
    rng
        Random generator; a fresh unseeded one is built when omitted.
    """

    require_nonnegative("duration", duration)
    require_nonnegative("pos_std", pos_std)
    require_nonnegative("vel_std", vel_std)
    require_nonnegative("pos_rate", pos_rate)
    require_nonnegative("vel_rate", vel_rate)
    generator = resolve_rng(rng)

    pos_data: list[list[float]] = []
    vel_data: list[list[float]] = []

    stamp = 0.0
    for _ in range(int(duration * pos_rate)):
        stamp += 1.0 / pos_rate
        time_jittered = stamp + generator.standard_normal() * pos_time_std
        pos_data.append(
            [time_jittered, time_jittered + generator.standard_normal() * pos_std]
        )

    stamp = 0.0
    for _ in range(int(duration * vel_rate)):
        stamp += 1.0 / vel_rate
        time_jittered = stamp + generator.standard_normal() * vel_time_std
        vel_data.append(
            [time_jittered, vel_offset + generator.standard_normal() * vel_std]
        )
    return pos_data, vel_data


def gen_particles_uniform(
    *,
    x_range: Range,
    y_range: Range,
    heading_range: Range,
    num: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Particles drawn uniformly from the x / y / heading ranges.

    Headings are wrapped into ``[-pi, pi)``; the retired version used
    ``%= 2*pi`` and produced ``[0, 2*pi)``, a third heading convention in a
    package that already had two.
    """

    require_count(num)
    generator = resolve_rng(rng)

    particles = np.empty((num, 3))
    particles[:, 0] = generator.uniform(x_range[0], x_range[1], size=num)
    particles[:, 1] = generator.uniform(y_range[0], y_range[1], size=num)
    particles[:, 2] = generator.uniform(heading_range[0], heading_range[1], size=num)
    particles[:, 2] = wrap_angle(particles[:, 2])
    return particles


def gen_particles_gaussian(
    *, mean: Vector3, std: Vector3, num: int, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Particles drawn from independent Gaussians per state dimension."""

    require_count(num)
    for entry in std:
        require_nonnegative("std", entry)
    generator = resolve_rng(rng)

    particles = np.empty((num, 3))
    particles[:, 0] = mean[0] + (generator.standard_normal(num) * std[0])
    particles[:, 1] = mean[1] + (generator.standard_normal(num) * std[1])
    particles[:, 2] = mean[2] + (generator.standard_normal(num) * std[2])
    particles[:, 2] = wrap_angle(particles[:, 2])
    return particles


_ALIASES: dict[str, tuple[str | None, str]] = {
    "gen_sensor_data": (None, "gen_async_sensor_data"),
    "jitterfy": (None, "add_jitter"),
}


def __getattr__(name: str) -> object:
    from ._deprecation import deprecated_lookup

    return deprecated_lookup(name, _ALIASES, module_name=__name__, namespace=globals())
