"""Ground-radar and airborne-target simulators.

This module used to hold nine plotting helpers and a SymPy derivation next to
the physics, which made ``import simulators.radar`` pull in ``matplotlib``
(~350 ms) and ``sympy`` (~210 ms) for a file that only needs ``numpy``. The
plotting code now lives in :mod:`plots.plot_radar` and the derivation in
:mod:`symbols.radar`; the retired names are still reachable through this
module's ``__getattr__`` with a ``DeprecationWarning``.

Measurement-model naming
------------------------
The three range/bearing models used to be ``H2dRE``, ``H3dRE`` and ``H4dRE``,
which differed *silently* in which state entries they read: ``H2dRE`` took
``x[0], x[1]`` while the other two took ``x[0], x[2]``. The names now say which
layout they expect, so a 4-state call cannot quietly read the wrong column.
"""

import math

import numpy as np

from filters._typing import resolve_rng

__all__ = [
    "ACSim",
    "RadarSim",
    "RadarStation",
    "range_bearing_doppler",
    "range_bearing_xy",
    "range_bearing_xz",
    "slant_range",
    "slant_range_jacobian",
]


class RadarStation:
    """Ground radar reporting slant range and bearing to a target.

    Parameters
    ----------
    pos
        Radar position in the horizontal plane.
    range_std
        Standard deviation of the range noise.
    bearing_std
        Standard deviation of the bearing noise, in radians. The parameter was
        named ``elev_std`` while the angle computed was ``atan2(dy, dx)`` — an
        azimuth, not an elevation — so callers routinely passed degrees.
    rng
        Random generator; a fresh unseeded one is built when omitted.
    """

    def __init__(
        self,
        pos,
        *,
        range_std: float,
        bearing_std: float,
        rng: np.random.Generator | None = None,
    ):
        self.pos = np.asarray(pos, dtype=float)
        self.range_std = range_std
        self.bearing_std = bearing_std
        self.rng = resolve_rng(rng)

    def true_reading(self, ac_pos) -> tuple[float, float]:
        """Return the noise-free ``(range, bearing)`` to the target.

        Bearing is in radians, measured from ``+x``.
        """

        diff = np.subtract(ac_pos, self.pos)
        distance = np.linalg.norm(diff)
        bearing = math.atan2(diff[1], diff[0])
        return distance, bearing

    def reading(self, ac_pos) -> tuple[float, float]:
        """Return ``(range, bearing)`` with additive noise."""

        distance, bearing = self.true_reading(ac_pos)
        distance += self.rng.standard_normal() * self.range_std
        bearing += self.rng.standard_normal() * self.bearing_std
        return distance, bearing


class ACSim:
    """Aircraft flying at constant mean velocity with per-step velocity noise.

    Parameters
    ----------
    pos
        Initial position.
    vel
        Mean velocity.
    vel_std
        Standard deviation of the velocity noise.
    rng
        Random generator; a fresh unseeded one is built when omitted.
    """

    def __init__(
        self, pos, vel, *, vel_std: float, rng: np.random.Generator | None = None
    ):
        self.pos = np.asarray(pos, dtype=float)
        self.vel = np.asarray(vel, dtype=float)
        self.vel_std = vel_std
        self.rng = resolve_rng(rng)

    def update(self, *, dt: float) -> np.ndarray:
        """Advance ``dt`` and return the new position.

        The velocity noise is a single draw applied to both axes — that is what
        the original model did, and switching to per-axis noise would move every
        figure built on this simulator, so it is documented rather than silently
        altered.
        """

        dx = self.vel * dt + (self.rng.standard_normal() * self.vel_std) * dt
        self.pos = self.pos + dx
        return self.pos


class RadarSim:
    """Slant-range radar tracking a target flying at constant altitude.

    Parameters
    ----------
    dt
        Time between measurements.
    pos
        Initial horizontal position.
    vel
        Initial horizontal velocity.
    alt
        Initial altitude.
    vel_std, alt_std
        Per-step standard deviations of the velocity and altitude jitter. They
        used to be the literals ``0.1`` and ``0.1`` inside :meth:`measure`.
    range_frac_std
        Fractional range noise; the reading is scaled by it. Was the literal
        ``0.05``.
    rng
        Random generator; a fresh unseeded one is built when omitted.
    """

    def __init__(
        self,
        *,
        dt: float,
        pos: float,
        vel: float,
        alt: float,
        vel_std: float = 0.1,
        alt_std: float = 0.1,
        range_frac_std: float = 0.05,
        rng: np.random.Generator | None = None,
    ):
        self.pos = pos
        self.vel = vel
        self.alt = alt
        self.dt = dt
        self.vel_std = vel_std
        self.alt_std = alt_std
        self.range_frac_std = range_frac_std
        self.rng = resolve_rng(rng)

    def measure(self) -> float:
        """Advance the target one step and return the noisy slant range.

        Renamed from ``get_range``: the method advanced the target *and*
        injected process noise, so a getter was mutating the simulation. It is
        still one call per measurement — the docstring now says which step it
        performs.
        """

        self.vel = self.vel + self.vel_std * self.rng.standard_normal()
        self.alt = self.alt + self.alt_std * self.rng.standard_normal()
        self.pos = self.pos + self.vel * self.dt
        error = self.pos * self.range_frac_std * self.rng.standard_normal()
        slant = np.hypot(self.pos, self.alt)
        return slant + error


def slant_range(x) -> float:
    """Return the slant range ``hypot(x[0], x[2])`` of a 3-state target.

    Formerly ``hx``.
    """

    state = np.asarray(x, dtype=float).reshape(-1)
    return float(np.hypot(state[0], state[2]))


def slant_range_jacobian(x) -> np.ndarray:
    """Return the Jacobian of :func:`slant_range`, shape ``(1, 3)``.

    Formerly ``HJac3d``.
    """

    state = np.asarray(x, dtype=float).reshape(-1)
    horizontal = state[0]
    altitude = state[2]
    denominator = np.hypot(horizontal, altitude)
    return np.array([[horizontal / denominator, 0.0, altitude / denominator]])


def range_bearing_xy(pos, *, refpos=(0.0, 0.0)) -> tuple[float, float]:
    """Range and bearing from ``refpos`` to a 2-D position ``[x, y]``.

    Formerly ``H2dRE``.
    """

    state = np.asarray(pos, dtype=float).reshape(-1)
    dx = state[0] - refpos[0]
    dy = state[1] - refpos[1]
    return float(np.hypot(dx, dy)), float(math.atan2(dy, dx))


def range_bearing_xz(x, *, refpos=(0.0, 0.0)) -> tuple[float, float]:
    """Range and bearing for a 3-state ``[x, vx, z]`` target.

    ``refpos`` is ``(x_ref, z_ref)``. Formerly ``H3dRE``.
    """

    state = np.asarray(x, dtype=float).reshape(-1)
    dx = state[0] - refpos[0]
    dz = state[2] - refpos[1]
    return float(np.hypot(dx, dz)), float(math.atan2(dz, dx))


def range_bearing_doppler(x, *, refpos=(0.0, 0.0)) -> tuple[float, float, float, float]:
    """Range, bearing and the two velocity components for a 4-state target.

    State layout is ``[x, vx, z, vz]``; ``refpos`` is ``(x_ref, z_ref)``.
    Formerly ``H4dRE``, whose name promised two outputs and returned four.
    """

    state = np.asarray(x, dtype=float).reshape(-1)
    distance, bearing = range_bearing_xz(state, refpos=refpos)
    return distance, bearing, float(state[1]), float(state[3])


_ALIASES: dict[str, tuple[str | None, str]] = {
    "H2dRE": (None, "range_bearing_xy"),
    "H3dRE": (None, "range_bearing_xz"),
    "H4dRE": (None, "range_bearing_doppler"),
    "HJac3d": (None, "slant_range_jacobian"),
    "hx": (None, "slant_range"),
    "plot_iscts_two_sensors": ("plots.plot_radar", "plot_iscts_two_sensors"),
    "plot_moving_target": ("plots.plot_radar", "plot_moving_target"),
    "plot_radar3d": ("plots.plot_radar", "plot_radar3d"),
    "plot_radar3d_zs": ("plots.plot_radar", "plot_radar3d_zs"),
    "radar3d_symbol": ("symbols.radar", "radar3d_symbol"),
    "show_radar_chart": ("plots.plot_radar", "show_radar_chart"),
}


def __getattr__(name: str) -> object:
    from ._deprecation import deprecated_lookup

    return deprecated_lookup(name, _ALIASES, module_name=__name__, namespace=globals())
