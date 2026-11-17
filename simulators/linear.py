"""One- and two-dimensional constant-velocity simulators.

Every class accepts a `numpy.random.Generator` via `rng=` (no more
`seed=` integers scattered through the constructor). Callers build a
`default_rng(seed)` at the top of a script or notebook, hand it to the
simulator, and get byte-for-byte reproducibility across calls.

Noise parameters are spelled out in the name: ``*_var`` is a **variance**
(the same number a filter's ``Q``/``R`` matrix holds) and ``*_std`` is a
**standard deviation**. The two used to share the names ``Q`` and ``R`` while
meaning different things in adjacent classes, so a caller could not tell from
the call site whether to square.

Every batch method returns :class:`simulators._types.Tracks` with ``x`` of
shape ``(N, D)`` and ``z`` of shape ``(N, M)``; the previous mix of 1-D arrays,
``list`` objects and 3-D stacks is gone.
"""

import numpy as np

from filters._typing import resolve_rng

from ._types import Tracks
from ._validation import require_count, require_nonnegative

__all__ = ["CAObject", "CVObject", "DogSimulation", "MovingTarget2D"]


class DogSimulation:
    """1-D "dog walking" simulator with per-step velocity noise.

    A dog walks along a line at a mean velocity; every step its actual velocity
    jitters, and the position sensor reads the result with additive noise.

    Parameters
    ----------
    x0
        Initial position.
    model_mean
        Mean velocity, position units per second.
    vel_var
        Variance of the per-step velocity jitter — the ``Q`` of a
        constant-velocity model.
    meas_var
        Variance of the position measurement noise — the ``R`` of a
        constant-velocity model.
    rng
        Random generator; a fresh unseeded one is built when omitted.

    Examples
    --------
    >>> dog = DogSimulation(
    ...     vel_var=0.5, meas_var=0.2, rng=np.random.default_rng(13)
    ... )
    >>> dog.simulate(num=10).x.shape
    (10, 1)
    """

    def __init__(
        self,
        *,
        x0: float = 0.0,
        model_mean: float = 1.0,
        vel_var: float = 0.0,
        meas_var: float = 0.0,
        rng: np.random.Generator | None = None,
    ):
        require_nonnegative("vel_var", vel_var)
        require_nonnegative("meas_var", meas_var)
        self.x = x0
        self.model_mean = model_mean
        self.p_std = np.sqrt(vel_var)
        self.z_std = np.sqrt(meas_var)
        self.rng = resolve_rng(rng)

    def move(self, *, dt: float = 1.0, acc: float = 0.0) -> float:
        """Advance one step and return the new true position.

        ``acc`` is the change in mean velocity per unit time, so the mean grows
        by ``acc * dt`` per step. It used to be added without the ``dt`` factor,
        which silently rescaled the acceleration whenever ``dt != 1``.
        """

        velocity = self.model_mean + self.rng.standard_normal() * self.p_std
        self.x += velocity * dt
        self.model_mean += acc * dt
        return self.x

    def sense_position(self) -> float:
        """Return a noisy reading of the current true position."""

        return self.x + self.rng.standard_normal() * self.z_std

    def move_and_sense(self, *, dt: float = 1.0, acc: float = 0.0) -> float:
        """Advance one step, then return the noisy reading of that position."""

        self.move(dt=dt, acc=acc)
        return self.sense_position()

    def simulate(self, *, num: int = 1, dt: float = 1.0, acc: float = 0.0) -> Tracks:
        """Run ``num`` steps and return the truth and measurement samples.

        The old implementation advanced the state twice per sample — once in a
        loop building ``xs`` and again in a loop building ``zs`` — so the second
        half of the run measured states the first half had already left behind.
        """

        require_count(num)
        xs = np.empty(num)
        zs = np.empty(num)
        for i in range(num):
            xs[i] = self.move(dt=dt, acc=acc)
            zs[i] = self.sense_position()
        return Tracks(t=(np.arange(num) + 1.0) * dt, x=xs[:, None], z=zs[:, None])


class MovingTarget2D:
    """A target moving in the plane, observed by a position sensor.

    The class was called ``PosSensor``, which described only half of what it
    does: it owns the target's motion *and* the sensor. Motion and sensing are
    now separate calls — :meth:`advance` moves the truth, :meth:`measure` reads
    it without side effects, and :meth:`sample` does both in one step.

    Parameters
    ----------
    pos
        Initial ``(x, y)`` position.
    vel
        Constant ``(vx, vy)`` velocity, position units per second.
    meas_std
        Standard deviation of the additive position noise.
    rng
        Random generator; a fresh unseeded one is built when omitted.
    """

    def __init__(
        self,
        *,
        pos: tuple[float, float] = (0.0, 0.0),
        vel: tuple[float, float] = (0.0, 0.0),
        meas_std: float = 1.0,
        rng: np.random.Generator | None = None,
    ):
        require_nonnegative("meas_std", meas_std)
        self.pos = np.asarray(pos, dtype=float)
        self.vel = np.asarray(vel, dtype=float)
        self.meas_std = meas_std
        self.rng = resolve_rng(rng)

    def advance(self, *, dt: float = 1.0) -> np.ndarray:
        """Move the truth one step and return the new position."""

        self.pos = self.pos + self.vel * dt
        return self.pos

    def measure(self) -> np.ndarray:
        """Read the current true position with additive noise.

        Unlike the retired ``read()`` this does not advance the target, so a
        caller can sample the initial state or take several readings of one
        position.
        """

        return self.pos + self.rng.standard_normal(2) * self.meas_std

    def sample(self, *, dt: float = 1.0) -> np.ndarray:
        """Advance one step, then return the noisy reading of that position."""

        self.advance(dt=dt)
        return self.measure()

    def simulate(self, *, num: int, dt: float = 1.0) -> Tracks:
        """Run ``num`` steps and return the truth ``(N, 2)`` and readings."""

        require_count(num)
        xs = np.empty((num, 2))
        zs = np.empty((num, 2))
        for i in range(num):
            xs[i] = self.advance(dt=dt)
            zs[i] = self.measure()
        return Tracks(t=(np.arange(num) + 1.0) * dt, x=xs, z=zs)


class CVObject:
    """Constant-velocity object with process noise on velocity.

    Parameters
    ----------
    x0
        Initial position.
    vel
        Initial velocity, position units per second.
    vel_std
        Per-step standard deviation of the velocity increment. It is a
        *discrete* white-noise term, so it does not scale with ``dt``.
    meas_std
        Standard deviation of the position measurement noise.
    rng
        Random generator; a fresh unseeded one is built when omitted.

    Notes
    -----
    ``vel_std``/``meas_std`` replaced the parameters named ``Q`` and ``R``.
    Those were standard deviations here but variances in :class:`DogSimulation`
    and :func:`datagen.gen_jittered_vel`, so the same ``Q=0.02`` meant two
    different things one file apart.
    """

    def __init__(
        self,
        *,
        x0: float = 0.0,
        vel: float = 1.0,
        vel_std: float = 0.06,
        meas_std: float = 0.1,
        rng: np.random.Generator | None = None,
    ):
        require_nonnegative("vel_std", vel_std)
        require_nonnegative("meas_std", meas_std)
        self.x = x0
        self.vel = vel
        self.vel_std = vel_std
        self.meas_std = meas_std
        self.rng = resolve_rng(rng)

    def update(self, *, dt: float = 1.0) -> tuple[float, float]:
        """Advance one step and return ``(position, velocity)``."""

        self.vel += self.rng.standard_normal() * self.vel_std
        self.x += self.vel * dt
        return (self.x, self.vel)

    def sense(self) -> float:
        """Return a noisy position reading."""

        return self.x + self.rng.standard_normal() * self.meas_std

    def simulate(self, *, num: int, dt: float = 1.0) -> Tracks:
        """Run ``num`` steps; ``x`` is ``(N, 2)`` as ``[position, velocity]``.

        Written as a cumulative sum rather than a Python loop, but the noise
        draw order is preserved (one process sample and one measurement sample
        per step, interleaved), so a pinned seed yields the same numbers as the
        per-step API in :meth:`update` / :meth:`sense`.
        """

        require_count(num)
        noise = self.rng.standard_normal((num, 2))
        vels = self.vel + np.cumsum(noise[:, 0] * self.vel_std)
        xs = self.x + np.cumsum(vels * dt)
        zs = xs + noise[:, 1] * self.meas_std
        return Tracks(
            t=(np.arange(num) + 1.0) * dt, x=np.column_stack((xs, vels)), z=zs[:, None]
        )


class CAObject:
    """Constant-acceleration object with noise on the acceleration term.

    Parameters
    ----------
    x0
        Initial position.
    vel
        Initial velocity.
    acc
        Initial acceleration.
    acc_std
        Per-step standard deviation of the acceleration increment.
    meas_std
        Standard deviation of the position measurement noise.
    rng
        Random generator; a fresh unseeded one is built when omitted.
    """

    def __init__(
        self,
        *,
        x0: float = 0.0,
        vel: float = 1.0,
        acc: float = 0.1,
        acc_std: float = 0.1,
        meas_std: float = 0.1,
        rng: np.random.Generator | None = None,
    ):
        require_nonnegative("acc_std", acc_std)
        require_nonnegative("meas_std", meas_std)
        self.x = x0
        self.vel = vel
        self.acc = acc
        self.acc_std = acc_std
        self.meas_std = meas_std
        self.rng = resolve_rng(rng)

    def update(self, *, dt: float = 1.0) -> tuple[float, float, float]:
        """Advance one step and return ``(position, velocity, acceleration)``."""

        self.acc += self.rng.standard_normal() * self.acc_std
        self.vel += self.acc * dt
        self.x += self.vel * dt
        return (self.x, self.vel, self.acc)

    def sense(self) -> float:
        """Return a noisy position reading."""

        return self.x + self.rng.standard_normal() * self.meas_std

    def simulate(self, *, num: int, dt: float = 1.0) -> Tracks:
        """Run ``num`` steps; ``x`` is ``(N, 3)`` as ``[pos, vel, acc]``.

        The retired version returned ``zs`` as a plain ``list`` while ``xs`` was
        an array; both are now ``(N, 1)`` columns inside a :class:`Tracks`.
        """

        require_count(num)
        noise = self.rng.standard_normal((num, 2))
        accs = self.acc + np.cumsum(noise[:, 0] * self.acc_std)
        vels = self.vel + np.cumsum(accs * dt)
        xs = self.x + np.cumsum(vels * dt)
        zs = xs + noise[:, 1] * self.meas_std
        return Tracks(
            t=(np.arange(num) + 1.0) * dt,
            x=np.column_stack((xs, vels, accs)),
            z=zs[:, None],
        )


_ALIASES: dict[str, tuple[str | None, str]] = {"PosSensor": (None, "MovingTarget2D")}


def __getattr__(name: str) -> object:
    from ._deprecation import deprecated_lookup

    return deprecated_lookup(name, _ALIASES, module_name=__name__, namespace=globals())
