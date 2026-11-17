"""Differential-drive robot: kinematics, landmark sensing, particle-filter run.

Three things were pulled out of this module during the layering pass:

* ``RobotEKF`` moved to :mod:`filters.robot_ekf` — a filter is not a simulator,
  and its presence here made ``simulators`` import ``filters``.
* ``robot3d_symbol`` moved to :mod:`symbols.robot`.
* The plotting calls that were inlined in the particle-filter scenario are now
  :func:`plots.plot_pf.plot_robot_pf`, driven by returned data rather than an
  ``Axes`` passed in.

The kinematic model is not duplicated: :func:`robot_move` is re-exported from
:mod:`filters.robot_ekf`, which is where the EKF needs it.
"""

import math
from typing import NamedTuple

import numpy as np

from filters import particle, resamplers
from filters._typing import resolve_rng
from filters.robot_ekf import robot_move

from ._angles import wrap_angle
from .datagen import gen_particles_gaussian, gen_particles_uniform

__all__ = [
    "DEFAULT_LANDMARKS",
    "RobotPFRun",
    "landmark_jacobian",
    "landmark_range",
    "landmark_ranges",
    "landmark_reading",
    "measurement_mean",
    "normalize_angle",
    "residual_landmark_array",
    "residual_range_bearing",
    "residual_state",
    "robot_move",
    "robot_turn",
    "simulate_robot_pf",
    "state_mean",
]

DEFAULT_LANDMARKS = np.array([[-1, 2], [5, 10], [12, 14], [18, 21]])


def robot_turn(
    *, speed: float, start_deg: float, end_deg: float, steps: int
) -> np.ndarray:
    """Build ``[speed, steering]`` commands sweeping an arc.

    Angles are given in degrees and converted to the radians the control input
    expects. The result is an ``(steps, 2)`` array; the retired version returned
    a list of lists, which then had to be concatenated with other lists.

    Parameters
    ----------
    speed
        Constant forward speed during the turn.
    start_deg, end_deg
        Steering angle at the first and last step, in degrees.
    steps
        Number of command rows to generate.
    """

    if steps < 1:
        msg = f"steps must be at least 1, got {steps}"
        raise ValueError(msg)
    angles = np.linspace(np.radians(start_deg), np.radians(end_deg), steps)
    return np.column_stack((np.full(steps, speed), angles))


def normalize_angle(angle):
    """Wrap an angle in radians into ``[-pi, pi)``.

    The docstring always claimed ``[-pi, pi)`` while the arithmetic produced
    ``(-pi, pi]`` — ``-pi`` came back as ``+pi``. Both now match.
    """

    return wrap_angle(angle)


def _state_xy_heading(x):
    """Flatten a robot state to ``(x, y, heading)``.

    States reach these helpers either as 1-D ``(3,)`` arrays (UKF convention)
    or as ``(3, 1)`` column vectors (EKF convention). Indexing a column vector
    with ``x[0]`` yields a length-1 array, which ``math.atan2`` refuses to
    accept as a scalar, so both callers must be normalised here.
    """

    px, py, heading = np.asarray(x, dtype=float).reshape(-1)[:3]
    return px, py, heading


def landmark_ranges(x, landmarks) -> np.ndarray:
    """Return ``[distance, bearing]`` for every landmark, flattened.

    Formerly ``Hx``.
    """

    px, py, heading = _state_xy_heading(x)
    measurements = []
    for landmark in landmarks:
        dx, dy = landmark[0] - px, landmark[1] - py
        distance = np.hypot(dx, dy)
        angle = math.atan2(dy, dx)
        measurements += [distance, normalize_angle(angle - heading)]
    return np.array(measurements)


def landmark_range(x, landmark) -> np.ndarray:
    """Return ``[distance, bearing]`` for a single landmark.

    Formerly ``Hx_1``.
    """

    px, py, heading = _state_xy_heading(x)
    dx, dy = landmark[0] - px, landmark[1] - py
    distance = np.hypot(dx, dy)
    angle = math.atan2(dy, dx)
    return np.array([distance, normalize_angle(angle - heading)])


def landmark_jacobian(x, landmark) -> np.ndarray:
    """Jacobian of :func:`landmark_range` with respect to the state.

    Formerly ``HJac``.
    """

    px, py, _ = _state_xy_heading(x)
    dx, dy = landmark[0] - px, landmark[1] - py
    distance = np.hypot(dx, dy)
    return np.array(
        [
            [-dx / distance, -dy / distance, 0.0],
            [dy / distance**2, -dx / distance**2, -1.0],
        ]
    )


def landmark_reading(
    landmark,
    sim_pos,
    *,
    range_std: float,
    bearing_std: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Simulate a noisy ``[distance, bearing]`` reading of one landmark.

    Formerly ``z_landmark``. The bearing is now wrapped: :func:`landmark_range`
    wraps it on the model side, so the simulator previously produced readings the
    model could not represent whenever the bearing crossed ``±pi``.
    """

    generator = resolve_rng(rng)
    px, py, heading = _state_xy_heading(sim_pos)
    distance = np.hypot(landmark[0] - px, landmark[1] - py)
    angle = math.atan2(landmark[1] - py, landmark[0] - px) - heading
    return np.array(
        [
            distance + generator.standard_normal() * range_std,
            normalize_angle(angle + generator.standard_normal() * bearing_std),
        ]
    )


def state_mean(sigmas, Wm) -> np.ndarray:
    """Circular mean of sigma points for the ``[x, y, heading]`` state."""

    mean = np.zeros(3)
    mean[0] = sigmas[:, 0] @ Wm
    mean[1] = sigmas[:, 1] @ Wm
    mean[2] = math.atan2(np.sin(sigmas[:, 2]) @ Wm, np.cos(sigmas[:, 2]) @ Wm)
    return mean


def measurement_mean(sigmas, Wm) -> np.ndarray:
    """Circular mean for repeated ``[distance, bearing]`` measurement pairs.

    Formerly ``z_mean``.
    """

    count = sigmas.shape[1]
    mean = np.zeros(count)
    for index in range(0, count, 2):
        mean[index] = sigmas[:, index] @ Wm
        mean[index + 1] = math.atan2(
            np.sin(sigmas[:, index + 1]) @ Wm, np.cos(sigmas[:, index + 1]) @ Wm
        )
    return mean


def residual_range_bearing(a, b) -> np.ndarray:
    """Residual of a single ``[distance, bearing]`` pair, bearing wrapped.

    Formerly ``residual``.
    """

    residual = a - b
    residual[1] = normalize_angle(residual[1])
    return residual


def residual_landmark_array(a, b) -> np.ndarray:
    """Residual of a flat ``[distance, bearing, ...]`` array.

    Formerly ``residual_h``.
    """

    residual = a - b
    for index in range(0, len(residual), 2):
        residual[index + 1] = normalize_angle(residual[index + 1])
    return residual


def residual_state(a, b) -> np.ndarray:
    """Residual of a ``[x, y, heading]`` state, heading wrapped.

    Formerly ``residual_x``.
    """

    residual = a - b
    residual[2] = normalize_angle(residual[2])
    return residual


class RobotPFRun(NamedTuple):
    """Output of :func:`simulate_robot_pf`.

    Attributes
    ----------
    t
        Step indices, shape ``(iters,)``.
    x_true
        True robot position per step, shape ``(iters, 2)``.
    x_est
        Particle-filter mean per step, shape ``(iters, 2)``.
    var
        Particle-filter variance per step, shape ``(iters, 2)``.
    particles
        Particle clouds when ``keep_particles=True``, else ``None``. Entry 0 is
        the prior cloud before the first prediction; entry ``i + 1`` is the
        cloud after step ``i``.
    """

    t: np.ndarray
    x_true: np.ndarray
    x_est: np.ndarray
    var: np.ndarray
    particles: list[np.ndarray] | None = None


def simulate_robot_pf(
    *,
    num_particles: int,
    iters: int = 18,
    meas_std: float = 0.1,
    initial_state=None,
    landmarks=None,
    keep_particles: bool = False,
    rng: np.random.Generator | None = None,
) -> RobotPFRun:
    """Localise a robot that walks the diagonal in front of known landmarks.

    Returns the run instead of plotting it. The retired ``robot_pf(ax, N, ...)``
    required a Matplotlib ``Axes``, drew into it, printed the final error and
    returned ``None``, so the scenario could not be used headless or swept over
    parameters.

    Parameters
    ----------
    num_particles
        Number of particles. Formerly ``N``.
    iters
        Number of steps.
    meas_std
        Standard deviation of the range measurements, in metres. It is passed
        to ``pf_update`` as the Gaussian scale, which is what that function
        takes.
    initial_state
        ``(x, y, heading)`` to scatter the particles around; ``None`` spreads
        them uniformly over ``[0, 20] x [0, 20]``.
    landmarks
        Landmark array; defaults to :data:`DEFAULT_LANDMARKS`.
    keep_particles
        Store each step's particle cloud for visualisation.
    rng
        Random generator; a fresh unseeded one is built when omitted.
    """

    generator = resolve_rng(rng)
    marks = (
        DEFAULT_LANDMARKS if landmarks is None else np.asarray(landmarks, dtype=float)
    )
    count = len(marks)

    if initial_state is not None:
        particles = gen_particles_gaussian(
            mean=initial_state, std=(5, 5, np.pi / 4), num=num_particles, rng=generator
        )
    else:
        particles = gen_particles_uniform(
            x_range=(0, 20),
            y_range=(0, 20),
            heading_range=(0, 2 * np.pi),
            num=num_particles,
            rng=generator,
        )
    weights = np.ones(num_particles) / num_particles

    history: list[np.ndarray] | None = (
        [particles.copy()] if keep_particles else None
    )
    truth = np.empty((iters, 2))
    means = np.empty((iters, 2))
    variances = np.empty((iters, 2))

    robot_pos = np.array([0.0, 0.0])
    for step in range(iters):
        robot_pos = robot_pos + np.array([1.0, 1.0])
        zs = np.linalg.norm(marks - robot_pos, axis=1) + (
            generator.standard_normal(count) * meas_std
        )
        particle.pf_predict(particles, u=(0.00, 1.414), std=(0.2, 0.05), rng=generator)
        particle.pf_update(particles, weights, z=zs, R=meas_std, landmarks=marks)
        if particle.neff(weights) < num_particles / 2:
            indexes = resamplers.systematic_resample(weights, rng=generator)
            resamplers.resample_from_index(particles, weights, indexes)
        mu, var = particle.pf_estimate(particles, weights)

        truth[step] = robot_pos
        means[step] = mu
        variances[step] = var
        if history is not None:
            history.append(particles.copy())

    return RobotPFRun(
        t=np.arange(1, iters + 1),
        x_true=truth,
        x_est=means,
        var=variances,
        particles=history,
    )


_ALIASES: dict[str, tuple[str | None, str]] = {
    "HJac": (None, "landmark_jacobian"),
    "Hx": (None, "landmark_ranges"),
    "Hx_1": (None, "landmark_range"),
    "RobotEKF": ("filters.robot_ekf", "RobotEKF"),
    "robot3d_symbol": ("symbols.robot", "robot3d_symbol"),
    "robot_pf": (None, "simulate_robot_pf"),
    "z_landmark": (None, "landmark_reading"),
    "z_mean": (None, "measurement_mean"),
}


def __getattr__(name: str) -> object:
    from ._deprecation import deprecated_lookup

    return deprecated_lookup(name, _ALIASES, module_name=__name__, namespace=globals())
