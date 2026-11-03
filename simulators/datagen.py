"""Reusable synthetic-measurement generators.

Every function that touches randomness accepts an explicit
`numpy.random.Generator` (`rng=`), so callers own the stream and can pin
reproducibility by building a fresh `default_rng(seed)` before the call.
The previous `seed=` parameter has been retired: pass
`rng=np.random.default_rng(seed)` at the call site instead.
"""

from collections.abc import Callable, Sequence

import numpy as np

from filters._typing import resolve_rng

NoiseFn = Callable[[], float]
Range = Sequence[float]
Vector3 = Sequence[float]
Samples = tuple[np.ndarray, np.ndarray]


def _validate_sample_count(num: int) -> None:
    if num < 0:
        raise ValueError("num must be non-negative")


def _validate_variance(name: str, value: float) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _wrap_heading(particles: np.ndarray) -> np.ndarray:
    particles[:, 2] %= 2 * np.pi
    return particles


def _resolve_noise_fn(random_func: NoiseFn | None, rng: np.random.Generator) -> NoiseFn:
    """Return a zero-argument standard-normal sampler.

    If the caller supplies `random_func` it wins; otherwise we bind to the
    provided Generator (which itself defaults to a fresh one).
    """

    if random_func is not None:
        return random_func
    return rng.standard_normal


def gen_cvca(
    num: int,
    x0: float,
    dx: float,
    ddx: float = 0,
    dt: float = 1,
    R: float = 1,
    random_func: NoiseFn | None = None,
    rng: np.random.Generator | None = None,
) -> Samples:
    """Constant-velocity / constant-acceleration truth + noisy observations."""
    _validate_sample_count(num)
    _validate_variance("R", R)
    sampler = _resolve_noise_fn(random_func, resolve_rng(rng))

    x = x0
    xs, zs = [], []
    noise_std = np.sqrt(R)
    for i in range(num):
        x += dx * dt
        xs.append(x)
        x += ddx * (i**2) / 2 + dx * i
        zs.append(x + sampler() * noise_std)
        dx += ddx
    return np.array(xs), np.asarray(zs)


def gen_jittered_vel(
    num: int,
    x0: float,
    dx: float,
    dt: float = 1,
    Q: float = 0,
    R: float = 1,
    random_func: NoiseFn | None = None,
    rng: np.random.Generator | None = None,
) -> Samples:
    """Constant-velocity trajectory with process noise on the velocity term."""
    _validate_sample_count(num)
    _validate_variance("Q", Q)
    _validate_variance("R", R)
    sampler = _resolve_noise_fn(random_func, resolve_rng(rng))

    x = x0
    xs, zs = [], []
    process_noise_std = np.sqrt(Q)
    measurement_noise_std = np.sqrt(R)
    for _ in range(num):
        x += (dx + (sampler() * process_noise_std)) * dt
        xs.append(x)
        zs.append(x + sampler() * measurement_noise_std)
    return np.array(xs), np.asarray(zs)


def jitterfy(
    center: float | np.ndarray, std: float, rng: np.random.Generator | None = None
) -> float | np.ndarray:
    """Add normally distributed jitter to a scalar or array."""
    _validate_variance("std", std)
    return center + (resolve_rng(rng).standard_normal() * std)


def gen_sensor_data(
    time: int, pos_std: float, vel_std: float, rng: np.random.Generator | None = None
) -> tuple[list[list[float]], list[list[float]]]:
    """Asynchronous position + velocity sensor readings at 3 Hz / 7 Hz."""
    _validate_sample_count(time)
    _validate_variance("pos_std", pos_std)
    _validate_variance("vel_std", vel_std)

    generator = resolve_rng(rng)
    pos_data, vel_data = [], []
    dt = 0.0
    for _ in range(time * 3):
        dt += 1 / 3.0
        time_jittered = dt + generator.standard_normal() * 0.01
        pos_data.append(
            [time_jittered, time_jittered + generator.standard_normal() * pos_std]
        )

    dt = 0.0
    for _ in range(time * 7):
        dt += 1 / 7.0
        time_jittered = dt + generator.standard_normal() * 0.006
        vel_data.append([time_jittered, 1.0 + generator.standard_normal() * vel_std])
    return pos_data, vel_data


def gen_particles_uniform(
    x_range: Range,
    y_range: Range,
    hdg_range: Range,
    N: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Particles drawn uniformly from x/y/heading ranges."""
    _validate_sample_count(N)
    generator = resolve_rng(rng)

    particles = np.empty((N, 3))
    particles[:, 0] = generator.uniform(x_range[0], x_range[1], size=N)
    particles[:, 1] = generator.uniform(y_range[0], y_range[1], size=N)
    particles[:, 2] = generator.uniform(hdg_range[0], hdg_range[1], size=N)
    return _wrap_heading(particles)


def gen_particles_gaussian(
    mean: Vector3, std: Vector3, N: int, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Particles drawn from independent Gaussians per state dimension."""
    _validate_sample_count(N)
    if any(value < 0 for value in std):
        raise ValueError("std values must be non-negative")
    generator = resolve_rng(rng)

    particles = np.empty((N, 3))
    particles[:, 0] = mean[0] + (generator.standard_normal(N) * std[0])
    particles[:, 1] = mean[1] + (generator.standard_normal(N) * std[1])
    particles[:, 2] = mean[2] + (generator.standard_normal(N) * std[2])
    return _wrap_heading(particles)
