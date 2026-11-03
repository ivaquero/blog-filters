from collections.abc import Callable, Sequence

import numpy as np

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


def _resolve_noise_fn(random_func: NoiseFn | None) -> NoiseFn:
    """Return a zero-argument standard-normal sampler.

    If the caller does not supply ``random_func`` we draw a fresh
    ``numpy.random.Generator`` so behaviour is reproducible-per-call rather than
    relying on a shared global state.
    """
    if random_func is not None:
        return random_func
    rng = np.random.default_rng()
    return rng.standard_normal


def gen_cvca(
    num: int,
    x0: float,
    dx: float,
    ddx: float = 0,
    dt: float = 1,
    R: float = 1,
    random_func: NoiseFn | None = None,
) -> Samples:
    """Generate samples from a constant velocity and constant acceleration model.

    Args:
        num: Number of samples.
        x0: Initial state.
        dx: Velocity.
        ddx: Acceleration. Defaults to 0.
        dt: Time step. Defaults to 1.
        R: Measurement noise variance. Defaults to 1.
        random_func: Optional zero-argument standard normal sampler. When None
            a fresh ``numpy.random.Generator.standard_normal`` is used.

    Returns:
        Truth values and noisy measurements.
    """
    _validate_sample_count(num)
    _validate_variance("R", R)
    sampler = _resolve_noise_fn(random_func)

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
) -> Samples:
    """Generate samples with process noise added to a constant velocity model.

    Args:
        num: Number of samples.
        x0: Initial state.
        dx: Base velocity.
        dt: Time step. Defaults to 1.
        Q: Process noise variance. Defaults to 0.
        R: Measurement noise variance. Defaults to 1.
        random_func: Optional zero-argument standard normal sampler. When None
            a fresh ``numpy.random.Generator.standard_normal`` is used.

    Returns:
        Truth values and noisy measurements.
    """
    _validate_sample_count(num)
    _validate_variance("Q", Q)
    _validate_variance("R", R)
    sampler = _resolve_noise_fn(random_func)

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
    """Add normally distributed jitter to a scalar or array.

    Args:
        center: Original value or array.
        std: Standard deviation of the jitter.
        rng: Optional ``numpy.random.Generator``; a fresh one is created when
            omitted so callers can pin reproducibility by passing their own.

    Returns:
        Jittered value or array.
    """
    _validate_variance("std", std)
    if rng is None:
        rng = np.random.default_rng()
    return center + (rng.standard_normal() * std)


def gen_sensor_data(
    time: int, pos_std: float, vel_std: float, seed: int | None = 1123
) -> tuple[list[list[float]], list[list[float]]]:
    """Generate asynchronous position and velocity sensor readings."""
    _validate_sample_count(time)
    _validate_variance("pos_std", pos_std)
    _validate_variance("vel_std", vel_std)

    rng = np.random.default_rng(seed)
    pos_data, vel_data = [], []
    dt = 0.0
    for _ in range(time * 3):
        dt += 1 / 3.0
        time_jittered = dt + rng.standard_normal() * 0.01
        pos_data.append(
            [time_jittered, time_jittered + rng.standard_normal() * pos_std]
        )

    dt = 0.0
    for _ in range(time * 7):
        dt += 1 / 7.0
        time_jittered = dt + rng.standard_normal() * 0.006
        vel_data.append([time_jittered, 1.0 + rng.standard_normal() * vel_std])
    return pos_data, vel_data


def gen_particles_uniform(
    x_range: Range, y_range: Range, hdg_range: Range, N: int, seed: int | None = None
) -> np.ndarray:
    """Generate particles uniformly over x, y, and heading ranges."""
    _validate_sample_count(N)
    rng = np.random.default_rng(seed)

    particles = np.empty((N, 3))
    particles[:, 0] = rng.uniform(x_range[0], x_range[1], size=N)
    particles[:, 1] = rng.uniform(y_range[0], y_range[1], size=N)
    particles[:, 2] = rng.uniform(hdg_range[0], hdg_range[1], size=N)
    return _wrap_heading(particles)


def gen_particles_gaussian(
    mean: Vector3, std: Vector3, N: int, seed: int | None = None
) -> np.ndarray:
    """Generate particles from independent Gaussian dimensions."""
    _validate_sample_count(N)
    if any(value < 0 for value in std):
        raise ValueError("std values must be non-negative")
    rng = np.random.default_rng(seed)

    particles = np.empty((N, 3))
    particles[:, 0] = mean[0] + (rng.standard_normal(N) * std[0])
    particles[:, 1] = mean[1] + (rng.standard_normal(N) * std[1])
    particles[:, 2] = mean[2] + (rng.standard_normal(N) * std[2])
    return _wrap_heading(particles)
