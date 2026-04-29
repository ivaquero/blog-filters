from collections.abc import Callable, Sequence

import numpy as np
from numpy import random

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


def gen_cvca(
    num: int,
    x0: float,
    dx: float,
    ddx: float = 0,
    dt: float = 1,
    R: float = 1,
    random_func: NoiseFn = random.randn,
) -> Samples:
    """Generate samples from a constant velocity and constant acceleration model.

    Args:
        num: Number of samples.
        x0: Initial state.
        dx: Velocity.
        ddx: Acceleration. Defaults to 0.
        dt: Time step. Defaults to 1.
        R: Measurement noise variance. Defaults to 1.
        random_func: Function that returns a standard normal random value.

    Returns:
        Truth values and noisy measurements.
    """
    _validate_sample_count(num)
    _validate_variance("R", R)

    x = x0
    xs, zs = [], []
    noise_std = np.sqrt(R)
    for i in range(num):
        x += dx * dt
        xs.append(x)
        x += ddx * (i**2) / 2 + dx * i
        zs.append(x + random_func() * noise_std)
        dx += ddx
    return np.array(xs), np.asarray(zs)


def gen_jittered_vel(
    num: int,
    x0: float,
    dx: float,
    dt: float = 1,
    Q: float = 0,
    R: float = 1,
    random_func: NoiseFn = random.randn,
) -> Samples:
    """Generate samples with process noise added to a constant velocity model.

    Args:
        num: Number of samples.
        x0: Initial state.
        dx: Base velocity.
        dt: Time step. Defaults to 1.
        Q: Process noise variance. Defaults to 0.
        R: Measurement noise variance. Defaults to 1.
        random_func: Function that returns a standard normal random value.

    Returns:
        Truth values and noisy measurements.
    """
    _validate_sample_count(num)
    _validate_variance("Q", Q)
    _validate_variance("R", R)

    x = x0
    xs, zs = [], []
    process_noise_std = np.sqrt(Q)
    measurement_noise_std = np.sqrt(R)
    for _ in range(num):
        x += (dx + (random_func() * process_noise_std)) * dt
        xs.append(x)
        zs.append(x + random_func() * measurement_noise_std)
    return np.array(xs), np.asarray(zs)


def jitterfy(center: float | np.ndarray, std: float) -> float | np.ndarray:
    """Add normally distributed jitter to a scalar or array.

    Args:
        center: Original value or array.
        std: Standard deviation of the jitter.

    Returns:
        Jittered value or array.
    """
    _validate_variance("std", std)
    return center + (random.randn() * std)


def gen_sensor_data(
    time: int, pos_std: float, vel_std: float, seed: int = 1123
) -> tuple[list[list[float]], list[list[float]]]:
    """Generate asynchronous position and velocity sensor readings."""
    _validate_sample_count(time)
    _validate_variance("pos_std", pos_std)
    _validate_variance("vel_std", vel_std)

    random.seed(seed)
    pos_data, vel_data = [], []
    dt = 0.0
    for _ in range(time * 3):
        dt += 1 / 3.0
        time_jittered = dt + random.randn() * 0.01
        pos_data.append([time_jittered, time_jittered + random.randn() * pos_std])

    dt = 0.0
    for _ in range(time * 7):
        dt += 1 / 7.0
        time_jittered = dt + random.randn() * 0.006
        vel_data.append([time_jittered, 1.0 + random.randn() * vel_std])
    return pos_data, vel_data


def gen_particles_uniform(
    x_range: Range, y_range: Range, hdg_range: Range, N: int
) -> np.ndarray:
    """Generate particles uniformly over x, y, and heading ranges."""
    _validate_sample_count(N)

    particles = np.empty((N, 3))
    particles[:, 0] = random.uniform(x_range[0], x_range[1], size=N)
    particles[:, 1] = random.uniform(y_range[0], y_range[1], size=N)
    particles[:, 2] = random.uniform(hdg_range[0], hdg_range[1], size=N)
    return _wrap_heading(particles)


def gen_particles_gaussian(mean: Vector3, std: Vector3, N: int) -> np.ndarray:
    """Generate particles from independent Gaussian dimensions."""
    _validate_sample_count(N)
    if any(value < 0 for value in std):
        raise ValueError("std values must be non-negative")

    particles = np.empty((N, 3))
    particles[:, 0] = mean[0] + (random.randn(N) * std[0])
    particles[:, 1] = mean[1] + (random.randn(N) * std[1])
    particles[:, 2] = mean[2] + (random.randn(N) * std[2])
    return _wrap_heading(particles)
