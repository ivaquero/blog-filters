"""Lock in the RNG-injection contract across simulators and resamplers.

Every entry point that touches randomness now accepts an explicit
``numpy.random.Generator`` (``rng=``). Callers build their own generator with
``np.random.default_rng(seed)`` to pin reproducibility. This file asserts that
contract so a future refactor cannot silently reintroduce nondeterministic
figures or a shared global stream.
"""

from __future__ import annotations

import numpy as np

from filters import resamplers
from simulators import datagen, linear, maneuver, radar, trajectory


def test_datagen_gen_particles_uniform_reproducible():
    kwargs = dict(x_range=(0, 10), y_range=(0, 10), heading_range=(0, 6.28), num=20)
    a = datagen.gen_particles_uniform(**kwargs, rng=np.random.default_rng(42))
    b = datagen.gen_particles_uniform(**kwargs, rng=np.random.default_rng(42))
    c = datagen.gen_particles_uniform(**kwargs, rng=np.random.default_rng(43))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_datagen_gen_particles_gaussian_reproducible():
    kwargs = dict(mean=[1, 2, 0.5], std=[5, 5, 0.1], num=15)
    a = datagen.gen_particles_gaussian(**kwargs, rng=np.random.default_rng(7))
    b = datagen.gen_particles_gaussian(**kwargs, rng=np.random.default_rng(7))
    assert np.array_equal(a, b)


def test_datagen_gen_sensor_data_reproducible():
    kwargs = dict(duration=3, pos_std=0.1, vel_std=0.05)
    pa, va = datagen.gen_async_sensor_data(**kwargs, rng=np.random.default_rng(99))
    pb, vb = datagen.gen_async_sensor_data(**kwargs, rng=np.random.default_rng(99))
    assert pa == pb
    assert va == vb


def test_datagen_gen_cvca_reproducible():
    kwargs = dict(num=5, x0=0.0, vel=1.0, meas_var=1.0)
    a = datagen.gen_cvca(**kwargs, rng=np.random.default_rng(4))
    b = datagen.gen_cvca(**kwargs, rng=np.random.default_rng(4))
    assert np.array_equal(a.x, b.x)
    assert np.array_equal(a.z, b.z)


def test_linear_dog_simulation_reproducible():
    kwargs = dict(x0=0, model_mean=1.0, vel_var=0.5, meas_var=0.2)
    a = linear.DogSimulation(**kwargs, rng=np.random.default_rng(13))
    b = linear.DogSimulation(**kwargs, rng=np.random.default_rng(13))
    for _ in range(20):
        assert a.move_and_sense() == b.move_and_sense()


def test_linear_dog_simulate_reproducible():
    kwargs = dict(x0=0, model_mean=1.0, vel_var=0.5, meas_var=0.2)
    a = linear.DogSimulation(**kwargs, rng=np.random.default_rng(13))
    b = linear.DogSimulation(**kwargs, rng=np.random.default_rng(13))
    first = a.simulate(num=20)
    second = b.simulate(num=20)
    assert np.array_equal(first.x, second.x)
    assert np.array_equal(first.z, second.z)


def test_linear_moving_target_reproducible():
    kwargs = dict(pos=(0, 0), vel=(1, 1), meas_std=0.5)
    a = linear.MovingTarget2D(**kwargs, rng=np.random.default_rng(3))
    b = linear.MovingTarget2D(**kwargs, rng=np.random.default_rng(3))
    for _ in range(10):
        assert np.array_equal(a.sample(), b.sample())


def test_maneuver_simulate_maneuver_reproducible():
    kwargs = dict(steady_count=20, meas_std=0.5)
    a = maneuver.simulate_maneuver(**kwargs, rng=np.random.default_rng(17))
    b = maneuver.simulate_maneuver(**kwargs, rng=np.random.default_rng(17))
    c = maneuver.simulate_maneuver(**kwargs, rng=np.random.default_rng(18))
    # Truth trajectory is deterministic (only sensing noise is random).
    assert np.array_equal(a.x, b.x)
    assert np.array_equal(a.z, b.z)
    assert not np.array_equal(a.z, c.z)


def test_radar_station_and_sim_reproducible():
    station = dict(range_std=1.0, bearing_std=0.1)
    st_a = radar.RadarStation((0, 0), **station, rng=np.random.default_rng(1))
    st_b = radar.RadarStation((0, 0), **station, rng=np.random.default_rng(1))
    ac_a = radar.ACSim((5, 5), (1, 0), vel_std=0.1, rng=np.random.default_rng(2))
    ac_b = radar.ACSim((5, 5), (1, 0), vel_std=0.1, rng=np.random.default_rng(2))
    for _ in range(8):
        pos_a = ac_a.update(dt=1.0)
        pos_b = ac_b.update(dt=1.0)
        assert np.array_equal(pos_a, pos_b)
        assert st_a.reading(pos_a) == st_b.reading(pos_b)

    kwargs = dict(dt=1.0, pos=0.0, vel=1.0, alt=100.0)
    rs_a = radar.RadarSim(**kwargs, rng=np.random.default_rng(5))
    rs_b = radar.RadarSim(**kwargs, rng=np.random.default_rng(5))
    for _ in range(10):
        assert rs_a.measure() == rs_b.measure()


def test_trajectory_ball_and_baseball_reproducible():
    ball = dict(x0=0, y0=0, speed=10, launch_angle_deg=30, meas_std=(0.1, 0.1))
    bt_a = trajectory.BallTrajectory2D(**ball, rng=np.random.default_rng(42))
    bt_b = trajectory.BallTrajectory2D(**ball, rng=np.random.default_rng(42))
    for _ in range(5):
        assert bt_a.step(dt=0.1) == bt_b.step(dt=0.1)

    bat = dict(x0=0, y0=0, speed=40, launch_angle_deg=30, meas_std=(0.1, 0.1))
    bp_a = trajectory.BaseballPath(**bat, rng=np.random.default_rng(11))
    bp_b = trajectory.BaseballPath(**bat, rng=np.random.default_rng(11))
    for _ in range(5):
        assert bp_a.update(dt=0.1) == bp_b.update(dt=0.1)


def test_resamplers_accept_injected_generator():
    """The legacy global RNG is gone; callers pass their own Generator."""
    weights = np.array([0.1, 0.2, 0.4, 0.3])
    a = resamplers.systematic_resample(weights, rng=np.random.default_rng(100))
    b = resamplers.systematic_resample(weights, rng=np.random.default_rng(100))
    assert np.array_equal(a, b)


def test_default_rng_is_used_when_none_provided():
    """Simulators still construct without a generator (they build a fresh one)."""
    target = linear.MovingTarget2D(pos=(0, 0), vel=(1, 1), meas_std=0.1)
    assert target.sample().shape == (2,)
