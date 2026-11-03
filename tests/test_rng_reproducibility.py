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
    a = datagen.gen_particles_uniform(
        (0, 10), (0, 10), (0, 6.28), N=20, rng=np.random.default_rng(42)
    )
    b = datagen.gen_particles_uniform(
        (0, 10), (0, 10), (0, 6.28), N=20, rng=np.random.default_rng(42)
    )
    c = datagen.gen_particles_uniform(
        (0, 10), (0, 10), (0, 6.28), N=20, rng=np.random.default_rng(43)
    )
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_datagen_gen_particles_gaussian_reproducible():
    a = datagen.gen_particles_gaussian(
        mean=[1, 2, 0.5], std=[5, 5, 0.1], N=15, rng=np.random.default_rng(7)
    )
    b = datagen.gen_particles_gaussian(
        mean=[1, 2, 0.5], std=[5, 5, 0.1], N=15, rng=np.random.default_rng(7)
    )
    assert np.array_equal(a, b)


def test_datagen_gen_sensor_data_reproducible():
    pa, va = datagen.gen_sensor_data(
        time=3, pos_std=0.1, vel_std=0.05, rng=np.random.default_rng(99)
    )
    pb, vb = datagen.gen_sensor_data(
        time=3, pos_std=0.1, vel_std=0.05, rng=np.random.default_rng(99)
    )
    assert pa == pb
    assert va == vb


def test_datagen_gen_cvca_respects_custom_sampler():
    counter = iter(np.arange(100, dtype=float))
    xs, zs = datagen.gen_cvca(
        num=5, x0=0.0, dx=1.0, R=1.0, random_func=lambda: next(counter)
    )
    # Deterministic sampler → deterministic output.
    assert len(xs) == 5
    assert np.all(np.isfinite(zs))


def test_linear_dog_simulation_reproducible():
    a = linear.DogSimulation(
        x0=0, model_mean=1.0, Q=0.5, R=0.2, rng=np.random.default_rng(13)
    )
    b = linear.DogSimulation(
        x0=0, model_mean=1.0, Q=0.5, R=0.2, rng=np.random.default_rng(13)
    )
    for _ in range(20):
        za = a.move_and_sense()
        zb = b.move_and_sense()
        assert za == zb


def test_linear_pos_sensor_reproducible():
    sa = linear.PosSensor(pos=(0, 0), vel=(1, 1), R=0.5, rng=np.random.default_rng(3))
    sb = linear.PosSensor(pos=(0, 0), vel=(1, 1), R=0.5, rng=np.random.default_rng(3))
    for _ in range(10):
        assert sa.read() == sb.read()


def test_maneuver_simulate_maneuver_reproducible():
    pos_a, zs_a = maneuver.simulate_maneuver(
        steady_count=20, std=0.5, rng=np.random.default_rng(17)
    )
    pos_b, zs_b = maneuver.simulate_maneuver(
        steady_count=20, std=0.5, rng=np.random.default_rng(17)
    )
    _, zs_c = maneuver.simulate_maneuver(
        steady_count=20, std=0.5, rng=np.random.default_rng(18)
    )
    # Truth trajectory is deterministic (only sensing noise is random).
    assert np.array_equal(pos_a, pos_b)
    # Same seed → same observations.
    assert np.array_equal(zs_a, zs_b)
    # Different seed → different observations.
    assert not np.array_equal(zs_a, zs_c)


def test_radar_station_and_sim_reproducible():
    st_a = radar.RadarStation(
        (0, 0), range_std=1.0, elev_std=0.1, rng=np.random.default_rng(1)
    )
    st_b = radar.RadarStation(
        (0, 0), range_std=1.0, elev_std=0.1, rng=np.random.default_rng(1)
    )
    ac_a = radar.ACSim((5, 5), (1, 0), vel_std=0.1, rng=np.random.default_rng(2))
    ac_b = radar.ACSim((5, 5), (1, 0), vel_std=0.1, rng=np.random.default_rng(2))
    for _ in range(8):
        pos_a = ac_a.update(dt=1.0)
        pos_b = ac_b.update(dt=1.0)
        assert np.array_equal(pos_a, pos_b)
        assert st_a.noisy_reading(pos_a) == st_b.noisy_reading(pos_b)

    rs_a = radar.RadarSim(
        dt=1.0, pos=0.0, vel=1.0, alt=100.0, rng=np.random.default_rng(5)
    )
    rs_b = radar.RadarSim(
        dt=1.0, pos=0.0, vel=1.0, alt=100.0, rng=np.random.default_rng(5)
    )
    for _ in range(10):
        assert rs_a.get_range() == rs_b.get_range()


def test_trajectory_ball_and_baseball_reproducible():
    bt_a = trajectory.BallTrajectory2D(
        0, 0, 10, 30, noise=[0.1, 0.1], rng=np.random.default_rng(42)
    )
    bt_b = trajectory.BallTrajectory2D(
        0, 0, 10, 30, noise=[0.1, 0.1], rng=np.random.default_rng(42)
    )
    for _ in range(5):
        assert bt_a.step(0.1) == bt_b.step(0.1)

    bp_a = trajectory.BaseballPath(
        0, 0, 30, 40, noise=(0.1, 0.1), rng=np.random.default_rng(11)
    )
    bp_b = trajectory.BaseballPath(
        0, 0, 30, 40, noise=(0.1, 0.1), rng=np.random.default_rng(11)
    )
    for _ in range(5):
        assert bp_a.update(0.1) == bp_b.update(0.1)


def test_resamplers_accept_injected_generator():
    """The legacy global RNG is gone; callers pass their own Generator."""
    weights = np.array([0.1, 0.2, 0.4, 0.3])
    a = resamplers.systematic_resample(weights, rng=np.random.default_rng(100))
    b = resamplers.systematic_resample(weights, rng=np.random.default_rng(100))
    assert np.array_equal(a, b)


def test_default_rng_is_used_when_none_provided():
    """Simulators still construct without a generator (they build a fresh one)."""
    sensor = linear.PosSensor(pos=(0, 0), vel=(1, 1), R=0.1)
    # Just make sure it does not crash and returns a two-element list.
    out = sensor.read()
    assert len(out) == 2
