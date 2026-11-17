"""Semantic regression tests for the simulators.

Reproducibility was already covered; this file covers *correctness*, which is
what let the defects behind the 2026 refactor survive: a simulator that advanced
its state twice per sample, a generator whose measurements were not measurement
of its own truth, a commanded heading that steered to the wrong side, and a
filter that raised ``ZeroDivisionError`` on a straight-line command all passed a
suite that only checked "same seed, same numbers".
"""

from __future__ import annotations

import importlib
import subprocess
import sys
import warnings

import numpy as np
import pytest
import sympy as sy

from filters import robot_ekf
from simulators import Tracks, datagen, linear, maneuver, radar, robot, trajectory


class TestTracks:
    def test_fields_are_index_aligned(self):
        tracks = linear.DogSimulation(vel_var=0.0, meas_var=0.0).simulate(num=4, dt=0.5)
        assert isinstance(tracks, Tracks)
        assert tracks.t.shape == (4,)
        assert tracks.x.shape == (4, 1)
        assert tracks.z.shape == (4, 1)
        assert np.allclose(np.diff(tracks.t), 0.5)

    def test_unpacks_like_a_tuple(self):
        t, x, z = linear.DogSimulation().simulate(num=2)
        assert t.shape == x.shape[:1] == z.shape[:1]

    def test_sensor_less_scenario_reports_empty_channel(self):
        z = maneuver.simulate_turning(num=5).z
        assert z.shape == (5, 0)


class TestDogSimulation:
    def test_simulate_advances_once_per_sample(self):
        """The old version moved the state twice per sample: 5 samples walked 10 steps."""
        dog = linear.DogSimulation(x0=0.0, model_mean=1.0, vel_var=0.0, meas_var=0.0)
        tracks = dog.simulate(num=5)
        assert dog.x == pytest.approx(5.0)
        assert np.allclose(tracks.x.ravel(), [1, 2, 3, 4, 5])

    def test_measurement_tracks_the_truth(self):
        dog = linear.DogSimulation(vel_var=0.0, meas_var=0.0)
        tracks = dog.simulate(num=6)
        assert np.array_equal(tracks.x, tracks.z)

    def test_points_are_columns(self):
        tracks = linear.DogSimulation().simulate(num=3)
        assert tracks.x.shape == (3, 1)
        assert tracks.x[:, 0].shape == (3,)

    def test_acceleration_scales_with_dt(self):
        unit = linear.DogSimulation(model_mean=0.0, vel_var=0.0)
        unit.move(dt=1.0, acc=0.1)
        unit.move(dt=1.0, acc=0.1)
        assert unit.x == pytest.approx(0.1)

        double = linear.DogSimulation(model_mean=0.0, vel_var=0.0)
        double.move(dt=2.0, acc=0.1)
        double.move(dt=2.0, acc=0.1)
        assert double.x == pytest.approx(0.4)

    def test_negative_variance_is_rejected(self):
        with pytest.raises(ValueError, match="vel_var"):
            linear.DogSimulation(vel_var=-1.0)


class TestMovingTarget:
    def test_measure_does_not_advance_the_target(self):
        target = linear.MovingTarget2D(pos=(1.0, 2.0), vel=(3.0, 4.0))
        first = target.measure()
        second = target.measure()
        assert not np.array_equal(first, second)  # noise differs
        assert np.array_equal(target.pos, [1.0, 2.0])

    def test_sample_advances_then_reads(self):
        target = linear.MovingTarget2D(pos=(0.0, 0.0), vel=(1.0, 0.0), meas_std=0.0)
        assert np.array_equal(target.sample(dt=2.0), [2.0, 0.0])


class TestCVCAObjects:
    def test_cv_simulate_matches_step_api(self):
        stepped = linear.CVObject(
            vel_std=0.2, meas_std=0.3, rng=np.random.default_rng(5)
        )
        xs, zs = [], []
        for _ in range(6):
            xs.append(stepped.update())
            zs.append(stepped.sense())

        batched = linear.CVObject(
            vel_std=0.2, meas_std=0.3, rng=np.random.default_rng(5)
        ).simulate(num=6)
        assert np.allclose(batched.x, np.asarray(xs))
        assert np.allclose(batched.z, np.asarray(zs)[:, None])

    def test_ca_simulate_returns_arrays_not_lists(self):
        tracks = linear.CAObject().simulate(num=4)
        assert isinstance(tracks.x, np.ndarray)
        assert isinstance(tracks.z, np.ndarray)
        assert tracks.x.shape == (4, 3)
        assert tracks.z.shape == (4, 1)

    def test_ca_simulate_matches_step_api(self):
        stepped = linear.CAObject(
            acc_std=0.05, meas_std=0.2, rng=np.random.default_rng(9)
        )
        xs, zs = [], []
        for _ in range(6):
            xs.append(stepped.update())
            zs.append(stepped.sense())

        batched = linear.CAObject(
            acc_std=0.05, meas_std=0.2, rng=np.random.default_rng(9)
        ).simulate(num=6)
        assert np.allclose(batched.x, np.asarray(xs))
        assert np.allclose(batched.z, np.asarray(zs)[:, None])


class TestDatagen:
    def test_cvca_measurement_equals_truth_without_noise(self):
        """The old generator drifted: z - x grew by 1.15, 2.60, 4.35 over five steps."""
        tracks = datagen.gen_cvca(num=5, x0=0.0, vel=1.0, acc=0.1, meas_var=0.0)
        assert np.array_equal(tracks.x, tracks.z)

    def test_cvca_matches_closed_form(self):
        tracks = datagen.gen_cvca(num=5, x0=0.0, vel=1.0, acc=0.1, dt=1.0, meas_var=0.0)
        t = np.arange(1, 6)
        assert np.allclose(tracks.x.ravel(), t + 0.5 * 0.1 * t**2)

    def test_jittered_velocity_measurement_equals_truth_without_noise(self):
        tracks = datagen.gen_jittered_vel(
            num=6, x0=0.0, vel=1.0, vel_var=0.0, meas_var=0.0
        )
        assert np.array_equal(tracks.x, tracks.z)

    def test_async_sensor_rates_match_duration(self):
        pos_data, vel_data = datagen.gen_async_sensor_data(
            duration=4.0, pos_std=0.1, vel_std=0.05, pos_rate=3.0, vel_rate=7.0
        )
        assert len(pos_data) == 12
        assert len(vel_data) == 28

    def test_particle_headings_are_wrapped_into_half_open_range(self):
        particles = datagen.gen_particles_uniform(
            x_range=(0, 1), y_range=(0, 1), heading_range=(0, 2 * np.pi), num=200
        )
        assert (particles[:, 2] >= -np.pi).all()
        assert (particles[:, 2] < np.pi).all()


class TestManeuver:
    def test_commanded_heading_is_reached(self):
        target = maneuver.ManeuveringTarget(speed=0.3, heading_deg=0.0)
        target.set_commanded_heading(310.0, 25)
        for _ in range(25):
            target.update()
        assert target.hdg == pytest.approx(-50.0)

    def test_shortest_turn_is_signed_and_bounded(self):
        assert maneuver.heading_delta(310.0, 0.0) == pytest.approx(-50.0)
        assert maneuver.heading_delta(10.0, 350.0) == pytest.approx(20.0)
        assert maneuver.heading_delta(0.0, 720.0) == pytest.approx(0.0)

    def test_heading_stays_wrapped(self):
        target = maneuver.ManeuveringTarget(speed=1.0, heading_deg=0.0)
        target.set_commanded_heading(-170.0, 5)
        for _ in range(40):
            target.update()
        assert -180.0 <= target.hdg < 180.0

    def test_zero_steps_is_a_value_error(self):
        target = maneuver.ManeuveringTarget()
        with pytest.raises(ValueError, match="steps"):
            target.set_commanded_heading(90.0, 0)
        with pytest.raises(ValueError, match="steps"):
            target.set_commanded_speed(1.0, 0)

    def test_compass_convention_is_documented_by_behaviour(self):
        target = maneuver.ManeuveringTarget(speed=1.0, heading_deg=0.0)
        x, y = target.update()
        assert x == pytest.approx(0.0, abs=1e-12)
        assert y == pytest.approx(1.0)

    def test_simulate_maneuver_warmup_is_straight(self):
        tracks = maneuver.simulate_maneuver(steady_count=10, meas_std=0.0, warmup=5)
        assert len(tracks.t) == 15
        assert np.allclose(tracks.x[:5, 0], 0.0)
        assert np.allclose(tracks.x[:5, 1], 0.3 * np.arange(1, 6))

    def test_simulate_turning_returns_flat_state(self):
        tracks = maneuver.simulate_turning(num=20, turn_start=10)
        assert tracks.x.shape == (20, 4)


class TestTrajectory:
    def test_vacuum_projectile_matches_analytic_apex(self):
        ball = trajectory.BallTrajectory2D(
            x0=0.0,
            y0=0.0,
            speed=10.0,
            launch_angle_deg=90.0,
            gravity=10.0,
            meas_std=(0.0, 0.0),
        )
        for _ in range(100):
            y = ball.step(dt=0.01)[1]
        assert y == pytest.approx(5.0)

    def test_both_classes_default_to_a_silent_sensor(self):
        ball = trajectory.BallTrajectory2D(x0=0, y0=0, speed=5, launch_angle_deg=45)
        baseball = trajectory.BaseballPath(x0=0, y0=0, speed=40, launch_angle_deg=30)
        assert ball.meas_std == (0.0, 0.0)
        assert baseball.meas_std == (0.0, 0.0)

        twin = trajectory.BallTrajectory2D(x0=0, y0=0, speed=5, launch_angle_deg=45)
        assert np.array_equal(
            ball.simulate(num=3, dt=0.1).x, twin.simulate(num=3, dt=0.1).z
        )

    def test_drag_force_is_non_negative_and_monotone(self):
        assert trajectory.drag_force(0.0) >= 0.0
        assert trajectory.drag_force(90.0) > trajectory.drag_force(10.0)


class TestRadar:
    def test_state_layouts_are_not_interchangeable(self):
        state = np.array([10.0, 3.0, 20.0, 1.0])
        xy_range, _ = radar.range_bearing_xy(state[:2])
        xz_range, _ = radar.range_bearing_xz(state)
        assert xy_range == pytest.approx(np.hypot(10.0, 3.0))
        assert xz_range == pytest.approx(np.hypot(10.0, 20.0))

    def test_doppler_model_returns_four_values(self):
        distance, _bearing, vx, vz = radar.range_bearing_doppler([10.0, 3.0, 20.0, 1.0])
        assert (vx, vz) == (3.0, 1.0)
        assert distance == pytest.approx(np.hypot(10.0, 20.0))

    def test_slant_range_jacobian_matches_symbolic(self):
        _, jacobian, _ = __import__(
            "symbols.radar", fromlist=["radar3d_symbol"]
        ).radar3d_symbol()
        subs = {sy.Symbol("x"): 7.0, sy.Symbol("x_vel"): -2.0, sy.Symbol("z"): 11.0}
        expected = np.array(jacobian.evalf(subs=subs), dtype=float)
        assert np.allclose(radar.slant_range_jacobian([7.0, -2.0, 11.0]), expected)

    def test_slant_range_matches_hypot(self):
        assert radar.slant_range([3.0, 0.0, 4.0]) == pytest.approx(5.0)

    def test_radar_sim_noise_is_configurable(self):
        quiet = radar.RadarSim(
            dt=1.0,
            pos=0.0,
            vel=0.0,
            alt=100.0,
            vel_std=0.0,
            alt_std=0.0,
            range_frac_std=0.0,
        )
        assert quiet.measure() == pytest.approx(100.0)


class TestRobotEKF:
    def test_straight_line_predict_does_not_raise(self):
        """Regression: the symbolic Jacobian divided by tan(0) on a straight command."""
        ekf = robot_ekf.RobotEKF(1.0, wheelbase=1.0, std_vel=0.1, std_steer=0.05)
        ekf.x = np.zeros((3, 1))
        ekf.P = np.eye(3) * 0.1
        ekf.predict(u=[1.0, 0.0])
        ekf.predict()
        assert np.isfinite(ekf.P).all()
        # one commanded step of 1 m/s for 1 s; the second predict gets u = 0
        assert ekf.x[0, 0] == pytest.approx(1.0)

    def test_jacobians_match_symbolic_arc(self):
        _, f_j, v_j, vars_ = importlib.import_module("symbols.robot").robot3d_symbol()
        a_sym, x_sym, y_sym, v_sym, w_sym, theta_sym, t_sym = vars_

        for angle in (0.05, 0.1, 0.3, 0.5, -0.4, 1.2, -2.0):
            subs = {
                x_sym: 0,
                y_sym: 0,
                v_sym: 1.3,
                a_sym: angle,
                w_sym: 0.8,
                theta_sym: 0.37,
                t_sym: 0.5,
            }
            expected_f = np.array(f_j.evalf(subs=subs), dtype=float)
            expected_v = np.array(v_j.evalf(subs=subs), dtype=float)
            actual_f, actual_v = robot_ekf.transition_jacobians(
                [0.0, 0.0, 0.37], dt=0.5, u=[1.3, angle], wheelbase=0.8
            )
            assert np.allclose(actual_f, expected_f, atol=1e-10)
            assert np.allclose(actual_v, expected_v, atol=1e-9)

    def test_limit_branch_is_continuous_with_the_arc(self):
        tiny = robot_ekf.transition_jacobians(
            [0.0, 0.0, 0.37], dt=0.5, u=[1.3, 1e-5], wheelbase=0.8
        )
        limit = robot_ekf.transition_jacobians(
            [0.0, 0.0, 0.37], dt=0.5, u=[1.3, 0.0], wheelbase=0.8
        )
        assert np.allclose(tiny[0], limit[0], atol=1e-3)
        assert np.allclose(tiny[1], limit[1], atol=1e-3)

    def test_move_matches_the_shared_model(self):
        ekf = robot_ekf.RobotEKF(0.1, wheelbase=0.5, std_vel=0.1, std_steer=0.05)
        expected = robot_ekf.robot_move(
            [1.0, 2.0, 0.3], dt=0.1, u=[1.0, 0.2], wheelbase=0.5
        )
        assert np.allclose(
            ekf.move(np.array([[1.0], [2.0], [0.3]]), [1.0, 0.2], 0.1).ravel(), expected
        )

    def test_process_noise_does_not_vanish_at_zero_speed(self):
        ekf = robot_ekf.RobotEKF(1.0, wheelbase=1.0, std_vel=0.1, std_steer=0.05)
        ekf.x = np.zeros((3, 1))
        ekf.P = np.zeros((3, 3))
        ekf.predict(u=[0.0, 0.0])
        assert np.trace(ekf.P) > 0.0


class TestRobotScenario:
    def test_particle_filter_returns_data_instead_of_plotting(self):
        run = robot.simulate_robot_pf(
            num_particles=200,
            iters=4,
            keep_particles=True,
            rng=np.random.default_rng(2),
        )
        assert run.x_true.shape == (4, 2)
        assert run.x_est.shape == (4, 2)
        assert run.particles is not None
        assert len(run.particles) == 5  # prior cloud + one per step

    def test_landmark_reading_is_wrapped(self):
        reading = robot.landmark_reading(
            np.array([0.0, 1.0]),
            np.array([0.0, 0.0, np.pi - 1e-3]),
            range_std=0.0,
            bearing_std=0.0,
        )
        assert -np.pi <= reading[1] < np.pi

    def test_normalize_angle_range(self):
        assert robot.normalize_angle(-np.pi) == pytest.approx(-np.pi)
        assert robot.normalize_angle(np.pi) == pytest.approx(-np.pi)
        assert robot.normalize_angle(3 * np.pi) == pytest.approx(-np.pi)

    def test_robot_turn_returns_array_in_radians(self):
        commands = robot.robot_turn(speed=1.0, start_deg=0.0, end_deg=2.0, steps=3)
        assert commands.shape == (3, 2)
        assert commands[0, 1] == pytest.approx(0.0)
        assert commands[-1, 1] == pytest.approx(np.radians(2.0))


class TestLayering:
    def test_importing_a_simulator_does_not_import_plotting_or_symbolic_layers(self):
        code = (
            "import sys; import simulators.datagen;"
            "print('matplotlib' in sys.modules, 'sympy' in sys.modules)"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        assert result.stdout.strip() == "False False"

    def test_retired_names_warn_before_resolving(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            alias = linear.PosSensor
        assert alias is linear.MovingTarget2D
        assert any(issubclass(item.category, DeprecationWarning) for item in caught)

    def test_retired_radar_plots_forward_to_the_plot_module(self):
        from plots import plot_radar

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            forwarded = radar.plot_radar3d
        assert forwarded is plot_radar.plot_radar3d
