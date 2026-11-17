"""Tests for the state-space model contract and its EKF/UKF entry points.

Three things are asserted here:

1. the model protocol is satisfied structurally by every concrete model, so
   ``from_model`` needs no inheritance from the ``models`` package;
2. an EKF driven by a model reproduces the linear Kalman filter on a linear
   plant, which is the only way to prove the ``fx``/``jacobian_x`` wiring is
   consistent;
3. the coordinated-turn Jacobian stays correct as ``ω → 0``. It used to lose
   two entire derivatives once ``ω * dt`` dropped below ``1e-8`` because
   ``1 - cos θ`` cancels to zero, giving a 100% error on ``dJ/dω`` for a target
   flying straight.
"""

from __future__ import annotations

import numpy as np
import pytest

from filters._typing import StateSpaceModelLike
from filters.kalman import KalmanFilter
from filters.kalman_ekf import ExtendedKalmanFilter
from filters.kalman_ukf import UnscentedKalmanFilter
from filters.sigma_points import MerweScaledSigmas
from models import ConstantAcceleration, ConstantVelocity, CoordinatedTurn, LinearModel
from models.noise import white_noise_discrete
from models.ssmodel_nonlinear import coordinated_turn_Jfx, coordinated_turn_fx


def _cv_measurements(steps: int = 40, dt: float = 0.1, rng=None):
    """Straight-line truth with position-only measurements."""

    rng = np.random.default_rng(11) if rng is None else rng
    times = np.arange(steps) * dt
    truth = np.column_stack([2.0 + 3.0 * times, 1.0 + 0.5 * times])
    return truth, truth[:, 0] + rng.standard_normal(steps) * 0.3


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model", [ConstantVelocity(), ConstantAcceleration(), CoordinatedTurn()]
)
def test_models_satisfy_the_protocol_structurally(model):
    assert isinstance(model, StateSpaceModelLike)
    assert model.ndim == {
        "x": model.dim_x,
        "z": model.dim_z,
        "u": model.dim_u,
        "w": model.dim_w,
        "v": model.dim_v,
    }


@pytest.mark.parametrize("axes", [1, 2, 3])
def test_constant_velocity_dimensions_follow_axes(axes):
    """``dim_x``/``dim_z`` and the matrices must agree for every axis count.

    The pre-refactor class exposed a hard-coded ``NDIM = {"x": 4, "z": 2, ...}``
    next to the real ``ndim``, so a three-axis model reported two different
    state dimensions depending on which attribute you read.
    """

    model = ConstantVelocity(dt=0.1, axes=axes)
    assert model.dim_x == 2 * axes
    assert model.dim_z == axes
    assert model.F.shape == (2 * axes, 2 * axes)
    assert model.H.shape == (axes, 2 * axes)
    assert model.process_cov().shape == (2 * axes, 2 * axes)
    assert not hasattr(model, "NDIM")


def test_state_equation_defaults_to_noise_free():
    """The abstract signature used to default ``w=0``, which no subclass accepted."""

    model = ConstantVelocity(dt=0.1)
    state = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.allclose(model.state_equation(0.0, state), model.F @ state)
    assert np.allclose(model.fx(state), model.F @ state)


def test_ukf_compatible_pointwise_signatures():
    """``fx``/``hx`` must accept exactly what the UKF passes."""

    model = ConstantVelocity(dt=0.1)
    state = np.array([1.0, 2.0, 3.0, 4.0])
    assert model.fx(state, 0.1).shape == (4,)
    assert model.hx(state).shape == (2,)
    assert model.jacobian_x(state).shape == (4, 4)
    assert model.jacobian_h(state).shape == (2, 4)


def test_linear_model_built_from_matrices_matches_constant_velocity():
    dt = 0.1
    model = ConstantVelocity(dt=dt, axes=1)
    by_matrix = LinearModel(
        F=[[1.0, dt], [0.0, 1.0]], H=[[1.0, 0.0]], L=[[0.5 * dt**2], [dt]]
    )
    state = np.array([3.0, -1.5])
    assert np.allclose(by_matrix.fx(state), model.fx(state))
    assert np.allclose(by_matrix.hx(state), model.hx(state))
    assert np.allclose(by_matrix.jacobian_x(state), model.jacobian_x(state))
    assert np.allclose(by_matrix.process_cov(), model.process_cov())
    assert by_matrix.dim_x == 2
    assert by_matrix.dim_z == 1


# ---------------------------------------------------------------------------
# Guards that replaced silent failures
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_dim", [1, 5])
def test_discrete_noise_message_matches_implementation(bad_dim):
    """``dim=1`` has no branch, so the error must not advertise it."""

    with pytest.raises(ValueError, match=r"dim must be 2, 3 or 4"):
        white_noise_discrete(dim=bad_dim)


def test_flat_covariance_is_rejected_instead_of_promoted():
    """``np.atleast_2d`` used to turn a length-4 vector into a ``(1, 4)`` matrix."""

    from models.ssmodel_linear import constant_velocity_filter

    with pytest.raises(ValueError, match=r"\(4, 4\) matrix"):
        constant_velocity_filter(P=[1.0, 2.0, 3.0, 4.0], R=1.0, dt=0.1, x=(0, 0, 0, 0))


def test_constant_velocity_filter_2d_measures_every_axis():
    """A four-element state must not silently fall back to one measurement."""

    from models.ssmodel_linear import constant_velocity_filter_2d

    kf = constant_velocity_filter_2d(P=1.0, R=1.0, Q=0.1, dt=0.1, x=(0, 0, 0, 0))
    assert kf.dim_x == 4
    assert kf.dim_z == 2
    assert np.allclose(kf.H, [[1, 0, 0, 0], [0, 0, 1, 0]])


# ---------------------------------------------------------------------------
# Coordinated-turn numerics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("turn_rate", "rtol"),
    [(0.0, 1e-12), (1e-12, 1e-9), (1e-9, 1e-9), (1e-7, 1e-6), (1e-5, 1e-4)],
)
def test_coordinated_turn_jacobian_holds_the_zero_rate_limit(turn_rate, rtol):
    """``dJ/dω`` entries must converge to their analytic limit, not to zero.

    The tolerance tightens with ``ω`` because the residual is the genuine
    ``O(ω dt)`` term of the exact derivative, not numerical noise. Below
    ``ω = 1e-7`` the old implementation returned twice the true value (the
    ``1 - cos θ`` term had already underflowed to zero).
    """

    dt = 0.1
    state = np.array([10.0, 3.0, -5.0, 2.0, turn_rate])
    jacobian = coordinated_turn_Jfx(state, dt)

    assert jacobian[0, 4] == pytest.approx(-0.5 * dt**2 * 2.0, rel=rtol, abs=0)
    assert jacobian[2, 4] == pytest.approx(0.5 * dt**2 * 3.0, rel=rtol, abs=0)
    assert jacobian[0, 1] == pytest.approx(dt, rel=1e-9)
    # (1 - cos θ)/ω → ω dt²/2, up to the O(θ²) correction; the old
    # implementation returned exactly 0 here.
    assert jacobian[0, 3] == pytest.approx(-0.5 * dt**2 * turn_rate, rel=1e-6, abs=0)
    assert jacobian[2, 1] == pytest.approx(0.5 * dt**2 * turn_rate, rel=1e-6, abs=0)
    assert np.isfinite(jacobian).all()


def test_coordinated_turn_jacobian_matches_finite_differences():
    """Away from the singular limit the analytic form must still be right."""

    dt = 0.1
    state = np.array([10.0, 3.0, -5.0, 2.0, 0.3])
    analytic = coordinated_turn_Jfx(state, dt)

    step = 1e-6
    numeric = np.zeros((5, 5))
    for index in range(5):
        perturbation = np.zeros(5)
        perturbation[index] = step
        forward = coordinated_turn_fx(state + perturbation, dt)
        backward = coordinated_turn_fx(state - perturbation, dt)
        numeric[:, index] = (forward - backward) / (2 * step)

    assert np.allclose(analytic, numeric, rtol=1e-5, atol=1e-7)


def test_coordinated_turn_transition_holds_the_zero_rate_limit():
    dt = 0.1
    zero_rate = coordinated_turn_fx(np.array([1.0, 3.0, 2.0, -1.0, 0.0]), dt)
    for turn_rate in (1e-12, 1e-9, 1e-8):
        near_zero = coordinated_turn_fx(np.array([1.0, 3.0, 2.0, -1.0, turn_rate]), dt)
        # The cross-coupling is O(ω dt²), so the states agree to that order.
        assert np.allclose(near_zero, zero_rate, atol=1e-7)


# ---------------------------------------------------------------------------
# End-to-end: filters driven by models
# ---------------------------------------------------------------------------


def test_ekf_from_model_reproduces_the_linear_kalman_filter():
    dt, q, r = 0.1, 0.01, 0.09
    _, measurements = _cv_measurements(dt=dt)

    model = ConstantVelocity(dt=dt, axes=1, process_var=q, measurement_var=r)

    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x = np.array([0.0, 0.0])
    kf.P = np.eye(2)
    kf.F = model.transition_matrix(0.0)
    kf.H = model.measurement_matrix(0.0)
    kf.Q = model.process_cov()
    kf.R = model.measurement_cov()

    ekf = ExtendedKalmanFilter.from_model(model, x0=[0.0, 0.0], P0=np.eye(2))

    for z in measurements:
        kf.predict()
        kf.update([z])

        ekf.predict()
        ekf.update(np.array([[z]]))

    assert np.allclose(kf.x, ekf.x.reshape(-1), atol=1e-10)
    assert np.allclose(kf.P, ekf.P, atol=1e-10)


def test_ukf_from_model_tracks_a_constant_velocity_target():
    dt = 0.1
    truth, measurements = _cv_measurements(dt=dt, rng=np.random.default_rng(5))

    model = ConstantVelocity(dt=dt, axes=1, process_var=0.01, measurement_var=0.25)
    points = MerweScaledSigmas(
        n=model.dim_x, alpha=1e-3, beta=2.0, kappa=3 - model.dim_x
    )
    ukf = UnscentedKalmanFilter.from_model(model, points)
    ukf.x = np.array([0.0, 0.0])
    ukf.P = np.eye(model.dim_x) * 10.0

    for z in measurements:
        ukf.predict()
        ukf.update(np.array([z]))

    assert np.allclose(ukf.Q, model.process_cov())
    assert np.allclose(ukf.R, model.measurement_cov())
    assert abs(ukf.x[0] - truth[-1, 0]) < 0.5
    assert ukf.x[1] == pytest.approx(3.0, abs=0.2)


def test_ekf_from_model_tracks_a_coordinated_turn():
    """The EKF must stay locked onto a turning target through the model path."""

    dt = 0.05
    turn_rate = 0.2
    model = CoordinatedTurn(dt=dt, process_var=1e-4, measurement_var=0.04)

    state = np.array([0.0, 5.0, 0.0, 0.0, turn_rate])
    truth = []
    for _ in range(120):
        state = model.fx(state)
        truth.append(state[[0, 2]].copy())
    truth = np.array(truth)

    ekf = ExtendedKalmanFilter.from_model(
        model,
        x0=[0.0, 5.0, 0.0, 0.0, turn_rate + 0.05],
        P0=np.diag([1.0, 1.0, 1.0, 1.0, 0.01]),
    )
    for position in truth:
        ekf.predict()
        ekf.update(position.reshape(2, 1))

    assert np.allclose(ekf.x[[0, 2]].ravel(), truth[-1], atol=0.15)
    assert ekf.x[4, 0] == pytest.approx(turn_rate, abs=0.02)


def test_ekf_update_without_model_still_requires_callables():
    ekf = ExtendedKalmanFilter(dim_x=2, dim_z=1)
    with pytest.raises(TypeError, match="HJac and Hx are required"):
        ekf.update(np.array([[1.0]]))


def test_attach_model_rejects_dimension_mismatch():
    ekf = ExtendedKalmanFilter(dim_x=3, dim_z=1)
    with pytest.raises(ValueError, match="do not match"):
        ekf.attach_model(ConstantVelocity(dt=0.1, axes=1))


# ---------------------------------------------------------------------------
# Batch propagation (S5)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model",
    [ConstantVelocity(dt=0.1), ConstantAcceleration(dt=0.1), CoordinatedTurn(dt=0.1)],
)
def test_fx_many_matches_pointwise(model):
    r"""``fx_many`` must be numerically identical to looping over ``fx``."""

    rng = np.random.default_rng(7)
    batch = rng.standard_normal((50, model.dim_x))
    point = np.vstack([model.fx(row) for row in batch])
    assert np.allclose(model.fx_many(batch), point, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize(
    "model",
    [ConstantVelocity(dt=0.1), ConstantAcceleration(dt=0.1), CoordinatedTurn(dt=0.1)],
)
def test_hx_many_matches_pointwise(model):
    r"""``hx_many`` must be numerically identical to looping over ``hx``."""

    rng = np.random.default_rng(9)
    batch = rng.standard_normal((50, model.dim_x))
    point = np.vstack([model.hx(row) for row in batch])
    assert np.allclose(model.hx_many(batch), point, rtol=1e-12, atol=1e-14)


def test_single_vector_is_promoted_to_one_row():
    model = ConstantVelocity(dt=0.1)
    state = np.array([1.0, 2.0, 3.0, 4.0])
    assert model.fx_many(state).shape == (1, model.dim_x)
    assert np.allclose(model.fx_many(state)[0], model.fx(state))
    assert model.hx_many(state).shape == (1, model.dim_z)


def test_batch_rejects_bad_column_count_and_rank():
    model = ConstantVelocity(dt=0.1)
    with pytest.raises(ValueError, match="4 columns"):
        model.fx_many(np.zeros((5, 3)))
    with pytest.raises(ValueError, match="1-D or 2-D"):
        model.fx_many(np.zeros((5, 4, 1)))


def test_batched_and_pointwise_ukf_agree_exactly():
    """The UKF batch path is an optimisation, not a different estimator."""

    _, measurements = _cv_measurements(dt=0.1, rng=np.random.default_rng(21))

    def build(force_pointwise):
        model = ConstantVelocity(dt=0.1, axes=1, process_var=0.01, measurement_var=0.25)
        points = MerweScaledSigmas(n=model.dim_x, alpha=1e-3, beta=2.0, kappa=1)
        ukf = UnscentedKalmanFilter.from_model(model, points)
        ukf.x = np.array([0.0, 0.0])
        ukf.P = np.eye(model.dim_x) * 10.0
        if force_pointwise:
            # Break the "bound to model" identity so the pointwise path is used.
            ukf.model = None
        return ukf

    fast, slow = build(False), build(True)
    for z in measurements:
        fast.predict()
        fast.update(np.array([z]))
        slow.predict()
        slow.update(np.array([z]))

    # Matrix-product vs per-vector accumulation order differs in the last bits;
    # over 40 filtering steps that rounding noise grows to ~3e-9 in the state.
    assert np.allclose(fast.x, slow.x, rtol=1e-7, atol=1e-9)
    assert np.allclose(fast.P, slow.P, rtol=1e-8, atol=1e-12)
    assert np.allclose(fast.sigmas_f, slow.sigmas_f, rtol=1e-7, atol=1e-9)
    assert np.allclose(fast.sigmas_h, slow.sigmas_h, rtol=1e-7, atol=1e-9)


def test_custom_fx_callable_keeps_pointwise_path():
    """A caller-supplied ``fx`` must never be routed through the model batch."""

    model = ConstantVelocity(dt=0.1, axes=1)
    points = MerweScaledSigmas(n=model.dim_x, alpha=1e-3, beta=2.0, kappa=1)
    ukf = UnscentedKalmanFilter.from_model(model, points)

    calls = {"n": 0}

    def counting_fx(x, dt, **kwargs):
        calls["n"] += 1
        return model.fx(x, dt, **kwargs)

    ukf.predict(fx=counting_fx)
    assert calls["n"] == points.num_sigmas()
