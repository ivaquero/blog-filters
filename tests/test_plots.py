"""Regression tests for the plotting helpers (:mod:`plots.plot_common`).

The track helper is the one place where a wrong call shape fails silently:
matplotlib accepts a tuple of arrays as a *single* data argument and then
draws one two-point line per sample, so a broken figure still "renders".
These tests pin the line count and the data both call shapes produce.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb

from plots import plot_common, plot_kf_plus, plot_pf


@pytest.fixture(autouse=True)
def _agg_backend():
    """Keep the suite headless and release figures between tests."""
    plt.switch_backend("Agg")
    yield
    plt.close("all")


def test_plot_track_with_dt_draws_one_line_over_a_time_axis():
    """``dt`` builds the time axis; the result must be a single series."""
    _, ax = plt.subplots()
    xs = np.array([0.0, 1.0, 4.0, 9.0])

    plot_common.plot_track(ax, xs, dt=0.5)

    (line,) = ax.get_lines()
    x_data, y_data = line.get_data()
    assert np.allclose(x_data, [0.0, 0.5, 1.0, 1.5])
    assert np.allclose(y_data, xs)


def test_plot_track_with_explicit_time_and_values_draws_one_line():
    """Passing ``ys`` explicitly must not be read as a column stack."""
    _, ax = plt.subplots()
    ts = np.linspace(0.0, 2.0, 5)
    ys = ts**2

    plot_common.plot_track(ax, ts, ys=ys, label="truth")

    (line,) = ax.get_lines()
    x_data, y_data = line.get_data()
    assert np.allclose(x_data, ts)
    assert np.allclose(y_data, ys)
    assert line.get_label() == "truth"


def test_plot_track_keeps_kwargs_and_style():
    """Forwarded style kwargs land on the artist instead of being dropped."""
    _, ax = plt.subplots()

    plot_common.plot_track(ax, np.arange(3.0), c="red", lw=3, ls="--", alpha=0.4)

    (line,) = ax.get_lines()
    assert line.get_color() == "red"
    assert line.get_linewidth() == 3
    assert line.get_linestyle() == "--"
    assert line.get_alpha() == 0.4


def test_plot_track_accepts_color_keyword():
    """`color` may be forwarded: hard-coding it next to ``**kwargs`` raised."""
    _, ax = plt.subplots()

    plot_common.plot_track(ax, np.arange(3.0), color="steelblue")

    (line,) = ax.get_lines()
    assert line.get_color() == "steelblue"


def _offsets(ax):
    """Scatter offsets as a plain float array (``get_offsets`` is a typed union)."""
    return np.asarray(ax.collections[0].get_offsets(), dtype=float)


def test_plot_zs_prepends_zero_initial_value():
    """``x0=0`` is a real initial state, not a missing argument."""
    _, ax = plt.subplots()

    plot_common.plot_zs(ax, np.array([1.0, 2.0, 3.0]), x0=0)

    offsets = _offsets(ax)
    np.testing.assert_allclose(offsets[:, 0], np.array([0.0, 1.0, 2.0, 3.0]))
    np.testing.assert_allclose(offsets[:, 1], np.array([0.0, 1.0, 2.0, 3.0]))


def test_plot_zs_prepends_sequence_initial_value():
    """List and scalar ``x0`` must produce the same leading sample."""
    _, ax_scalar = plt.subplots()
    _, ax_list = plt.subplots()

    plot_common.plot_zs(ax_scalar, np.array([1.0, 2.0]), x0=0.5)
    plot_common.plot_zs(ax_list, np.array([1.0, 2.0]), x0=[0.5])

    np.testing.assert_allclose(_offsets(ax_scalar), _offsets(ax_list))


def test_plot_zs_rejects_x0_when_ys_is_given():
    """``x0`` has no measurement to attach to once ``ys`` is explicit."""
    _, ax = plt.subplots()

    with pytest.raises(ValueError, match="x0 is only supported"):
        plot_common.plot_zs(ax, np.array([1.0, 2.0]), ys=np.array([1.0, 2.0]), x0=0)


def test_plot_zs_allows_marker_and_color_override():
    """Style keywords are defaults; overriding them must not raise."""
    _, ax = plt.subplots()

    plot_common.plot_zs(ax, np.array([1.0, 2.0]), marker="o", color="crimson")

    face = np.asarray(ax.collections[0].get_facecolor(), dtype=float)[0, :3]
    assert face == pytest.approx(np.asarray(to_rgb("crimson")), abs=1e-3)


def test_prepend_x0_accepts_scalar_and_keeps_float_dtype():
    """A scalar used to fall through the isinstance chain and vanish."""
    out = plot_common.prepend_x0(0.5, np.array([1.0, 2.0]))

    assert isinstance(out, np.ndarray)
    np.testing.assert_allclose(out, [0.5, 1.0, 2.0])


def test_plot_particles_accepts_pixel_marker_without_warning():
    """``","`` is a pixel marker for ``plot`` only; ``scatter`` warns and degrades."""
    _, ax = plt.subplots()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        plot_pf.plot_particles(ax, np.zeros((4, 2)), marker=",")

    assert len(ax.collections) == 1


def test_plot_sigmas_selection_keeps_its_fixed_limits(capsys):
    """``axis("equal")`` used to drop the y limit and print to stderr."""
    _, ax = plt.subplots()

    plot_kf_plus.plot_sigmas_selection(ax)
    ax.figure.canvas.draw()

    assert capsys.readouterr().err == ""
    np.testing.assert_allclose(ax.get_xlim(), (0, 10))
    np.testing.assert_allclose(ax.get_ylim(), (0, 10))
