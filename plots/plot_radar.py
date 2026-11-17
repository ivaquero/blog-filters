"""Radar and two-sensor plotting helpers.

These lived in ``simulators/radar.py``, which forced every consumer of the
simulator — including headless ones — to import ``matplotlib``. They are pure
presentation and belong here.
"""

import math

import numpy as np

from filters._typing import resolve_rng

from .plot_kf import plot_kf, plot_track

__all__ = [
    "line_intersection",
    "plot_iscts_two_sensors",
    "plot_moving_target",
    "plot_radar3d",
    "plot_radar3d_zs",
    "show_radar_chart",
]

# Column of the 3-state ``[x, x_vel, z]`` estimate selected by ``obj``.
STATE_COLUMNS = {"position": 0, "velocity": 1, "altitude": 2}


def plot_radar3d(axes, time, xs, ylabels=None, track=None):
    """Plot each column of a 3-state estimate against time."""

    xs = np.asarray(xs, dtype=float)
    if ylabels is None:
        ylabels = ["position", "velocity", "altitude"]
    if track is not None:
        track = np.asarray(track, dtype=float)

    for column, (ax, ylabel) in enumerate(zip(axes, ylabels, strict=True)):
        plot_kf(ax, time, ys=xs[:, column], label=f"{ylabel} filtered")
        if track is not None:
            plot_track(ax, time, ys=track[:, column], label=f"{ylabel} track")
        ax.set(xlabel="time", ylabel=ylabel)
        ax.legend(fontsize="small")
        ax.grid(1)
        ax.tick_params(labelsize="small")


def plot_radar3d_zs(ax, time, xs, obj="position", track=None):
    """Plot one state column of the estimate, optionally against the truth.

    Parameters
    ----------
    obj
        ``"position"``, ``"velocity"`` or ``"altitude"``.

    Notes
    -----
    The signature used to take ``ylabels`` and ignore it, and tested
    ``if track:`` — which raises ``ValueError`` for the ``numpy`` array every
    caller passes. Both are gone: the column is chosen by name and the truth is
    tested for ``None``.
    """

    xs = np.asarray(xs, dtype=float)
    column = STATE_COLUMNS[obj]
    ax.plot(time, xs[:, column], label=f"filtered {obj}")
    if track is not None:
        ax.plot(time, track, label="track", lw=2, ls="--", c="k")
    ax.set(xlabel="time", ylabel=obj)
    ax.legend()


def plot_moving_target(ax, rng: np.random.Generator | None = None):
    """Scatter bearing-only measurements of a target moving diagonally."""

    generator = resolve_rng(rng)
    pos = np.array([5.0, 5.0])
    for _ in range(5):
        pos = pos + np.array([0.5, 1.0])
        actual_angle = math.atan2(pos[1], pos[0])
        distance = np.hypot(pos[0], pos[1])

        angles = actual_angle + generator.standard_normal(100) * math.radians(1)
        ax.scatter(distance * np.cos(angles), distance * np.sin(angles), c="C0")

    ax.plot([5.5, pos[0]], [6, pos[1]], c="g", linestyle="--")


def plot_iscts_two_sensors(axes, rng: np.random.Generator | None = None):
    """Show how two bearings intersect at the target's position."""

    generator = resolve_rng(rng)
    positions = [np.array([4.0, 4.0]), np.array([3.0, 3.0])]
    sensor_a = [[0, 2], [3, 4]]
    sensor_b = [[8, 2], [3, 7]]
    counts = [4, 5]

    for ax, pos, sa, sb, count in zip(
        axes, positions, sensor_a, sensor_b, counts, strict=True
    ):
        ax.scatter(*sa, s=200, marker="v", c="k", label="Sensor 1")
        ax.scatter(*sb, s=200, marker="s", c="C0", label="Sensor 2")
        _plot_iscts(ax, pos, sa, sb, num=count, rng=generator)
        ax.legend()


def line_intersection(sa, sb, aa, ab):
    """Intersect the rays leaving ``sa`` along ``aa`` and ``sb`` along ``ab``.

    Vectorised over the angle arrays. Formerly the scalar ``_isct``, which was
    called once per sample inside a 300-iteration Python loop.
    """

    sa = np.asarray(sa, dtype=float)
    sb = np.asarray(sb, dtype=float)
    a1, a2 = np.sin(aa), np.sin(ab)
    b1, b2 = -np.cos(aa), -np.cos(ab)
    c1 = a1 * sa[0] + b1 * sa[1]
    c2 = a2 * sb[0] + b2 * sb[1]

    det = a1 * b2 - a2 * b1
    x = (b2 * c1 - b1 * c2) / det
    y = (a1 * c2 - a2 * c1) / det
    return x, y


def _plot_iscts(ax, pos, sa, sb, *, num=4, rng=None):
    generator = resolve_rng(rng)
    position = np.asarray(pos, dtype=float).copy()
    sa = np.asarray(sa, dtype=float)
    sb = np.asarray(sb, dtype=float)

    for _ in range(num):
        position = position + np.array([0.5, 1.0])
        angle_a = math.atan2(position[1] - sa[1], position[0] - sa[0])
        angle_b = math.atan2(position[1] - sb[1], position[0] - sb[0])
        distance_a = np.hypot(sa[0] - position[0], sa[1] - position[1])
        distance_b = np.hypot(sb[0] - position[0], sb[1] - position[1])

        noise = generator.standard_normal((300, 2)) * math.radians(1)
        aas = angle_a + noise[:, 0]
        bbs = angle_b + noise[:, 1]

        ax.scatter(
            distance_a * np.cos(aas) + sa[0],
            distance_a * np.sin(aas) + sa[1],
            c="k",
            edgecolor="k",
        )
        ax.scatter(
            distance_b * np.cos(bbs) + sb[0],
            distance_b * np.sin(bbs) + sb[1],
            marker="v",
            edgecolor=None,
            c="C0",
        )
        xs, ys = line_intersection(sa, sb, aas, bbs)
        ax.scatter(xs, ys, marker=".", c="r", alpha=0.5)


def show_radar_chart(ax):
    """Annotated sketch of the range/bearing geometry."""

    ax.scatter([1, 2], [1, 2])
    ax.annotate(
        "",
        xy=(2, 2),
        xytext=(1, 1),
        arrowprops={"arrowstyle": "->", "ec": "r", "shrinkA": 3, "shrinkB": 4},
    )
    ax.annotate(
        "",
        xy=(2, 1),
        xytext=(1, 1),
        arrowprops={"arrowstyle": "->", "ec": "b", "shrinkA": 0, "shrinkB": 0},
    )
    ax.annotate(
        "",
        xy=(2, 2),
        xytext=(2, 1),
        arrowprops={"arrowstyle": "->", "ec": "b", "shrinkA": 0, "shrinkB": 4},
    )

    ax.annotate("ϵ", xy=(1.2, 1.05), color="b")
    ax.annotate("Aircraft", xy=(2.04, 2.0), color="b")
    ax.annotate("altitude (y)", xy=(2.04, 1.5), color="k")
    ax.annotate("x", xy=(1.5, 0.9))
    ax.annotate("Radar", xy=(0.95, 0.8))
    ax.annotate("Slant\n  (r)", xy=(1.5, 1.62), color="r")

    ax.set(xlim=[0.9, 2.5], ylim=[0.5, 2.5])
    ax.xaxis.set_ticks([])
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticks([])
    ax.yaxis.set_ticklabels([])
    ax.set(frame_on=False)
