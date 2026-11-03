"""Smoother visualisations (RTS over a constant-velocity track)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from filters._typing import resolve_rng
from filters.kalman import KalmanFilter

from .plot_common import plot_zs

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def plot_rts(
    ax: Axes,
    R: float,
    Q: float = 0.001,
    rng: np.random.Generator | None = None,
    *,
    show_velocity: bool = False,
) -> None:
    """Filter and smooth a synthetic constant-velocity track."""
    generator = resolve_rng(rng)
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x = np.array([0.0, 1.0])
    kf.F = np.array([[1.0, 1.0], [0.0, 1.0]])
    kf.H = np.array([[1.0, 0.0]])
    kf.P = 10.0 * np.eye(len(kf.x))
    kf.R = np.array([[R]])
    kf.Q = Q * np.eye(len(kf.x))
    # create noisy data
    zs = np.asarray([t + generator.standard_normal() * R for t in range(40)])
    # filter data with Kalman filter, than run smoother on it
    means, covs, _, _ = kf.batch_filter(zs)
    smoothed, _, _, _ = kf.rts_smoother(means, covs)
    # plot data
    if show_velocity:
        index = 1
    else:
        index = 0
        plot_zs(ax, zs, lw=1)
        N = len(zs)
        ax.plot([0, N], [0, N], "k", lw=2, label="track")

    ax.plot(smoothed[:, index], c="b", label="RTS")
    ax.plot(means[:, index], c="g", ls="--", label="KF output")

    ax.legend()
