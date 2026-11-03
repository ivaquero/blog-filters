"""Particle-filter plotting helpers.

Randomised visuals accept a caller-supplied ``numpy.random.Generator`` so
notebooks can pin reproducibility without touching the global numpy RNG.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colorbar

from filters._typing import resolve_rng
from filters.resamplers import residual_resample

if TYPE_CHECKING:
    from matplotlib.axes import Axes

__all__ = [
    "plot_hbar",
    "plot_particles",
    "plot_pf",
    "show_resample_multinomial",
    "show_resample_residual",
    "show_resample_stratified",
    "show_resample_systematic",
]


def plot_particles(
    ax: Axes,
    particles: np.ndarray,
    marker: str = "o",
    markersize: float = 1,
    color: str = "g",
) -> None:
    N = len(particles)
    alpha = 0.20
    if N > 5000:
        alpha *= np.sqrt(5000) / np.sqrt(N)
    ax.scatter(
        particles[:, 0],
        particles[:, 1],
        alpha=alpha,
        marker=marker,
        s=markersize,
        color=color,
    )


def plot_pf(
    axes, pf, xlim: float = 100, ylim: float = 100, *, weights: bool = True
) -> None:
    if weights:
        axes[1].set(yticklabels="", xlim=(0, ylim), ylim=(0, xlim))
        axes[1].scatter(pf.particles[:, 0], pf.weights, marker=".", s=1, color="k")

        axes[2].set(xticklabels="", xlim=(0, ylim), ylim=(0, xlim))
        axes[2].scatter(pf.weights, pf.particles[:, 1], marker=".", s=1, color="k")

    axes[0].scatter(pf.particles[:, 0], pf.particles[:, 1], marker=".", s=1, color="k")
    axes[0].set(xlim=(0, ylim), ylim=(0, xlim))


def plot_hbar(a, N: int, figsize):
    cmap = mpl.colors.ListedColormap(
        [[0.0, 0.4, 1.0], [0.0, 0.8, 1.0], [1.0, 0.8, 0.0], [1.0, 0.4, 0.0]]
        * (N // 4 + 1)
    )
    cumsum = np.cumsum(np.asarray(a) / np.sum(a))
    cumsum = np.insert(cumsum, 0, 0)
    norm = mpl.colors.BoundaryNorm(cumsum, cmap.N)

    _ = plt.figure(figsize=figsize)
    ax = plt.gcf().add_axes((0.01, 0.01, 1, 0.2))
    bar = colorbar.ColorbarBase(
        ax,
        cmap=cmap,
        norm=norm,
        drawedges=False,
        spacing="proportional",
        orientation="horizontal",
    )
    bar.set_ticks([])
    return ax, cumsum


def show_resample_multinomial(
    a, figsize: tuple[float, float] = (6, 2), rng: np.random.Generator | None = None
) -> None:
    generator = resolve_rng(rng)
    N = len(a)
    ax, _ = plot_hbar(a, N, figsize=figsize)

    # make N subdivisions, and chose a random position within each one
    b = generator.random(N)
    ax.scatter(b, [0.5] * len(b), s=60, facecolor="k", edgecolor="k")
    ax.set(title="multinomial resampling")


def show_resample_stratified(
    a, figsize: tuple[float, float] = (6, 2), rng: np.random.Generator | None = None
) -> None:
    generator = resolve_rng(rng)
    N = len(a)
    ax, _ = plot_hbar(a, N, figsize=figsize)

    xs = np.linspace(0.0, 1.0 - 1.0 / N, N)
    ax.vlines(xs, 0, 1, lw=2)

    # make N subdivisions, and chose a random position within each one
    b = (generator.random(N) + range(N)) / N
    ax.scatter(b, [0.5] * len(b), s=60, facecolor="k", edgecolor="k")
    ax.set(title="stratified resampling")


def show_resample_systematic(
    a, figsize: tuple[float, float] = (6, 2), rng: np.random.Generator | None = None
) -> None:
    generator = resolve_rng(rng)
    N = len(a)
    ax, _ = plot_hbar(a, N, figsize=figsize)

    xs = np.linspace(0.0, 1.0 - 1.0 / N, N)
    ax.vlines(xs, 0, 1, lw=2)

    # make N subdivisions, and chose a random position within each one
    b = (generator.random() + np.array(range(N))) / N
    ax.scatter(b, [0.5] * len(b), s=60, facecolor="k", edgecolor="k")
    ax.set(title="systematic resampling")


def show_resample_residual(a, figsize: tuple[float, float] = (6, 2)) -> None:
    N = len(a)
    ax, cumsum = plot_hbar(a, N, figsize=figsize)

    indexes = residual_resample(np.asarray(a) / np.sum(a))
    bins = np.bincount(indexes)
    for i in range(1, N):
        n = bins[i - 1]  # number particles in this sample
        if n > 0:
            b = np.linspace(cumsum[i - 1], cumsum[i], n + 2)[1:-1]
            plt.scatter(b, [0.5] * len(b), s=60, facecolor="k", edgecolor="k")
    ax.set(title="residual resampling")
