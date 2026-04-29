import sys

import numpy as np
from numpy import random
from scipy import stats

from .plot_common import plot_cov_ellipse

sys.path.append("..")
from filters.kalman_ukf import unscented_transform
from filters.sigma_points import MerweScaledSigmas


def _plot_comparison(
    ax, data, transformed, computed_mean, computed_std, method_name, color="b"
):
    """Plot EKF/UKF vs Monte Carlo comparison."""
    norm = stats.norm(computed_mean, computed_std)
    xs = np.linspace(-3, 5, 200)
    ax.plot(xs, norm.pdf(xs), ls="--", lw=2, color=color)
    ax.hist(transformed, bins=200, density=True, histtype="step", lw=2, color="g")

    actual_mean = transformed.mean()
    ax.axvline(actual_mean, lw=2, color="g", label="Monte Carlo")
    ax.axvline(computed_mean, lw=2, ls="--", color=color, label=method_name)
    ax.legend()

    print(f"actual mean={transformed.mean():.2f}, std={transformed.std():.2f}")
    print(f"{method_name:4} mean={computed_mean:.2f}, std={computed_std:.2f}")


def plot_ekf_vs_mc(ax):
    """Plot EKF vs Monte Carlo comparison for nonlinear transformation."""

    def fx(x):
        return x**3

    def dfx(x):
        return 3 * x**2

    mean, var = 1, 0.1
    std = np.sqrt(var)
    data = random.normal(loc=mean, scale=std, size=50000)
    transformed = fx(data)

    ekf_mean = fx(mean)
    slope = dfx(mean)
    ekf_std = abs(slope * std)

    _plot_comparison(ax, data, transformed, ekf_mean, ekf_std, "EKF", "b")


def plot_ukf_vs_mc(ax, kappa=1.0, alpha=0.001, beta=3.0):
    """Plot UKF vs Monte Carlo comparison for nonlinear transformation."""

    def fx(x):
        return x**3

    mean, var = 1, 0.1
    std = np.sqrt(var)
    data = random.normal(loc=mean, scale=std, size=50000)
    transformed = fx(data)

    points = MerweScaledSigmas(1, kappa, alpha, beta)
    sigmas = points.sigma_points(mean, var)
    sigmas_f = np.array([[fx(s[0])] for s in sigmas])

    # Pass through unscented transform
    ukf_mean, ukf_cov = unscented_transform(sigmas_f, points.Wm, points.Wc)
    ukf_mean = ukf_mean[0]
    ukf_std = np.sqrt(ukf_cov[0])

    _plot_comparison(ax, data, transformed, ukf_mean, ukf_std, "UKF", "b")


def show_linearization(ax, tan_x=1.5):
    """Show linearization example with tangent line."""
    xs = np.arange(0, 2, 0.01)
    ys = xs**2 - 2 * xs
    tan_y = tan_x**2 - 2 * tan_x

    def tan_line(x):
        return (2 * tan_x - 2) * (x - tan_x) + tan_y

    ax.plot(xs, ys, label="$f(x)=x^2-2x$")
    ax.plot(
        [tan_x - 0.5, tan_x + 0.5],
        [tan_line(tan_x - 0.5), tan_line(tan_x + 0.5)],
        color="k",
        ls="--",
        label="linearization",
    )
    ax.axvline(tan_x, lw=1, c="k")
    ax.set(xlabel=f"$x={tan_x}$", title=f"Linearization of $f(x)$ at $x={tan_x}$")
    ax.legend()


def plot_sigmas(ax, sigmas, x, cov):
    """Plot sigma points with weights."""
    pts = sigmas.sigma_points(x, cov)
    ax.scatter(pts[:, 0], pts[:, 1], s=sigmas.Wm * 1000)
    ax.axis("equal")
    ax.grid(1, linestyle="--")


def _plot_sigmas(ax, s, w, alpha=0.5, **kwargs):
    """Plot sigma points with scaled sizes based on weights."""
    min_w = min(abs(w))
    scale_factor = 100 / min_w
    ax.scatter(s[:, 0], s[:, 1], s=abs(w) * scale_factor, alpha=alpha, **kwargs)


def plot_sigmas_selection(ax, kappas=None, alphas=None, betas=None, var=None):
    """Plot sigma points selection for different parameters."""
    if kappas is None:
        kappas = [1.0, 0.15, 10]
    if alphas is None:
        alphas = [0.09, 0.15, 0.2]
    if betas is None:
        betas = [2.0, 1.0, 3.0]
    if var is None:
        var = [0.5]

    P = np.array([[3, 1.1], [1.1, 4]])
    xs = np.array([[2, 5], [5, 5], [8, 5]])

    for x, κ, α, β in zip(xs, kappas, alphas, betas, strict=True):
        points = MerweScaledSigmas(2, κ, α, β)
        sigmas = points.sigma_points(x, P)
        _plot_sigmas(ax, sigmas, points.Wc, alpha=1.0, facecolor="k")
        plot_cov_ellipse(
            ax, x, P, stds=np.sqrt(var), facecolor="b", alpha=0.3, show_title=False
        )

    ax.axis("equal")
    ax.set(xlim=(0, 10), ylim=(0, 10))
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)


def plot_sigmas_compar_param(axes, obj, kappas=None, alphas=None, betas=None, var=None):
    """Plot sigma points comparison for different parameters."""
    if kappas is None:
        kappas = [1.0, 1.0]
    if alphas is None:
        alphas = [0.3, 1.0]
    if betas is None:
        betas = [2.0, 2.0]
    if var is None:
        var = [1, 4]

    x = np.array([0, 0])
    P = np.array([[4, 2], [2, 4]])

    for ax, κ, α, β in zip(axes, kappas, alphas, betas, strict=True):
        sigmas = MerweScaledSigmas(n=2, kappa=κ, alpha=α, beta=β)
        _plot_sigmas(ax, sigmas.sigma_points(x, P), sigmas.Wc, c="b")
        plot_cov_ellipse(ax, x, P, stds=np.sqrt(var), facecolor="g", alpha=0.2)

        obj_dict = {"kappa": κ, "alpha": α, "beta": β}
        ax.set(title=f"{obj} = {obj_dict[obj]}")
