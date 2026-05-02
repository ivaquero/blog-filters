import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from scipy import stats


def prepend_x0(x0, data):
    """Prepend initial value to data array."""
    if isinstance(x0, list):
        return [x0, *data.tolist()]
    if isinstance(x0, np.ndarray):
        return np.concatenate([x0, data])
    return data


def gen_data_by_only_x(xs, dt):
    """Generate time and data arrays from data sequence."""
    return np.arange(0, len(xs) * dt, dt), xs


def plot_zs(ax, xs, ys=None, x0=None, dt=1, label="Measured", **scatter_kwargs):
    """Plot scatter data with optional initial value."""
    if x0:
        xs = prepend_x0(x0, xs)
    if ys is None:
        xs, ys = gen_data_by_only_x(xs, dt)

    ax.scatter(xs, ys, label=label, marker="x", color="k", **scatter_kwargs)
    ax.grid(1)


def plot_track(ax, xs, ys=None, dt=None, label="Track", c="k", lw=2, ls=":", **kwargs):
    """Plot track line with optional time axis."""
    if ys is None and dt is not None:
        xs, ys = gen_data_by_only_x(xs, dt)

    plot_data = xs if ys is None else (xs, ys)
    return ax.plot(plot_data, color=c, lw=lw, ls=ls, label=label, **kwargs)


def plot_preds(ax, priors, kind=None):
    """Plot prediction data."""
    rng = range(len(priors))
    plot_func = ax.scatter if kind == "scatter" else ax.plot
    plot_func(
        rng,
        priors,
        marker="d" if kind == "scatter" else None,
        ls="-.",
        label="Predicted",
        color="r",
    )
    ax.legend()


def plot_cov2d(axes, cov):
    """Plot 2D covariance components."""
    axes[0].set(title=r"$σ^2_x$")
    plot_covariance(axes[0], cov, (0, 0))
    axes[1].set(title=r"$σ^2_ẋ$")
    plot_covariance(axes[1], cov, (1, 1))


def plot_covariance(ax, P, index=(0, 0)):
    """Plot covariance time series."""
    ps = [p[index[0], index[1]] for p in P]
    ax.plot(ps)


def plot_track_ellipses(ax, N, zs, xs, cov, title):
    """Plot covariance ellipses for track visualization."""
    for i, p in enumerate(cov):
        plot_cov_ellipse(
            ax,
            (i + 1, xs[i]),
            cov=p,
            stds=[2] * N,
            edgecolor="darkslateblue",
            facecolor="white",
        )
    ax.set(title=title)
    return p


def cal_cov_ellipse(cov, deviations=1):
    """Calculate covariance ellipse parameters."""
    U, s, _ = np.linalg.svd(cov)
    orientation = math.atan2(U[1, 0], U[0, 0])
    width_radius = deviations * np.sqrt(s[0])
    height_radius = deviations * np.sqrt(s[1])

    if height_radius > width_radius:
        raise ValueError("width_radius must be greater than height_radius")

    return orientation, width_radius, height_radius


def plot_cov_ellipse(
    ax,
    mean,
    cov,
    stds=None,
    show_semiaxis=False,
    show_center=True,
    edgecolor="darkslateblue",
    facecolor="green",
    alpha=0.2,
    label="",
    show_title=True,
    **line_kwargs,
):
    """Plot covariance ellipse on axes."""
    stds = [1] if stds is None else stds
    ellipse = cal_cov_ellipse(cov)

    angle = np.degrees(ellipse[0])
    width = ellipse[1] * 2.0
    height = ellipse[2] * 2.0

    for stdi in stds:
        e = patches.Ellipse(mean, stdi * width, stdi * height, angle=angle)
        ax.add_patch(e)
        e.set(
            facecolor=facecolor,
            edgecolor=edgecolor,
            alpha=alpha,
            label=label,
            **line_kwargs,
        )

    x, y = mean
    if show_center:
        ax.scatter(x, y, marker="+", color=edgecolor)

    if show_semiaxis:
        a = ellipse[0]
        h, w = height / 4, width / 4
        ax.plot(
            [x, x + h * math.cos(a + math.pi / 2)],
            [y, y + h * math.sin(a + math.pi / 2)],
        )
        ax.plot([x, x + w * math.cos(a)], [y, y + w * math.sin(a)])

    if show_title:
        ax.set(title=f"[{cov[0]}\n   {cov[1]}]")


def plot_resids_lims(ax, Ps, stds=1.0):
    """Plot residual confidence limits."""
    std = np.sqrt(Ps) * stds
    ax.plot(-std, color="k", ls=":", lw=2)
    ax.plot(std, color="k", ls=":", lw=2)
    ax.fill_between(range(len(std)), -std, std, facecolor="#ffff00", alpha=0.3)


def plot_resids(ax, xs, data, col, ylabel, stds=1, title=None, *, limits=True):
    """Plot residuals with confidence limits."""
    res = xs - data.x[:, col]
    ax.plot(res)

    if limits:
        Ps = data.P[:, col, col]
        plot_resids_lims(ax, Ps, stds=stds)

    ax.set(title=title, xlabel="time", ylabel=ylabel)


def plot_transferred_gaussian(
    data, func, func_name="f(x)", out_lim=None, num_bins=300, figsize=(8, 6)
):
    """Plot transferred Gaussian distribution analysis."""
    ys = func(data)
    std = np.std(ys)
    x0 = np.mean(data)
    in_std = np.std(data)
    in_lims = [x0 - in_std * 3, x0 + in_std * 3]
    y = func(x0)

    if out_lim is None:
        out_lim = [y - std * 3, y + std * 3]

    _, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)

    # Plot output histogram
    h = np.histogram(ys, num_bins, density=False)
    axes[0, 0].plot(h[1][1:], h[0], lw=2, alpha=0.8)
    if out_lim is not None:
        axes[0, 0].set(xlim=(out_lim[0], out_lim[1]))
    axes[0, 0].set(title="Output", yticklabels=[])
    axes[0, 0].axvline(np.mean(ys), ls="--", lw=2, label="computed mean")
    axes[0, 0].axvline(func(x0), lw=1, label="actual mean")
    axes[0, 0].legend()

    # Hide unused subplot
    axes[0, 1].set_visible(False)

    # Plot transfer function
    x = np.arange(in_lims[0], in_lims[1], 0.1)
    y_vals = func(x)
    axes[1, 0].plot(x, y_vals, "k")
    axes[1, 0].plot([x0, x0, in_lims[1]], [out_lim[1], y, y], color="r", lw=1)
    axes[1, 0].set(xlim=in_lims, ylim=out_lim, title=f"f(x) = {func_name}")

    # Plot input histogram
    h = np.histogram(data, num_bins, density=True)
    axes[1, 1].plot(h[0], h[1][1:], lw=2)
    axes[1, 1].set(title="Input", xticklabels=[])


def plot_distributed_scatters(data, f, N, figsize=(6, 3)):
    """Plot distributed scatter plots for input and output data."""
    _, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    axes[0].scatter(data[:N], range(N), alpha=0.2, s=1)
    axes[0].set(title="Input")

    axes[1].scatter(f(data[:N]), range(N), alpha=0.2, s=1)
    axes[1].set(title="Output")


def plot_bivariate_colormap(ax, xs, ys):
    """Plot bivariate colormap using Gaussian kernel density estimation."""
    xs, ys = np.asarray(xs), np.asarray(ys)
    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()
    values = np.vstack([xs, ys])
    kernel = stats.gaussian_kde(values)
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    pos = np.vstack([X.ravel(), Y.ravel()])

    Z = np.reshape(kernel.evaluate(pos).T, X.shape)
    ax.imshow(np.rot90(Z), cmap=plt.cm.Greys, extent=[xmin, xmax, ymin, ymax])


def plot_monte_carlo_mean(
    axes, xs, ys, f, mean_fx, label, figsize=(8, 4), *, plot_colormap=True
):
    """Plot Monte Carlo mean analysis with bivariate colormap."""
    if plot_colormap:
        plot_bivariate_colormap(axes[0], xs, ys)

    axes[0].scatter(xs, ys, marker=".", alpha=0.02, color="k")
    axes[0].set(xlim=(-20, 20), ylim=(-20, 20))
    axes[0].grid(0)

    fxs, fys = f(xs, ys)
    computed_mean_x = np.average(fxs)
    computed_mean_y = np.average(fys)

    if plot_colormap:
        plot_bivariate_colormap(axes[1], fxs, fys)

    axes[1].scatter(fxs, fys, marker=".", alpha=0.02, color="k")
    axes[1].scatter(
        mean_fx[0], mean_fx[1], marker="v", s=figsize[1] * 30, c="r", label=label
    )
    axes[1].scatter(
        computed_mean_x,
        computed_mean_y,
        marker="*",
        s=figsize[1] * 20,
        c="b",
        label="Computed Mean",
    )
    axes[1].set(xlim=[-100, 100], ylim=[-10, 200])
    axes[1].grid(0)
    axes[1].legend()

    print(
        f"Difference in mean x={computed_mean_x - mean_fx[0]:.3f}, y={computed_mean_y - mean_fx[1]:.3f}"
    )
