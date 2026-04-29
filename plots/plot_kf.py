import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

from .plot_common import gen_data_by_only_x, plot_track, plot_zs


def plot_kf(
    ax, xs, ys=None, dt=1, var=None, label="Filter", band_color="blue", **kwargs
):
    """Plot Kalman filter results with optional variance bands."""
    if ys is None:
        xs, ys = gen_data_by_only_x(xs, dt)
    ax.plot(xs, ys, label=label, **kwargs)

    if var is not None:
        std = np.sqrt(var)
        ax.plot(xs, ys + std, linestyle=":", color="k", lw=2)
        ax.plot(xs, ys - std, linestyle=":", color="k", lw=2)
        ax.fill_between(xs, ys - std, ys + std, facecolor=band_color, alpha=0.1)


def plot_kf_track(ax, xs, filter_xs, zs, label=None, title=None):
    """Plot Kalman filter tracking results."""
    plot_kf(ax, filter_xs[:, 0])
    plot_track(ax, xs[:, 0])

    if zs is not None:
        plot_zs(ax, zs, label=label)

    ax.set(title=title, xlabel="time", ylabel="meters", xlim=(-1, len(xs)))
    ax.legend()


def plot_kf_with_cov(
    ax,
    xs,
    cov,
    track,
    zs,
    std_scale=1,
    y_lim=None,
    xlabel="time",
    ylabel="position",
    title="Kalman Filter",
):
    """Plot Kalman filter with covariance visualization."""
    num = len(zs)
    zs = np.asarray(zs)
    cov = np.asarray(cov)

    std = std_scale * np.sqrt(cov[:, 0, 0])
    std_top = track + std
    std_btm = track - std

    plot_track(ax, track, c="k")
    plot_zs(ax, xs=zs)
    plot_kf(ax, xs)

    ax.plot(std_top, linestyle=":", color="k", lw=1, alpha=0.4)
    ax.plot(std_btm, linestyle=":", color="k", lw=1, alpha=0.4)
    ax.fill_between(
        range(len(std_top)),
        std_btm,
        std_top,
        facecolor="yellow",
        alpha=0.2,
        interpolate=True,
    )

    ax.set(xlabel=xlabel, ylabel=ylabel, xlim=(0, num), title=title)
    ax.set(ylim=y_lim or (-50, num + 10))
    ax.legend()


def plot_kf_with_resids(axes, dt, xs, z_xs, res):
    """Plot Kalman filter results with residuals."""
    t = np.arange(0, len(xs) * dt, dt)
    if z_xs is not None:
        plot_zs(axes[0], xs=t, ys=z_xs, dt=dt, label="z")
    plot_kf(axes[0], xs=t, ys=xs, dt=dt)
    axes[0].set(xlabel="time", ylabel="X", title="estimates vs measurements")
    axes[0].legend()

    axes[1].plot(t, res)
    axes[1].set(xlabel="time", ylabel="residual", title="residuals")


def show_markov_chain(figsize=(4, 4), facecolor="w"):
    """Show a markov chain showing relative probability of an object turning."""
    fig = plt.figure(figsize=figsize, facecolor=facecolor)
    ax = plt.axes((0, 0, 1, 1), xticks=[], yticks=[], frameon=False)

    box_bg = "#DDDDDD"
    kf1c = patches.Circle((4, 5), 0.5, fc=box_bg)
    kf2c = patches.Circle((6, 5), 0.5, fc=box_bg)
    ax.add_patch(kf1c)
    ax.add_patch(kf2c)

    ax.text(4, 5, "Straight", ha="center", va="center", fontsize="medium")
    ax.text(6, 5, "Turn", ha="center", va="center", fontsize="medium")

    # Define arrow configurations
    arrows = [
        {
            "xy": (6, 4.5),
            "xytext": (4.1, 4.5),
            "pos": (5, 3.9),
            "text": ".05",
            "style": "arc3,rad=-0.5",
        },
        {
            "xy": (4.1, 5.5),
            "xytext": (6, 5.5),
            "pos": (5, 6.1),
            "text": ".03",
            "style": "arc3,rad=-0.5",
        },
        {
            "xy": (3.9, 5.5),
            "xytext": (3.55, 5.2),
            "pos": (3.5, 5.6),
            "text": ".97",
            "style": "angle3,angleA=150,angleB=0",
        },
        {
            "xy": (6.1, 5.5),
            "xytext": (6.45, 5.2),
            "pos": (6.5, 5.6),
            "text": ".95",
            "style": "angle3,angleA=-150,angleB=2",
        },
    ]

    for arrow in arrows:
        ax.annotate(
            "",
            xy=arrow["xy"],
            xycoords="data",
            xytext=arrow["xytext"],
            textcoords="data",
            size=10,
            arrowprops={
                "arrowstyle": "->",
                "ec": "k",
                "connectionstyle": arrow["style"],
            },
        )
        ax.text(
            arrow["pos"][0],
            arrow["pos"][1],
            arrow["text"],
            ha="center",
            va="center",
            fontsize="large",
        )

    ax.axis("equal")
    return fig


def plot_adaptive_2d(
    axes, xs, z_xs2, dt, Q_scale_factor, std_scale, *, std_title=False, Q_title=False
):
    """Plot adaptive Kalman filter 2D results."""
    plot_zs(axes[0], z_xs2, dt=dt, label="z")
    plot_kf(axes[0], xs[:, 0], dt=dt, lw=1.5)

    title_suffix = f"(std={std_scale}, Q scale={Q_scale_factor})"
    axes[0].set(xlabel="time", ylabel="ϵ", title=f"position {title_suffix}")
    axes[1].plot(np.arange(0, len(xs) * dt, dt), xs[:, 1], lw=1.5)
    axes[1].set(xlabel="time", title=f"velocity {title_suffix}")


def plot_fusion_kf2d(ax, saver):
    """Plot fusion Kalman filter 2D results."""
    ts = np.arange(0.1, 10, 0.1)
    plot_zs(ax, ts, saver.z[:, 0])
    ax.plot(ts, saver.x[:, 0], label="Kalman Filter")
    ax.plot(ts, saver.z[:, 1], ls="--", label="Sensor")
    ax.legend(loc=4)
    ax.set(xlabel="time", ylabel="meters", ylim=(0, 100))


def plot_fusion_kf(axes, xs, ts, zs_pos, zs_vel):
    """Plot fusion Kalman filter position and velocity results."""
    ys = np.array(xs)

    axes[0].plot(zs_pos[:, 0], zs_pos[:, 1], ls="--", label="Pos Sensor")
    plot_kf(axes[0], ts, ys[:, 0], label="Kalman Filter")
    axes[0].set(title="Position", ylabel="meters")
    axes[0].legend()

    plot_zs(axes[1], zs_vel[:, 0], zs_vel[:, 1], label="Vel Sensor")
    plot_kf(axes[1], ts, ys[:, 1], label="Kalman Filter")
    axes[1].set(title="Velocity", ylabel="meters", xlabel="time")
    axes[1].legend()
