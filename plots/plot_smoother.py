import numpy as np

from filters.kalman import KalmanFilter

from .plot_common import plot_zs


def plot_rts(ax, R, Q=0.001, seed=123, *, show_velocity=False):
    rng = np.random.default_rng(seed)
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x = np.array([0.0, 1.0])
    kf.F = np.array([[1.0, 1.0], [0.0, 1.0]])
    kf.H = np.array([[1.0, 0.0]])
    kf.P = 10.0 * np.eye(len(kf.x))
    kf.R = R
    kf.Q = Q * np.eye(len(kf.x))
    # create noisy data
    zs = np.asarray([t + rng.standard_normal() * R for t in range(40)])
    # filter data with Kalman filter, than run smoother on it
    mu, cov, _, _ = kf.batch_filter(zs)
    M, _, _, _ = kf.rts_smoother(mu, cov)
    # plot data
    if show_velocity:
        index = 1
    else:
        index = 0
        plot_zs(ax, zs, lw=1)
        N = len(zs)
        ax.plot([0, N], [0, N], "k", lw=2, label="track")

    ax.plot(M[:, index], c="b", label="RTS")
    ax.plot(mu[:, index], c="g", ls="--", label="KF output")

    ax.legend()
