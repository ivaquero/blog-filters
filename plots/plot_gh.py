import numpy as np

from .plot_common import plot_preds, plot_zs, prepend_x0


def plot_ghf(ax, data, filt, x0, vary_obj="data"):
    if x0:
        data = prepend_x0(x0, data)
    rng = range(len(data) + 1)
    estimates, predictions = filt.batch_filter(data, save_preds=True)

    label_printed = {"dx": f" by dx={filt.dx}", "g": f" by g={filt.g}", "data": ""}

    if hasattr(filt, "h"):
        label_printed["h"] = f" by h={filt.h}"
        label_printed["gh"] = f" by g={filt.g}, h={filt.h}"

    ax.plot(
        rng,
        estimates[:, 0],
        ms=4,
        marker="o",
        label=f"Filtered{label_printed[vary_obj]}",
    )
    ax.legend()
    return estimates, predictions


def plot_gh_compar_param(
    axes,
    data,
    filters,
    x0s,
    rng=None,
    vary_obj="data",
    figwidth=8,
    *,
    sharey=False,
    combined=True,
    show_preds=False,
    **scatter_kwargs,
):
    if combined:
        if len(set(x0s)) != 1:
            error_message = "x0s must be uniform"
            raise ValueError(error_message)

        x0 = x0s[0]
        ax = axes
        plot_zs(ax, xs=data, x0=x0, **scatter_kwargs)
        for filt in filters:
            _, preds = plot_ghf(ax, data, filt, x0, vary_obj=vary_obj)
            if show_preds:
                plot_preds(ax, np.r_[x0, preds])

    else:
        for filt, x0, ax in zip(filters, x0s, axes.flatten()):
            plot_zs(ax, xs=data, x0=x0, **scatter_kwargs)
            _, preds = plot_ghf(ax, data, filt, x0, vary_obj=vary_obj)
            if show_preds:
                plot_preds(ax, np.r_[x0, preds])


def plot_gh_compar_data(axes, data_ls, filt, x0, *, show_preds=False, **scatter_kwargs):
    for ax, data in zip(axes.flatten(), data_ls):
        plot_zs(ax, xs=data, x0=x0, **scatter_kwargs)
        _, preds = plot_ghf(ax, data, filt, x0)
        if show_preds:
            plot_preds(ax, np.r_[x0, preds])
