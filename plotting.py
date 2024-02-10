import matplotlib
import matplotlib.pyplot as plt
from backend import np
from grid import LinearGrid


def _set_tight_layout(fig):
    # compat layer to deal with API changes in mpl 3.6.0
    if int(matplotlib._version.version.split(".")[1]) < 6:
        fig.set_tight_layout(True)
    else:
        fig.set_layout_engine("tight")


def plot_basis(basis, return_data=False, **kwargs):
    """Plot basis functions.

    Parameters
    ----------
    basis : Basis
        basis to plot
    return_data : bool
        if True, return the data plotted as well as fig,ax
    **kwargs : dict, optional
        Specify properties of the figure, axis, and plot appearance e.g.::

            plot_X(figsize=(4,6),cmap="plasma")

        Valid keyword arguments are:

        * ``figsize``: tuple of length 2, the size of the figure (to be passed to
          matplotlib)
        * ``cmap``: str, matplotlib colormap scheme to use, passed to ax.contourf
        * ``title_fontsize``: integer, font size of the title

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure being plotted to.
    ax : matplotlib.axes.Axes, ndarray of axes, or dict of axes
        Axes used for plotting. A single axis is used for 1d basis functions,
        2d or 3d bases return an ndarray or dict of axes.    return_data : bool
        if True, return the data plotted as well as fig,ax
    plot_data : dict
        dictionary of the data plotted, only returned if ``return_data=True``

    Examples
    --------
    .. image:: ../../_static/images/plotting/plot_basis.png

    .. code-block:: python

        from desc.plotting import plot_basis
        from desc.basis import DoubleFourierSeries
        basis = DoubleFourierSeries(M=3, N=2)
        fig, ax = plot_basis(basis)

    """
    title_fontsize = kwargs.pop("title_fontsize", None)

    if basis.__class__.__name__ in ["ZernikePolynomial", "FourierZernikeBasis"]:
        lmax = abs(basis.modes[:, 0]).max().astype(int)
        mmax = abs(basis.modes[:, 1]).max().astype(int)

        grid = LinearGrid(rho=100, theta=100, endpoint=True)
        r = grid.nodes[grid.unique_rho_idx, 0]
        v = grid.nodes[grid.unique_theta_idx, 1]

        fig = plt.figure(figsize=kwargs.get("figsize", (3 * mmax, 3 * lmax / 2)))

        plot_data = {"amplitude": [], "rho": r, "theta": v}

        ax = {i: {} for i in range(lmax + 1)}
        ratios = np.ones(2 * (mmax + 1) + 1)
        ratios[-1] = kwargs.get("cbar_ratio", 0.25)
        gs = matplotlib.gridspec.GridSpec(
            lmax + 2, 2 * (mmax + 1) + 1, width_ratios=ratios
        )

        modes = basis.modes[basis.modes[:, 2] == 0]
        plot_data["l"] = basis.modes[:, 0]
        plot_data["m"] = basis.modes[:, 1]
        Zs = basis.evaluate(grid.nodes, modes=modes)
        for i, (l, m) in enumerate(
            zip(modes[:, 0].astype(int), modes[:, 1].astype(int))
        ):
            Z = Zs[:, i].reshape((grid.num_rho, grid.num_theta))
            ax[l][m] = plt.subplot(
                gs[l + 1, m + mmax : m + mmax + 2], projection="polar"
            )
            ax[l][m].set_title("$l={}, m={}$".format(l, m))
            ax[l][m].axis("off")
            im = ax[l][m].contourf(
                v,
                r,
                Z,
                levels=np.linspace(-1, 1, 100),
                cmap=kwargs.get("cmap", "coolwarm"),
            )
            plot_data["amplitude"].append(Zs)

        cb_ax = plt.subplot(gs[:, -1])
        plt.subplots_adjust(right=0.8)
        cbar = fig.colorbar(im, cax=cb_ax)
        cbar.set_ticks(np.linspace(-1, 1, 9))
        fig.suptitle(
            "{}, $L={}$, $M={}$, spectral indexing = {}".format(
                basis.__class__.__name__, basis.L, basis.M, basis.spectral_indexing
            ),
            y=0.98,
            fontsize=title_fontsize,
        )
        _set_tight_layout(fig)
        if return_data:
            return fig, ax, plot_data

        return fig, ax