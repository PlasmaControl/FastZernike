"""Functions for plotting."""

import matplotlib
import matplotlib.pyplot as plt
from zernipax.backend import np
from zernipax.grid import LinearGrid
from zernipax.zernike import fourier, zernike_radial


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
    .. code-block:: python

        from zernipy.plotting import plot_basis
        from zernipy.basis import ZernikePolynomial
        basis = basis = ZernikePolynomial(L=5, M=5)
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


def plot_mode(mode, rho=100, theta=100, **kwargs):
    """Plot basis functions.

    Parameters
    ----------
    mode : (2,) array
        [L, M] mode to plot
    rho : int, optional
        Number of points in the radial direction
    theta : int, optional
        Number of points in the angular direction
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
    .. code-block:: python

        from zernipy.plotting import plot_mode
        mode = [3, 1]
        fig, ax = plot_mode(mode)

    """
    if len(mode) == 2:
        L = mode[0]
        M = mode[1]

        grid = LinearGrid(rho=rho, theta=theta, endpoint=True)
        r = grid.nodes[grid.unique_rho_idx, 0]
        v = grid.nodes[grid.unique_theta_idx, 1]

        fig = plt.figure(figsize=kwargs.get("figsize", (8, 8)))

        radial = zernike_radial(r, L, M)
        poloidal = fourier(v, M)

        Z = radial * poloidal

        ax = plt.subplot(1, 2, 1, projection="polar")
        ax.set_title("$l={}, m={}$".format(L, M))
        ax.axis("off")
        im = ax.contourf(
            v,
            r,
            Z,
            levels=np.linspace(-1, 1, 200),
            cmap=kwargs.get("cmap", "coolwarm"),
        )
        fig.colorbar(im, ax=ax, shrink=0.3, ticks=np.linspace(-1, 1, 9))

        return fig, ax


def plot_modes(modes, rho=100, theta=100, **kwargs):
    """Plot basis functions.

    Parameters
    ----------
    modes : (2,N) array
        N different [L, M] modes to plot
    rho : int, optional
        Number of points in the radial direction
    theta : int, optional
        Number of points in the angular direction
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
    .. code-block:: python

        from zernipy.plotting import plot_modes
        modes = np.array([[4, 2], [3,1], [3,3]])
        fig, ax = plot_modes(modes)

    """
    if modes.shape[1] == 2:
        L = modes[:, 0]
        M = modes[:, 1]

        grid = LinearGrid(rho=rho, theta=theta, endpoint=True)
        r = grid.nodes[grid.unique_rho_idx, 0]
        v = grid.nodes[grid.unique_theta_idx, 1]

        fig = plt.figure(figsize=kwargs.get("figsize", (8, 8)))

        Z = np.zeros((r.size, v.size))
        for l, m in zip(L, M):
            radial = zernike_radial(r, l, m)
            poloidal = fourier(v, m)
            Z += radial * poloidal
        Z = Z / L.size
        ax = plt.subplot(1, 2, 1, projection="polar")
        ax.axis("off")
        im = ax.contourf(
            v,
            r,
            Z,
            levels=np.linspace(-1, 1, 200),
            cmap=kwargs.get("cmap", "coolwarm"),
        )
        fig.colorbar(im, ax=ax, shrink=0.4, ticks=np.linspace(-1, 1, 9))

        return fig, ax
