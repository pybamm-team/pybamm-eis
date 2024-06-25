import matplotlib.pyplot as plt
import numpy as np


def nyquist_plot(data, ax=None, marker="o", linestyle="None", **kwargs):
    """
    Generates a Nyquist plot from data. Calls `matplotlib.pyplot.plot`
    with keyword arguments 'kwargs'. For a list of 'kwargs' see the
    `matplotlib plot documentation <https://tinyurl.com/mr3ztw7x>`_

    Parameters
    ----------
    data : list or array-like
        The data to be plotted.
    ax : matplotlib Axis, optional
        The axis on which to put the plot. If None, a new figure
        and axis is created.
    marker : str, optional
        The marker to use for the plot. Default is 'o'
    linestyle : str, optional
        The linestyle to use for the plot. Default is 'None'
    kwargs
        Keyword arguments, passed to plt.scatter.
    """

    if isinstance(data, list):
        data = np.array(data)

    if ax is None:
        _, ax = plt.subplots()
        show = True
    else:
        show = False

    ax.plot(data.real, -data.imag, marker=marker, linestyle=linestyle, **kwargs)
    _, xmax = ax.get_xlim()
    _, ymax = ax.get_ylim()
    axmax = max(xmax, ymax)
    plt.axis([0, axmax, 0, axmax])
    plt.gca().set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$Z_\mathrm{Re}$ [Ohm]")
    ax.set_ylabel(r"$-Z_\mathrm{Im}$ [Ohm]")
    if show:
        plt.show()

    return ax
