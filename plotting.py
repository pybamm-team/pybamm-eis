#
# Functions for plotting
#
import matplotlib.pyplot as plt
import numpy as np


def nyquist_plot(data, ax=None, **kwargs):
    """
    Generates a Nyquist plot from data. Calls `matplotlib.pyplot.scatter`
    with keyword arguments 'kwargs'. For a list of 'kwargs' see the
    `matplotlib plot documentation <https://tinyurl.com/mr3ztw7x>`_

    Parameters
    ----------
    data : list or array-like
        The data to be plotted.
    ax : matplotlib Axis, optional
        The axis on which to put the plot. If None, a new figure
        and axis is created.
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

    ax.scatter(data.real, -data.imag, **kwargs)
    ax.set_xlabel(r"$Z_\mathrm{Re}$")
    ax.set_ylabel(r"$-Z_\mathrm{Im}$")

    if show:
        plt.show()

    return ax
