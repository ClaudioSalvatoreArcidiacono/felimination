"""Callbacks for feature selection algorithms."""


def plot_progress_callback(selector, *args, **kwargs):
    """Plot the feature selection progress during the algorithm execution.

    Parameters
    ----------
    selector : object
        The feature selector object.
    """
    from IPython import display
    from matplotlib import pyplot as plt

    display.clear_output(wait=True)
    selector.plot()
    plt.show()
