def plot_progress_callback(selector, *args, **kwargs):
    """Plot the feature selection progress."""
    from IPython import display
    from matplotlib import pyplot as plt

    display.clear_output(wait=True)
    selector.plot()
    plt.show()
