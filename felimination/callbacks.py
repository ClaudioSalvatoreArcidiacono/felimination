def plot_progress_callback(selector, *args, **kwargs):
    """Plot the feature selection progress."""
    from IPython import display

    display.clear_output(wait=True)
    selector.plot()
