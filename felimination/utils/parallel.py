try:
    # For scikit-learn versions 1.3.0 and later
    from sklearn.utils.parallel import Parallel, delayed  # noqa
except ImportError:
    # Before 1.3.0
    from joblib import Parallel  # noqa
    from sklearn.utils.fixes import delayed  # noqa
