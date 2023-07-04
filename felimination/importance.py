from typing import Any

from sklearn.inspection import permutation_importance


class PermutationImportance:
    """Wrapper around sklearn.inspection.permutation_importance.

    Parameters
    ----------
    scoring : str, callable, list, tuple, or dict, default=None
        Scorer to use.
        If `scoring` represents a single score, one can use:
        - a single string;
        - a callable that returns a single value.
        If `scoring` represents multiple scores, one can use:
        - a list or tuple of unique strings;
        - a callable returning a dictionary where the keys are the metric
        names and the values are the metric scores;
        - a dictionary with metric names as keys and callables a values.
        Passing multiple scores to `scoring` is more efficient than calling
        `permutation_importance` for each of the scores as it reuses
        predictions to avoid redundant computation.
        If None, the estimator's default scorer is used.
    n_repeats : int, default=5
        Number of times to permute a feature.
    n_jobs : int or None, default=None
        Number of jobs to run in parallel. The computation is done by computing
        permutation score for each columns and parallelized over the columns.
        `None` means 1 unless in a :obj:`joblib.parallel_backend` context.
        `-1` means using all processors.
    random_state : int, RandomState instance, default=None
        Pseudo-random number generator to control the permutations of each
        feature.
        Pass an int to get reproducible results across function calls.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights used in scoring.
    max_samples : int or float, default=1.0
        The number of samples to draw from X to compute feature importance
        in each repeat (without replacement).
        - If int, then draw `max_samples` samples.
        - If float, then draw `max_samples * X.shape[0]` samples.
        - If `max_samples` is equal to `1.0` or `X.shape[0]`, all samples
        will be used.
        While using this option may provide less accurate importance estimates,
        it keeps the method tractable when evaluating feature importance on
        large datasets. In combination with `n_repeats`, this allows to control
        the computational speed vs statistical accuracy trade-off of this method.
    """

    def __init__(
        self,
        scoring=None,
        n_repeats=5,
        n_jobs=None,
        random_state=None,
        sample_weight=None,
        max_samples=1.0,
    ):
        self.scoring = scoring
        self.n_repeats = n_repeats
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.sample_weight = sample_weight
        self.max_samples = max_samples

    def __call__(self, estimator, X, y) -> Any:
        """Computes the permutation importance.

        Parameters
        ----------
        estimator : object
            An estimator that has already been fitted and is compatible
            with scorer.
        X : ndarray or DataFrame, shape (n_samples, n_features)
            Data on which permutation importance will be computed.
        y : array-like or None, shape (n_samples, ) or (n_samples, n_classes)
            Targets for supervised or `None` for unsupervised.

        Returns
        -------
        importances_mean : ndarray of shape (n_features, )
            Mean of feature importance over `n_repeats`.
        """
        return permutation_importance(
            estimator,
            X,
            y,
            scoring=self.scoring,
            n_repeats=self.n_repeats,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            sample_weight=self.sample_weight,
            max_samples=self.max_samples,
        ).importances_mean
