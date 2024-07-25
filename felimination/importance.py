from inspect import signature
from operator import attrgetter
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection._validation import _score
from sklearn.utils import safe_sqr
from sklearn.utils.metaestimators import _safe_split


def _train_score_get_importance(
    estimator, X, y, train, test, scorer, importance_getter, **fit_params
):
    """
    Train and test an estimator and get the feature importances.

    Parameters
    ----------
    estimator : estimator
        A scikit-learn estimator to train score and to calculate importance on.
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The feature samples to use to train and test the estimator.
    y : array-like of shape (n_samples,)
        The target values to use to train and test the estimator.
    train : array-like of shape (n_train_samples,)
        The indices of the training samples.
    test : array like of shape (n_test_samples,)
        The indices of the test samples.
    scorer : callable
        The scorer to use to score the estimator.
    importance_getter : "auto", str or callable
        An attribute or a callable to get the feature importance. If `"auto "`,
        `estimator` is expected to expose `coef_` or `feature_importances_`.

    Returns
    -------
    train_score : float
        The score of the estimator on the training set.
    test_score : float
        The score of the estimator on the test set.
    importances : ndarray of shape (n_features,)
        The features importances.
    """

    estimator = clone(estimator)
    X_train, y_train = _safe_split(estimator, X, y, train)
    X_test, y_test = _safe_split(estimator, X, y, test, train)
    estimator = estimator.fit(X_train, y_train, **fit_params)
    train_score = _score(estimator, X_train, y_train, scorer, score_params=None)
    test_score = _score(estimator, X_test, y_test, scorer, score_params=None)
    importances = _get_feature_importances(
        estimator, importance_getter, X=X_test, y=y_test
    )
    return train_score, test_score, importances


def _get_feature_importances(
    estimator, getter, transform_func=None, norm_order=1, X=None, y=None
):
    """
    Retrieve and aggregate (ndim > 1)  the feature importances
    from an estimator. Also optionally applies transformation.

    Parameters
    ----------
    estimator : estimator
        A scikit-learn estimator from which we want to get the feature
        importances.
    getter : "auto", str or callable
        An attribute or a callable to get the feature importance. If `"auto"`,
        `estimator` is expected to expose `coef_` or `feature_importances`.
    transform_func : {"norm", "square"}, default=None
        The transform to apply to the feature importances. By default (`None`)
        no transformation is applied.
    norm_order : int, default=1
        The norm order to apply when `transform_func="norm"`. Only applied
        when `importances.ndim > 1`.
    X : {array-like, sparse matrix} of shape (n_samples, n_features), default=None
        The feature samples to use to compute feature importance.
    y : array-like of shape (n_samples,), default=None
        The target values to use to compute feature importance.

    Returns
    -------
    importances : ndarray of shape (n_features,)
        The features importances, optionally transformed.
    """
    if isinstance(getter, str):
        if getter == "auto":
            if hasattr(estimator, "coef_"):
                getter = attrgetter("coef_")
            elif hasattr(estimator, "feature_importances_"):
                getter = attrgetter("feature_importances_")
            else:
                raise ValueError(
                    "when `importance_getter=='auto'`, the underlying "
                    f"estimator {estimator.__class__.__name__} should have "
                    "`coef_` or `feature_importances_` attribute. Either "
                    "pass a fitted estimator to feature selector or call fit "
                    "before calling transform."
                )
        else:
            getter = attrgetter(getter)
        importances = getter(estimator)

    else:
        # getter is a callable
        if len(signature(getter).parameters) == 3:
            importances = getter(estimator, X, y)
        else:
            importances = getter(estimator)

    if transform_func is None:
        return importances
    elif transform_func == "norm":
        if importances.ndim == 1:
            importances = np.abs(importances)
        else:
            importances = np.linalg.norm(importances, axis=0, ord=norm_order)
    elif transform_func == "square":
        if importances.ndim == 1:
            importances = safe_sqr(importances)
        else:
            importances = safe_sqr(importances).sum(axis=0)
    else:
        raise ValueError(
            "Valid values for `transform_func` are "
            + "None, 'norm' and 'square'. Those two "
            + "transformation are only supported now"
        )

    return importances


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
