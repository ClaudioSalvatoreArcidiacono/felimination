"""Module with tools to perform forward feature selection using the
Minimum Redundancy Maximum Relevance (MRMR) framework.

This module contains:

- `MRMRRanker`: stateful importance-getter callable implementing the MRMR
    score (mutual information relevance / absolute Pearson correlation
    redundance), suitable for use with `ForwardSelectorCV`.
- `MRMRCV`: preset of `ForwardSelectorCV` wired with `MRMRRanker`.
"""

import numpy as np
import pandas as pd
from sklearn.base import is_classifier
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from felimination.forward import ForwardSelectorCV


def _as_dense_array(X):
    if isinstance(X, pd.DataFrame):
        return X.values
    return np.asarray(X)


def abs_pearson_correlation(X, y):
    """Absolute Pearson correlation between each column of ``X`` and ``y``.

    Convenience helper for use as ``relevance_func`` or
    ``redundance_func`` in `MRMRRanker`. Only suitable for numeric data;
    use mutual-information based scoring (the default) when categorical
    features are present.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    y : array-like of shape (n_samples,)

    Returns
    -------
    ndarray of shape (n_features,)
        Absolute Pearson correlation per feature.
    """
    X_arr = np.asarray(_as_dense_array(X), dtype=float)
    y_arr = np.asarray(y, dtype=float).ravel()
    n = X_arr.shape[0]
    y_centered = y_arr - y_arr.mean()
    X_centered = X_arr - X_arr.mean(axis=0)
    cov = X_centered.T @ y_centered / max(n - 1, 1)
    y_std = y_arr.std(ddof=1)
    X_std = X_arr.std(axis=0, ddof=1)
    denom = X_std * y_std
    denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
    return np.abs(cov / denom)


class MRMRRanker:
    """Stateful importance getter implementing the Minimum Redundancy
    Maximum Relevance score, designed for `ForwardSelectorCV`.

    By default both relevance (feature-vs-target) and redundance
    (feature-vs-already-selected-feature) are computed with mutual
    information, which handles continuous and categorical features
    transparently when ``discrete_features`` is supplied. Both functions
    can be swapped out via ``relevance_func`` / ``redundance_func``.

    The ranker keeps internal caches across calls within a single forward
    selection run. Caches are reset whenever it is called with an empty
    ``selected_idx`` (which `ForwardSelectorCV` always does at the start
    of every ``fit``).

    Parameters
    ----------
    regression : bool, default=False
        Whether the target is continuous. Switches the default
        relevance between `mutual_info_regression` and
        `mutual_info_classif`. Ignored when ``relevance_func`` is set.
    scheme : {'ratio', 'difference'}, default='difference'
        How to combine relevance and redundance:

        - ``'ratio'``: ``relevance / redundance`` (MIQ-style).
        - ``'difference'``: ``relevance - redundance`` (MID-style).
    n_neighbors : int, default=3
        Number of neighbors used by the default mutual information
        estimators. Ignored when both functions are overridden.
    discrete_features : 'auto', bool, or array-like, default='auto'
        Indicates which input features are categorical. Accepted formats
        match `sklearn.feature_selection.mutual_info_classif`:

        - ``'auto'``: infer from dtype when ``X`` is a
          :class:`pandas.DataFrame` — columns with categorical, string,
          or object dtype are treated as discrete; all others as
          continuous. Falls back to all-continuous for plain arrays.
        - ``True``: treat all features as discrete.
        - boolean mask of shape ``(n_features,)``.
        - integer array of indices of the discrete features.

        Used by the default relevance and redundance functions, both to
        tell the mutual information estimator which inputs are
        categorical and to decide whether to use the classifier or
        regressor estimator when a categorical feature is the target of
        a redundance computation. Ignored when both
        ``relevance_func`` and ``redundance_func`` are overridden.
    random_state : int, RandomState instance or None, default=None
        Seed used by the default mutual information estimators.
    n_jobs : int or None, default=None
        Forwarded to the default mutual information estimators.
    relevance_func : callable, default=None
        Optional override for the relevance computation. Signature:
        ``relevance_func(X, y) -> ndarray`` of shape ``(n_features,)``,
        scoring each feature against the target. When ``None`` (default),
        mutual information is used (handles categorical features via
        ``discrete_features``). Use `abs_pearson_correlation` for a fast
        Pearson-based alternative on purely numeric data.
    redundance_func : callable, default=None
        Optional override for the redundance computation. Signature:
        ``redundance_func(X, y_feature) -> ndarray`` of shape
        ``(n_features,)``, scoring each feature against the
        already-selected ``y_feature``. When ``None`` (default), mutual
        information is used (the classifier vs. regressor estimator is
        chosen based on whether the target column is marked as
        categorical in ``discrete_features``).
    min_relevance : float or None, default=0.05
        If set, features whose relevance score is strictly below this
        threshold are assigned ``-inf`` and will never be selected.
        Applied at every call, including the first (no selected features).
    max_redundancy : float or None, default=0.3
        If set, features whose mean redundancy with the already-selected
        features exceeds this threshold are assigned ``-inf`` and will
        not be selected in that round. Only applied when at least one
        feature has already been selected.

    Attributes
    ----------
    relevance_ : ndarray of shape (n_features,)
        Per-feature relevance, populated on the first call.
    """

    def __init__(
        self,
        regression=False,
        scheme="difference",
        n_neighbors=3,
        discrete_features="auto",
        random_state=None,
        n_jobs=None,
        relevance_func=None,
        redundance_func=None,
        min_relevance=0.05,
        max_redundancy=0.3,
    ):
        if scheme not in ("ratio", "difference"):
            raise ValueError(f"scheme must be 'ratio' or 'difference', got {scheme!r}")
        self.regression = regression
        self.scheme = scheme
        self.n_neighbors = n_neighbors
        self.discrete_features = discrete_features
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.relevance_func = relevance_func
        self.redundance_func = redundance_func
        self.min_relevance = min_relevance
        self.max_redundancy = max_redundancy
        self._reset()

    def _reset(self):
        self.relevance_ = None
        self._redundance_sum = None
        self._seen = set()
        self._discrete_mask = None

    def _resolve_discrete_mask(self, X):
        n = X.shape[1]
        df = self.discrete_features
        if isinstance(df, str):
            if isinstance(X, pd.DataFrame):
                mask = np.zeros(n, dtype=bool)
                for i, col in enumerate(X.columns):
                    dtype = X[col].dtype
                    if (
                        isinstance(dtype, (pd.CategoricalDtype, pd.StringDtype))
                        or dtype == object
                    ):
                        mask[i] = True
                return mask
            return np.zeros(n, dtype=bool)
        if isinstance(df, bool):
            return np.full(n, df, dtype=bool)
        arr = np.asarray(df)
        if arr.dtype == bool:
            if arr.shape != (n,):
                raise ValueError(
                    f"discrete_features mask has shape {arr.shape}, "
                    f"expected ({n},)."
                )
            return arr
        mask = np.zeros(n, dtype=bool)
        mask[arr] = True
        return mask

    def _default_relevance(self, X_arr, y):
        mi_func = mutual_info_regression if self.regression else mutual_info_classif
        return mi_func(
            X_arr,
            y,
            discrete_features=self._discrete_mask,
            n_neighbors=self.n_neighbors,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

    def _default_redundance(self, X_arr, y_feature, target_is_discrete):
        mi_func = mutual_info_classif if target_is_discrete else mutual_info_regression
        return mi_func(
            X_arr,
            y_feature,
            discrete_features=self._discrete_mask,
            n_neighbors=self.n_neighbors,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

    def _compute_relevance(self, X_arr, y):
        if self.relevance_func is not None:
            return np.asarray(self.relevance_func(X_arr, y), dtype=float)
        return np.asarray(self._default_relevance(X_arr, y), dtype=float)

    def _compute_redundance(self, X_arr, idx):
        y_feature = X_arr[:, idx]
        if self.redundance_func is not None:
            return np.asarray(self.redundance_func(X_arr, y_feature), dtype=float)
        return np.asarray(
            self._default_redundance(X_arr, y_feature, self._discrete_mask[idx]),
            dtype=float,
        )

    def _combine(self, relevance, redundance):
        if self.scheme == "difference":
            return relevance - redundance
        denom = np.where(np.abs(redundance) < 1e-10, 1e-10, redundance)
        return relevance / denom

    def __call__(self, X, y, selected_idx):
        X_arr = _as_dense_array(X)
        n_features = X_arr.shape[1]

        if not selected_idx:
            self._reset()
            self._discrete_mask = self._resolve_discrete_mask(X)
            self.relevance_ = self._compute_relevance(X_arr, y)
            self._redundance_sum = np.zeros(n_features, dtype=float)
            scores = self.relevance_.copy()
            if self.min_relevance is not None:
                scores[self.relevance_ < self.min_relevance] = -np.inf
            return scores

        for i in selected_idx:
            if i not in self._seen:
                self._redundance_sum += self._compute_redundance(X_arr, i)
                self._seen.add(i)

        mean_red = self._redundance_sum / max(len(self._seen), 1)
        scores = self._combine(self.relevance_, mean_red)
        if self.min_relevance is not None:
            scores[self.relevance_ < self.min_relevance] = -np.inf
        if self.max_redundancy is not None:
            scores[mean_red > self.max_redundancy] = -np.inf
        return scores


class MRMRCV(ForwardSelectorCV):
    """Preset of `ForwardSelectorCV` wired for forward MRMR selection.

    Equivalent to ``ForwardSelectorCV(estimator,
    importance_getter=MRMRRanker(...), ...)``.

    Parameters
    ----------
    estimator : ``Estimator`` instance
        A supervised learning estimator used to score candidate feature
        subsets via cross-validation. Used to detect classification vs.
        regression for the default ranker (via `is_classifier`) and for
        the CV scoring loop.
    step : int or float, default=1
        See `ForwardSelectorCV`.
    min_features_to_select : int, default=None
        See `ForwardSelectorCV`.
    max_features_to_select : int, default=None
        See `ForwardSelectorCV`.
    cv : int, cross-validation generator or an iterable, default=None
        See `ForwardSelectorCV`.
    scoring : str, callable or None, default=None
        See `ForwardSelectorCV`.
    verbose : int, default=0
        See `ForwardSelectorCV`.
    n_jobs : int or None, default=None
        Forwarded to `ForwardSelectorCV` and to `MRMRRanker`.
    random_state : int, RandomState instance or None, default=None
        Forwarded to `MRMRRanker` and used by `plot`.
    scheme : {'ratio', 'difference'}, default='difference'
        MRMR combination scheme. See `MRMRRanker`.
    n_neighbors : int, default=3
        Mutual information estimator parameter. See `MRMRRanker`.
    discrete_features : 'auto', bool or array-like, default='auto'
        Indicates which input features are categorical. See
        `MRMRRanker`.
    relevance_func : callable, default=None
        Optional override for the relevance computation. See
        `MRMRRanker`.
    redundance_func : callable, default=None
        Optional override for the redundance computation. See
        `MRMRRanker`.
    min_relevance : float or None, default=0.05
        Minimum relevance threshold. See `MRMRRanker`.
    max_redundancy : float or None, default=0.3
        Maximum redundancy threshold. See `MRMRRanker`.
    callbacks : list of callable, default=None
        See `ForwardSelectorCV`.
    best_iteration_selection_criteria : str or callable, default='mean_test_score'
        See `ForwardSelectorCV`.

    Examples
    --------
    >>> from felimination.mrmr import MRMRCV
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.linear_model import LogisticRegression
    >>> X, y = make_classification(n_samples=200, n_features=10, random_state=0)
    >>> selector = MRMRCV(
    ...     LogisticRegression(),
    ...     min_features_to_select=2,
    ...     max_features_to_select=8,
    ...     step=1,
    ...     cv=3,
    ...     random_state=0,
    ... ).fit(X, y)
    >>> selector.support_.sum() > 0
    True
    """

    def __init__(
        self,
        estimator,
        *,
        step=1,
        min_features_to_select=None,
        max_features_to_select=None,
        cv=None,
        scoring=None,
        verbose=0,
        n_jobs=None,
        random_state=None,
        scheme="difference",
        n_neighbors=3,
        discrete_features="auto",
        relevance_func=None,
        redundance_func=None,
        min_relevance=0.05,
        max_redundancy=0.3,
        callbacks=None,
        best_iteration_selection_criteria="mean_test_score",
    ) -> None:
        self.scheme = scheme
        self.n_neighbors = n_neighbors
        self.discrete_features = discrete_features
        self.relevance_func = relevance_func
        self.redundance_func = redundance_func
        self.min_relevance = min_relevance
        self.max_redundancy = max_redundancy
        super().__init__(
            estimator,
            step=step,
            min_features_to_select=min_features_to_select,
            max_features_to_select=max_features_to_select,
            cv=cv,
            scoring=scoring,
            verbose=verbose,
            n_jobs=n_jobs,
            random_state=random_state,
            importance_getter=MRMRRanker(
                regression=not is_classifier(estimator),
                scheme=scheme,
                n_neighbors=n_neighbors,
                discrete_features=discrete_features,
                random_state=random_state,
                n_jobs=n_jobs,
                relevance_func=relevance_func,
                redundance_func=redundance_func,
                min_relevance=min_relevance,
                max_redundancy=max_redundancy,
            ),
            callbacks=callbacks,
            best_iteration_selection_criteria=best_iteration_selection_criteria,
        )
