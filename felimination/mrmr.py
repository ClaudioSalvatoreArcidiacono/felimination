"""Module with tools to perform forward feature selection using the
Minimum Redundancy Maximum Relevance (MRMR) framework.

This module contains:

- `MRMRRanker`: stateful importance-getter callable implementing the MRMR
    score (mutual information relevance / absolute Pearson correlation
    redundance), suitable for use with [`ForwardSelectorCV`](/felimination/reference/selectors/forward_selection/#felimination.forward.ForwardSelectorCV).
- `MRMRCV`: preset of [`ForwardSelectorCV`](/felimination/reference/selectors/forward_selection/#felimination.forward.ForwardSelectorCV) wired with [`MRMRRanker`](/felimination/reference/selectors/MRMR/#felimination.mrmr.MRMRRanker).
"""

import numpy as np
import pandas as pd
from sklearn.base import clone, is_classifier
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.utils import check_random_state

from felimination.forward import ForwardSelectorCV


def _as_dense_array(X):
    if isinstance(X, pd.DataFrame):
        return X.values
    return np.asarray(X)


def abs_pearson_correlation(X, y):
    """Absolute Pearson correlation between each column of ``X`` and ``y``.

    Convenience helper for use as ``relevance_func`` or
    ``redundance_func`` in [`MRMRRanker`](/felimination/reference/selectors/MRMR/#felimination.mrmr.MRMRRanker). Only suitable for numeric data;
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
    Maximum Relevance score, designed for [`ForwardSelectorCV`](/felimination/reference/selectors/forward_selection/#felimination.forward.ForwardSelectorCV).

    By default both relevance (feature-vs-target) and redundance
    (feature-vs-already-selected-feature) are computed with mutual
    information, which handles continuous and categorical features
    transparently when ``discrete_features`` is supplied. Both functions
    can be swapped out via ``relevance_func`` / ``redundance_func``.

    The ranker keeps internal caches across calls within a single forward
    selection run. Caches are reset whenever it is called with an empty
    ``selected_idx`` (which [`ForwardSelectorCV`](/felimination/reference/selectors/forward_selection/#felimination.forward.ForwardSelectorCV) always does at the start
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
    redundancy_aggregation : {'max', 'mean'} or callable, default='max'
        How to aggregate per-selected-feature redundancy scores into a
        single redundancy value before combining with relevance:

        - ``'max'``: take the element-wise maximum across all
          already-selected features. A candidate is penalised as soon as
          it is highly redundant with *any* selected feature, making the
          criterion more conservative.
        - ``'mean'``: take the element-wise mean, matching the
          formulation in the original MRMR paper (Peng et al., 2005).
        - callable: a function with signature
          ``f(redundancy_matrix) -> ndarray`` of shape ``(n_features,)``,
          where ``redundancy_matrix`` has shape
          ``(n_selected, n_features)``. Rows correspond to
          already-selected features; columns to candidate features.

        > Note:
        >    The default ``'max'`` deviates from the original MRMR paper,
        >    which uses the mean. ``'max'`` is chosen as the default
        >    because it more aggressively avoids adding features that
        >    duplicate information already captured, which tends to work
        >    better in practice for forward selection with CV scoring.
    min_relevance_perc : float or None, default=0.01
        If set, features are filtered based on cumulative relevance. After
        computing relevance scores, a minimum relevance threshold is derived
        as ``min_relevance_perc * sum(relevance scores)``. Features are then
        ordered by relevance ascending and their cumulative relevance is
        computed; any feature whose cumulative relevance (from the least
        relevant up to and including itself) is strictly below the threshold
        is assigned ``-inf`` and will never be selected. This removes the
        low-relevance tail that together contributes less than
        ``min_relevance_perc`` of the total relevance.
    max_redundancy : float or None, default=None
        If set, features whose aggregated redundancy with the
        already-selected features exceeds this threshold are assigned
        ``-inf`` and will not be selected in that round. The aggregation
        is controlled by ``redundancy_aggregation``. Only applied when at
        least one feature has already been selected.
    discrete_imputer : sklearn-compatible transformer or None, default=None
        Imputer applied to discrete (categorical) feature columns before
        encoding. When ``None``, defaults to
        ``SimpleImputer(strategy='constant', fill_value='MISSING')``,
        replacing missing values with the string ``'MISSING'`` (treated
        as an additional category). Pass any sklearn-compatible transformer
        with ``fit``/``transform``. Ignored when there are no discrete
        columns.
    continuous_imputer : sklearn-compatible transformer or None, default=None
        Imputer applied to continuous (numeric) feature columns before the
        mutual information computation. When ``None``, defaults to
        ``SimpleImputer(strategy='median')``. Pass any sklearn-compatible
        transformer with ``fit``/``transform``. Ignored when there are no
        continuous columns. For arrays with non-object ``dtype``, applied
        to all columns regardless of ``discrete_features``.
    max_samples : int, float or None, default=None
        Number of samples used when computing mutual information scores.
        Imputers are still fitted on the full training set; only the MI
        scoring (relevance on the first call and redundance on subsequent
        calls) uses the subsample.

        - ``None``: use all samples (no subsampling).
        - ``int``: use exactly this many samples (capped at ``n_samples``).
        - ``float`` in ``(0.0, 1.0]``: use this fraction of the training
          set (at least 1 sample).

        The same row indices are drawn once per forward-selection run
        (controlled by ``random_state``) and reused for every subsequent
        redundance computation, keeping relevance and redundance comparable.

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
        redundancy_aggregation="max",
        min_relevance_perc=0.01,
        max_redundancy=None,
        discrete_imputer=None,
        continuous_imputer=None,
        max_samples=None,
    ):
        if scheme not in ("ratio", "difference"):
            raise ValueError(f"scheme must be 'ratio' or 'difference', got {scheme!r}")
        if not callable(redundancy_aggregation) and redundancy_aggregation not in (
            "max",
            "mean",
        ):
            raise ValueError(
                f"redundancy_aggregation must be 'max', 'mean', or a callable, "
                f"got {redundancy_aggregation!r}"
            )
        self.regression = regression
        self.scheme = scheme
        self.n_neighbors = n_neighbors
        self.discrete_features = discrete_features
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.relevance_func = relevance_func
        self.redundance_func = redundance_func
        self.redundancy_aggregation = redundancy_aggregation
        self.min_relevance_perc = min_relevance_perc
        self.max_redundancy = max_redundancy
        self.discrete_imputer = discrete_imputer
        self.continuous_imputer = continuous_imputer
        self.max_samples = max_samples
        self._reset()

    def _reset(self):
        self.relevance_ = None
        self._redundance_sum = None
        self._redundance_max = None
        self._seen = set()
        self._discrete_mask = None
        self._low_relevance_mask = None
        self._fitted_discrete_imputer = None
        self._fitted_continuous_imputer = None
        self._sampled_indices = None

    def _resolve_discrete_mask(self, X):
        n = X.shape[1]
        df = self.discrete_features
        if isinstance(df, str):
            if isinstance(X, pd.DataFrame):
                mask = np.zeros(n, dtype=bool)
                for i, col in enumerate(X.columns):
                    dtype = X[col].dtype
                    if (
                        isinstance(
                            dtype,
                            (pd.CategoricalDtype, pd.StringDtype, pd.BooleanDtype),
                        )
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

    def _draw_sample_indices(self, n_rows):
        if self.max_samples is None:
            return None
        if isinstance(self.max_samples, float):
            n = max(1, int(n_rows * self.max_samples))
        else:
            n = min(int(self.max_samples), n_rows)
        rng = check_random_state(self.random_state)
        return rng.choice(n_rows, size=n, replace=False)

    def _fit_imputers(self, X_arr):
        self._fitted_discrete_imputer = None
        self._fitted_continuous_imputer = None
        disc_cols = np.where(self._discrete_mask)[0]
        cont_cols = np.where(~self._discrete_mask)[0]
        if len(disc_cols) > 0:
            imp = self.discrete_imputer
            if imp is None:
                imp = SimpleImputer(strategy="constant", fill_value="MISSING")
            # astype(object) keeps NaN detectable while handling numeric discrete cols
            self._fitted_discrete_imputer = clone(imp).fit(
                X_arr[:, disc_cols].astype(object)
            )
        if len(cont_cols) > 0:
            imp = self.continuous_imputer
            if imp is None:
                imp = SimpleImputer(strategy="median")
            self._fitted_continuous_imputer = clone(imp).fit(
                self._to_float_array(X_arr[:, cont_cols])
            )

    def _to_float_array(self, arr):
        if arr.dtype != object:
            return arr.astype(float)
        n_rows, n_cols = arr.shape
        result = np.empty((n_rows, n_cols), dtype=float)
        for j in range(n_cols):
            result[:, j] = pd.to_numeric(arr[:, j], errors="coerce")
        return result

    def _impute_X(self, X_arr):
        disc_cols = np.where(self._discrete_mask)[0]
        cont_cols = np.where(~self._discrete_mask)[0]
        has_disc = len(disc_cols) > 0 and self._fitted_discrete_imputer is not None
        has_cont = len(cont_cols) > 0 and self._fitted_continuous_imputer is not None
        if not has_disc and not has_cont:
            return X_arr
        # When discrete columns are present in a float array, promote to object so
        # the string fill value ('MISSING') can coexist with float continuous columns.
        result = (
            X_arr.astype(object) if X_arr.dtype != object and has_disc else X_arr.copy()
        )
        if has_disc:
            result[:, disc_cols] = self._fitted_discrete_imputer.transform(
                result[:, disc_cols].astype(object)
            )
        if has_cont:
            imputed = self._fitted_continuous_imputer.transform(
                self._to_float_array(result[:, cont_cols])
            )
            result[:, cont_cols] = imputed
        if not has_disc and X_arr.dtype != object:
            # Pure float array with only continuous imputation — keep float dtype
            return result.astype(float)
        return result

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

    def _default_redundance(
        self, X_arr, y_feature, target_is_discrete, discrete_mask=None
    ):
        mi_func = mutual_info_classif if target_is_discrete else mutual_info_regression
        if discrete_mask is None:
            discrete_mask = self._discrete_mask
        return mi_func(
            X_arr,
            y_feature,
            discrete_features=discrete_mask,
            n_neighbors=self.n_neighbors,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

    def _compute_relevance(self, X_arr, y):
        if self.relevance_func is not None:
            return np.asarray(self.relevance_func(X_arr, y), dtype=float)
        return np.asarray(self._default_relevance(X_arr, y), dtype=float)

    def _compute_redundance(self, X_arr, idx, candidate_mask=None):
        n_features = X_arr.shape[1]
        y_feature = X_arr[:, idx]
        if self.redundance_func is not None:
            return np.asarray(self.redundance_func(X_arr, y_feature), dtype=float)
        if candidate_mask is not None:
            red_sub = np.asarray(
                self._default_redundance(
                    X_arr[:, candidate_mask],
                    y_feature,
                    self._discrete_mask[idx],
                    self._discrete_mask[candidate_mask],
                ),
                dtype=float,
            )
            red = np.zeros(n_features, dtype=float)
            red[candidate_mask] = red_sub
            return red
        return np.asarray(
            self._default_redundance(X_arr, y_feature, self._discrete_mask[idx]),
            dtype=float,
        )

    def _encode_X(self, X_arr):
        """Return a float copy of X_arr with object columns label-encoded."""
        if X_arr.dtype != object:
            return X_arr.astype(float)
        result = np.empty(X_arr.shape, dtype=float)
        for j in range(X_arr.shape[1]):
            col = X_arr[:, j]
            if self._discrete_mask[j]:
                codes, _ = pd.factorize(col)
                result[:, j] = codes.astype(int)
            else:
                result[:, j] = col.astype(float)
        return result

    def _combine(self, relevance, redundance):
        if self.scheme == "difference":
            return relevance - redundance
        denom = np.where(np.abs(redundance) < 1e-10, 1e-10, redundance)
        return relevance / denom

    def _aggregate_redundance(self):
        agg = self.redundancy_aggregation
        if agg == "mean":
            return self._redundance_sum / max(len(self._seen), 1)
        if agg == "max":
            return self._redundance_max
        matrix = np.vstack(self._redundance_rows)
        return np.asarray(agg(matrix), dtype=float)

    def __call__(self, X, y, selected_idx):
        X_arr = _as_dense_array(X)
        n_features = X_arr.shape[1]

        if not selected_idx:
            self._reset()
            self._discrete_mask = self._resolve_discrete_mask(X)
            self._fit_imputers(X_arr)
            X_arr = self._impute_X(X_arr)
            X_arr = self._encode_X(X_arr)
            self._sampled_indices = self._draw_sample_indices(X_arr.shape[0])
            if self._sampled_indices is not None:
                X_sample = X_arr[self._sampled_indices]
                y_sample = np.asarray(y)[self._sampled_indices]
            else:
                X_sample, y_sample = X_arr, y
            self.relevance_ = self._compute_relevance(X_sample, y_sample)
            self._redundance_sum = np.zeros(n_features, dtype=float)
            self._redundance_max = np.full(n_features, -np.inf, dtype=float)
            self._redundance_rows = []
            scores = self.relevance_.copy()
            if self.min_relevance_perc is not None:
                total = self.relevance_.sum()
                min_rel = self.min_relevance_perc * total
                sorted_idx = np.argsort(self.relevance_)
                cumsum = np.cumsum(self.relevance_[sorted_idx])
                self._low_relevance_mask = np.zeros(n_features, dtype=bool)
                self._low_relevance_mask[sorted_idx[cumsum < min_rel]] = True
                scores[self._low_relevance_mask] = -np.inf
            return scores

        X_arr = self._impute_X(X_arr)
        X_arr = self._encode_X(X_arr)
        if self._sampled_indices is not None:
            X_arr = X_arr[self._sampled_indices]
        candidate_mask = (
            ~self._low_relevance_mask if self._low_relevance_mask is not None else None
        )
        for i in selected_idx:
            if i not in self._seen:
                red = self._compute_redundance(X_arr, i, candidate_mask)
                self._redundance_sum += red
                self._redundance_max = np.maximum(self._redundance_max, red)
                self._redundance_rows.append(red)
                self._seen.add(i)

        agg_red = self._aggregate_redundance()
        scores = self._combine(self.relevance_, agg_red)
        if self._low_relevance_mask is not None:
            scores[self._low_relevance_mask] = -np.inf
        if self.max_redundancy is not None:
            scores[agg_red > self.max_redundancy] = -np.inf
        return scores


class MRMRCV(ForwardSelectorCV):
    """Forward feature selector using Minimum Redundancy Maximum Relevance (MRMR) scoring.

    Performs forward feature selection driven by MRMR scores, using
    cross-validation to determine the optimal number of features.

    The selector starts by ranking all features by their relevance to the
    target and picks the highest-scoring one. It then iteratively selects
    the feature that maximises relevance minus (or divided by) redundance
    with already-selected features. Cross-validation scores the model at
    each evaluated step.

    By default both relevance (feature-vs-target) and redundance
    (feature-vs-already-selected-feature) are computed with mutual
    information, which handles continuous and categorical features
    transparently when ``discrete_features`` is supplied. Both functions
    can be swapped out via ``relevance_func`` / ``redundance_func``.

    Parameters
    ----------
    estimator : ``Estimator`` instance
        A supervised learning estimator with a ``fit`` method, used to
        score candidate feature subsets via cross-validation. Also used to
        detect classification vs. regression for the MRMR ranker (via
        ``is_classifier``).
    step : int or float, default=1
        Number of features added between two consecutive cross-validation
        evaluations. If greater than or equal to 1, this is the integer
        number of features added per evaluation. If within (0.0, 1.0), it
        is the fraction (rounded down, with a floor of 1) of the already-
        selected features added per evaluation, growing the selection
        geometrically. Selection within a step still happens one feature
        at a time.
    min_features_to_select : int, default=None
        Minimum number of features that must be selected before the first
        cross-validation evaluation. Features are still selected via MRMR
        scoring before this threshold, but no CV scoring takes place. If
        ``None``, defaults to 1 (CV evaluation starts from the very first
        selected feature).
    max_features_to_select : int, default=None
        Maximum number of features to select. The forward process stops
        once this many features have been selected. If ``None``, defaults
        to all features in ``X``.
    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy. See
        `~sklearn.model_selection.check_cv` for accepted inputs.
    scoring : str, callable or None, default=None
        Scorer used to evaluate the estimator on each CV fold.
    verbose : int, default=0
        Controls verbosity of output.
    n_jobs : int or None, default=None
        Number of cores to run in parallel while fitting across folds.
        Also forwarded to the default mutual information estimators used
        for MRMR scoring.
    random_state : int, RandomState instance or None, default=None
        Seed used by the default mutual information estimators and by
        ``plot``.
    scheme : {'ratio', 'difference'}, default='difference'
        How to combine relevance and redundance:

        - ``'ratio'``: ``relevance / redundance`` (MIQ-style).
        - ``'difference'``: ``relevance - redundance`` (MID-style).
    n_neighbors : int, default=3
        Number of neighbors used by the default mutual information
        estimators. Ignored when both ``relevance_func`` and
        ``redundance_func`` are overridden.
    discrete_features : 'auto', bool or array-like, default='auto'
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
    redundancy_aggregation : {'max', 'mean'} or callable, default='max'
        How to aggregate per-selected-feature redundancy scores into a
        single redundancy value before combining with relevance:

        - ``'max'``: take the element-wise maximum across all
          already-selected features. A candidate is penalised as soon as
          it is highly redundant with *any* selected feature, making the
          criterion more conservative.
        - ``'mean'``: take the element-wise mean, matching the
          formulation in the original MRMR paper (Peng et al., 2005).
        - callable: a function with signature
          ``f(redundancy_matrix) -> ndarray`` of shape ``(n_features,)``,
          where ``redundancy_matrix`` has shape
          ``(n_selected, n_features)``. Rows correspond to
          already-selected features; columns to candidate features.

        > Note:
        >    The default ``'max'`` deviates from the original MRMR paper,
        >    which uses the mean. ``'max'`` is chosen as the default
        >    because it more aggressively avoids adding features that
        >    duplicate information already captured, which tends to work
        >    better in practice for forward selection with CV scoring.
    min_relevance_perc : float or None, default=0.01
        If set, features are filtered based on cumulative relevance. After
        computing relevance scores, a minimum relevance threshold is derived
        as ``min_relevance_perc * sum(relevance scores)``. Features are then
        ordered by relevance ascending and their cumulative relevance is
        computed; any feature whose cumulative relevance (from the least
        relevant up to and including itself) is strictly below the threshold
        is assigned ``-inf`` and will never be selected. This removes the
        low-relevance tail that together contributes less than
        ``min_relevance_perc`` of the total relevance.
    max_redundancy : float or None, default=None
        If set, features whose aggregated redundancy with the
        already-selected features exceeds this threshold are assigned
        ``-inf`` and will not be selected in that round. The aggregation
        is controlled by ``redundancy_aggregation``. Only applied when at
        least one feature has already been selected.
    discrete_imputer : sklearn-compatible transformer or None, default=None
        Forwarded to :class:`MRMRRanker`. Imputer for discrete (categorical)
        columns. When ``None``, defaults to
        ``SimpleImputer(strategy='constant', fill_value='MISSING')``.
    continuous_imputer : sklearn-compatible transformer or None, default=None
        Forwarded to :class:`MRMRRanker`. Imputer for continuous (numeric)
        columns. When ``None``, defaults to ``SimpleImputer(strategy='median')``.
    max_samples : int, float or None, default=None
        Forwarded to :class:`MRMRRanker`. Number of samples used when
        computing mutual information scores. ``None`` means all samples.
        See :class:`MRMRRanker` for the full description.
    callbacks : list of callable, default=None
        List of callables called at the end of each evaluated step. Each
        callable receives ``(selector, scores)`` where ``scores`` is the
        last array of MRMR scores.
    best_iteration_selection_criteria : str or callable, default='mean_test_score'
        Either a key into ``cv_results_`` (the iteration that maximises
        that key is picked) or a callable
        ``f(cv_results) -> n_features`` that must return one of the values
        in ``cv_results_["n_features"]``.

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
        redundancy_aggregation="max",
        min_relevance_perc=0.01,
        max_redundancy=None,
        discrete_imputer=None,
        continuous_imputer=None,
        max_samples=None,
        callbacks=None,
        best_iteration_selection_criteria="mean_test_score",
    ) -> None:
        self.scheme = scheme
        self.n_neighbors = n_neighbors
        self.discrete_features = discrete_features
        self.relevance_func = relevance_func
        self.redundance_func = redundance_func
        self.redundancy_aggregation = redundancy_aggregation
        self.min_relevance_perc = min_relevance_perc
        self.max_redundancy = max_redundancy
        self.discrete_imputer = discrete_imputer
        self.continuous_imputer = continuous_imputer
        self.max_samples = max_samples
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
                redundancy_aggregation=redundancy_aggregation,
                min_relevance_perc=min_relevance_perc,
                max_redundancy=max_redundancy,
                discrete_imputer=discrete_imputer,
                continuous_imputer=continuous_imputer,
                max_samples=max_samples,
            ),
            callbacks=callbacks,
            best_iteration_selection_criteria=best_iteration_selection_criteria,
        )
