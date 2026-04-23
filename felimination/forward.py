"""Module with tools to perform forward feature selection with
cross-validation.

This module contains:

- `ForwardSelectorCV`: forward feature selector driven by a pluggable
    per-step importance getter, with cross-validation to choose the best
    number of features.
"""

from collections import defaultdict
from numbers import Integral

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import (
    BaseEstimator,
    MetaEstimatorMixin,
    _fit_context,
    clone,
    is_classifier,
)
from sklearn.feature_selection import (
    SelectorMixin,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.metrics import check_scoring
from sklearn.model_selection import check_cv, cross_validate
from sklearn.utils import Bunch
from sklearn.utils._metadata_requests import _routing_enabled, process_routing
from sklearn.utils._param_validation import HasMethods, Interval, RealNotInt, StrOptions
from sklearn.utils._tags import get_tags
from sklearn.utils.validation import check_is_fitted, validate_data


class ForwardSelectorCV(MetaEstimatorMixin, SelectorMixin, BaseEstimator):
    """Forward feature selection with cross-validation.

    The selector starts by asking ``importance_getter`` for scores against
    an empty selection and picks the top-scoring feature. It then
    iteratively asks the importance getter again — passing the indices of
    already-selected features — and adds the highest-scoring not-yet-
    selected feature. Cross-validation is used to score the model trained
    on the running selection.

    The algorithm:
    ```
    scores = importance_getter(X, y, [])
    selected = [argmax(scores)]
    while len(selected) < max_features_to_select:
        scores = importance_getter(X, y, selected)
        scores[already_selected] = -inf
        selected.append(argmax(scores))
        if len(selected) >= min_features_to_select and
           (len(selected) - min_features_to_select) is a multiple of step:
            evaluate(selected) via cross-validation
    ```

    Parameters
    ----------
    estimator : ``Estimator`` instance
        A supervised learning estimator with a ``fit`` method, used to
        score candidate feature subsets via cross-validation. The estimator
        is not used to drive selection.
    step : int or float, default=1
        Number of features added between two consecutive cross-validation
        evaluations. If greater than or equal to 1, this is the integer
        number of features added per evaluation. If within (0.0, 1.0), it
        is the fraction (rounded down, with a floor of 1) of the already-
        selected features added per evaluation, growing the selection
        geometrically. Selection within a step
        still happens one feature at a time, calling
        ``importance_getter`` after every addition.
    min_features_to_select : int, default=None
        Minimum number of features that must be selected before the first
        cross-validation evaluation. Features are still selected via the
        importance getter before this threshold, but no CV scoring takes
        place. If ``None``, defaults to 1 (CV evaluation starts from the
        very first selected feature).
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
    random_state : int, RandomState instance or None, default=None
        Seed used by the default mutual-information importance getter and
        by `plot`.
    importance_getter : 'auto' or callable, default='auto'
        Feature scoring strategy used to drive selection.

        - ``'auto'``: use `mutual_info_classif` when ``estimator`` is a
          classifier, otherwise `mutual_info_regression`. Scores are
          computed once on the full ``(X, y)`` and reused for every step,
          so the order of selection is just descending mutual information.
        - callable: a function with signature
          ``importance_getter(X, y, selected_idx) -> scores`` where
          ``selected_idx`` is a list of indices of currently-selected
          features and ``scores`` is an array of shape (n_features,). The
          feature with the highest score among those not in
          ``selected_idx`` is added next; already-selected features are
          masked by the selector, so the callable may return any value
          for them. The selector always starts a fresh selection by
          calling the callable with an empty list, so stateful scorers
          may use that signal to invalidate caches.
    callbacks : list of callable, default=None
        List of callables called at the end of each evaluated step. Each
        callable receives ``(selector, scores)`` where ``scores`` is the
        last array returned by ``importance_getter``.
    best_iteration_selection_criteria : str or callable, default='mean_test_score'
        Either a key into ``cv_results_`` (the iteration that maximises
        that key is picked) or a callable
        ``f(cv_results) -> n_features`` that must return one of the values
        in ``cv_results_["n_features"]``.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        The classes labels. Only available when `estimator` is a
        classifier.
    estimator_ : ``Estimator`` instance
        The estimator refit on the selected features.
    cv_results_ : dict of lists
        A dict with keys ``n_features``, ``mean_test_score``,
        ``std_test_score``, ``mean_train_score``, ``std_train_score`` and
        ``split{k}_{train,test}_score`` for each CV fold.
    n_features_ : int
        The number of selected features (after picking the best CV
        iteration).
    n_features_in_ : int
        Number of features seen during :term:`fit`.
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when ``X``
        has feature names that are all strings.
    ranking_ : ndarray of shape (n_features_in_,)
        The order in which features were selected. ``ranking_[i] == 1``
        means feature ``i`` was the first selected. Features that were
        never selected receive a rank greater than the highest assigned
        one.
    support_ : ndarray of shape (n_features_in_,)
        The mask of currently selected features. Can be changed via
        `set_n_features_to_select`.

    Examples
    --------
    >>> from felimination.forward import ForwardSelectorCV
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.linear_model import LogisticRegression
    >>> X, y = make_classification(n_samples=200, n_features=10, random_state=0)
    >>> selector = ForwardSelectorCV(
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

    _parameter_constraints: dict = {
        "estimator": [HasMethods(["fit"])],
        "step": [
            Interval(Integral, 1, None, closed="left"),
            Interval(RealNotInt, 0, 1, closed="neither"),
        ],
        "min_features_to_select": [
            None,
            Interval(Integral, 1, None, closed="left"),
        ],
        "max_features_to_select": [
            None,
            Interval(Integral, 1, None, closed="left"),
        ],
        "verbose": ["verbose"],
        "importance_getter": [StrOptions({"auto"}), callable],
        "best_iteration_selection_criteria": [str, callable],
    }

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
        importance_getter="auto",
        callbacks=None,
        best_iteration_selection_criteria="mean_test_score",
    ) -> None:
        self.estimator = estimator
        self.step = step
        self.min_features_to_select = min_features_to_select
        self.max_features_to_select = max_features_to_select
        self.cv = cv
        self.scoring = scoring
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.importance_getter = importance_getter
        self.callbacks = callbacks
        self.best_iteration_selection_criteria = best_iteration_selection_criteria

    def _get_support_mask(self):
        check_is_fitted(self)
        return self.support_

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        estimator_tags = get_tags(self.estimator)
        tags.input_tags.allow_nan = estimator_tags.input_tags.allow_nan
        tags.input_tags.sparse = False
        tags.target_tags.required = True
        return tags

    @staticmethod
    def _select_X_with_remaining_features(X, support):
        n_features = X.shape[1]
        features = np.arange(n_features)[support]
        if isinstance(X, pd.DataFrame):
            return X[X.columns[support]], features
        return X[:, features], features

    def _resolve_importance_getter(self):
        if callable(self.importance_getter):
            return self.importance_getter

        # 'auto' → mutual information ranker, computed once per fit.
        mi_func = (
            mutual_info_classif
            if is_classifier(self.estimator)
            else mutual_info_regression
        )
        cache = {}

        def _default_importance_getter(X, y, selected_idx):
            if not selected_idx or "scores" not in cache:
                X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
                cache["scores"] = mi_func(
                    X_arr,
                    y,
                    random_state=self.random_state,
                    n_jobs=self.n_jobs,
                )
            return cache["scores"]

        return _default_importance_getter

    def select_best_iteration(self, cv_results):
        """Return the best `n_features` value given ``cv_results_``."""
        if callable(self.best_iteration_selection_criteria):
            return self.best_iteration_selection_criteria(cv_results)
        return cv_results["n_features"][
            int(np.argmax(cv_results[self.best_iteration_selection_criteria]))
        ]

    @_fit_context(prefer_skip_nested_validation=False)
    def fit(self, X, y, groups=None, **params):
        if _routing_enabled():
            routed_params = process_routing(self, "fit", **params)
        else:
            routed_params = Bunch(
                estimator=Bunch(fit={}),
                splitter=Bunch(split={"groups": groups}),
                scorer=Bunch(score={}),
            )

        validate_data(
            self,
            X,
            y,
            ensure_min_features=2,
            ensure_all_finite=not get_tags(self.estimator).input_tags.allow_nan,
            multi_output=True,
            dtype=None,
        )

        cv = check_cv(self.cv, y, classifier=is_classifier(self.estimator))
        scorer = check_scoring(self.estimator, scoring=self.scoring)
        n_features = X.shape[1]

        min_features = self.min_features_to_select or 1
        max_features = self.max_features_to_select or n_features
        max_features = min(max_features, n_features)
        if min_features > max_features:
            raise ValueError(
                f"min_features_to_select ({min_features}) cannot be greater "
                f"than max_features_to_select ({max_features})."
            )

        importance_getter = self._resolve_importance_getter()
        support_ = np.zeros(n_features, dtype=bool)
        ranking_ = np.zeros(n_features, dtype=int)
        selected_idx = []
        last_scores = None

        def _pick_next():
            nonlocal last_scores
            last_scores = np.asarray(
                importance_getter(X, y, list(selected_idx)), dtype=float
            )
            if last_scores.shape != (n_features,):
                raise ValueError(
                    f"importance_getter returned shape {last_scores.shape}, "
                    f"expected ({n_features},)."
                )
            masked = last_scores.copy()
            masked[support_] = -np.inf
            return int(np.argmax(masked))

        def _evaluate(n_now):
            X_sel, _ = self._select_X_with_remaining_features(X, support=support_)
            if self.verbose > 0:
                print(f"Fitting estimator with {n_now} features.")
            cv_scores = cross_validate(
                self.estimator,
                X_sel,
                y,
                groups=groups,
                scoring=scorer,
                cv=cv,
                n_jobs=self.n_jobs,
                params=params,
                return_train_score=True,
            )
            for which in ("train", "test"):
                scores_per_fold = cv_scores[f"{which}_score"]
                for i, sc in enumerate(scores_per_fold):
                    self.cv_results_[f"split{i}_{which}_score"].append(sc)
                self.cv_results_[f"mean_{which}_score"].append(np.mean(scores_per_fold))
                self.cv_results_[f"std_{which}_score"].append(np.std(scores_per_fold))
            self.cv_results_["n_features"].append(n_now)

        first = _pick_next()
        support_[first] = True
        ranking_[first] = 1
        selected_idx.append(first)
        rank_counter = 2
        n_selected = 1

        self.cv_results_ = defaultdict(list)

        # Pre-eval phase: silently add features until reaching min_features.
        while n_selected < min_features and n_selected < max_features:
            nxt = _pick_next()
            support_[nxt] = True
            ranking_[nxt] = rank_counter
            selected_idx.append(nxt)
            rank_counter += 1
            n_selected += 1

        if n_selected >= min_features:
            _evaluate(n_selected)
            if self.callbacks:
                for callback in self.callbacks:
                    callback(self, last_scores)

        # Step-based selection with periodic CV evaluations.
        while n_selected < max_features:
            if 0.0 < self.step < 1.0:
                step = int(max(1, self.step * n_selected))
            else:
                step = int(self.step)
            features_to_add = min(step, max_features - n_selected)
            for _ in range(features_to_add):
                nxt = _pick_next()
                support_[nxt] = True
                ranking_[nxt] = rank_counter
                selected_idx.append(nxt)
                rank_counter += 1
                n_selected += 1
            _evaluate(n_selected)
            if self.callbacks:
                for callback in self.callbacks:
                    callback(self, last_scores)

        ranking_[ranking_ == 0] = rank_counter

        self.cv_results_ = dict(self.cv_results_)
        self.support_ = support_
        self.ranking_ = ranking_
        self.n_features_ = int(support_.sum())

        best_n_features = self.select_best_iteration(self.cv_results_)
        self.set_n_features_to_select(best_n_features)

        X_remaining_features, _ = self._select_X_with_remaining_features(
            X, support=self.support_
        )
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X_remaining_features, y, **routed_params.estimator.fit)
        if is_classifier(self.estimator_):
            self.classes_ = getattr(self.estimator_, "classes_", None)

        return self

    def set_n_features_to_select(self, n_features_to_select):
        """Change the number of selected features after fitting.

        The underlying estimator is **not** retrained — `predict` /
        `predict_proba` keep using the model fit on the originally
        selected features. Only `support_`, `transform` and
        `get_feature_names_out` are affected.

        Parameters
        ----------
        n_features_to_select : int
            Must be one of the values in ``cv_results_["n_features"]``.
        """
        check_is_fitted(self)
        if n_features_to_select not in self.cv_results_["n_features"]:
            raise ValueError(
                f"This selector has not been evaluated with "
                f"{n_features_to_select} features. Pick one of "
                f"{sorted(set(self.cv_results_['n_features']))}."
            )
        support_ = np.zeros_like(self.support_, dtype=bool)
        support_[np.argsort(self.ranking_)[:n_features_to_select]] = True
        self.support_ = support_
        self.n_features_ = n_features_to_select
        return self

    def plot(self, **kwargs):
        """Plot the cross-validation curve over number of features.

        Parameters
        ----------
        **kwargs : dict
            Forwarded to `seaborn.lineplot`.

        Returns
        -------
        matplotlib.axes.Axes
        """
        check_is_fitted(self)
        best_n = self.select_best_iteration(self.cv_results_)
        best_index = self.cv_results_["n_features"].index(best_n)
        best_train_score = self.cv_results_["mean_train_score"][best_index]
        best_test_score = self.cv_results_["mean_test_score"][best_index]
        df = pd.DataFrame(self.cv_results_)
        split_score_cols = [c for c in df if "split" in c]
        df_long = df[split_score_cols + ["n_features"]].melt(
            id_vars=["n_features"],
            value_vars=split_score_cols,
            var_name="split",
            value_name="score",
        )
        df_long["set"] = np.where(
            df_long["split"].str.contains("train"), "train", "validation"
        )
        lineplot_kwargs = dict(
            x="n_features",
            y="score",
            hue="set",
            markers=True,
            style="set",
            hue_order=["validation", "train"],
            style_order=["validation", "train"],
            seed=self.random_state,
            zorder=0,
        )
        lineplot_kwargs.update(**kwargs)
        ax = sns.lineplot(data=df_long, **lineplot_kwargs)
        ax.set_xticks(df.n_features)
        ax.plot(
            best_n,
            best_test_score,
            color="red",
            label="Best Iteration",
            zorder=1,
            marker="*",
            markersize=10,
            markeredgewidth=2,
            markeredgecolor="red",
            fillstyle="none",
        )
        ax.legend()
        ax.set_title(
            "\n".join(
                (
                    "Forward Feature Selection Plot",
                    f"Best Number of Features: {best_n}",
                    f"Best Test Score: {best_test_score:.3f}",
                    f"Best Train Score: {best_train_score:.3f}",
                )
            )
        )
        return ax
