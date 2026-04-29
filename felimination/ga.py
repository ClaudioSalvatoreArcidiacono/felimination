"""This module contains the implementation of the Hybrid Genetic Algorithm-Importance with
Cross-Validation. The algorithm is implemented in the `HybridImportanceGACVFeatureSelector` class.
"""

from itertools import cycle, islice
from numbers import Integral, Real
from operator import itemgetter
from typing import Any, Iterable

import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed, effective_n_jobs
from sklearn.base import (
    BaseEstimator,
    MetaEstimatorMixin,
    _fit_context,
    clone,
    is_classifier,
)
from sklearn.feature_selection import SelectorMixin
from sklearn.linear_model._logistic import LogisticRegression
from sklearn.metrics import check_scoring
from sklearn.model_selection import check_cv
from sklearn.utils import Bunch
from sklearn.utils._metadata_requests import _routing_enabled, process_routing
from sklearn.utils._param_validation import HasMethods, Interval, RealNotInt
from sklearn.utils._tags import get_tags
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_is_fitted, validate_data

from felimination.importance import _train_score_get_importance


def _estimator_has(attr):
    """Check if we can delegate a method to the underlying estimator.

    First, we check the fitted `estimator_` if available, otherwise we check the
    unfitted `estimator`. We raise the original `AttributeError` if `attr` does
    not exist. This function is used together with `available_if`.
    """

    def check(self):
        if hasattr(self, "estimator_"):
            getattr(self.estimator_, attr)
        else:
            getattr(self.estimator, attr)

        return True

    return check


def _select_X_with_features(X, features):
    if isinstance(X, pd.DataFrame):
        return X[features]
    else:
        return X[:, features]


def _roundrobin(*iterables: Iterable[Iterable[Any]]) -> Iterable[Any]:
    """Implements the round robin algorithm.

    Given a some iterables extracts the first item from each iterable in turn,
    until all iterables are exhausted.

    For example roundrobin("ABC", "123") -> ["A", "1", "B", "2", "C", "3"]

    taken from itertools recipes
    https://docs.python.org/3.9/library/itertools.html#itertools-recipes

    Yields:
        Iterable[Any]: items from the passed iterables in a round robin fashion.
    """
    num_active = len(iterables)
    nexts = cycle(iter(it).__next__ for it in iterables)
    while num_active:
        try:
            for next_ in nexts:
                yield next_()
        except StopIteration:
            # Remove the iterator we just exhausted from the cycle.
            num_active -= 1
            nexts = cycle(islice(nexts, num_active))


def _deduplicate(seq: list):
    seen = set()
    return [x for x in seq if not (x in seen or seen.add(x))]


def rank_mean_test_score_overfit_fitness(pool):
    """Define the fitness function as the sum of the rank of the mean test score and the rank of the
    overfit.

    The rank of the mean test score is calculated by ranking the mean test score in ascending order.
    The rank of the overfit is calculated by ranking the overfit in ascending order.
    The overfit is calculated as the difference between the mean train score and the mean test score.
    The fitness is the sum of the rank of the mean test score and the rank of the overfit.

    Parameters
    ----------
    pool : list of dict
        Each element in the list is a dictionary with the following keys:
        - features: list of int
            The features selected for this element.
        - mean_test_score: float
            The mean test score of the element.
        - mean_train_score: float
            The mean train score of the element.

    Returns
    -------
    fitness : list of float
        The fitness of each element in the pool.
    """

    pool_df = pd.DataFrame(pool)
    pool_df["rank_mean_test_score"] = pool_df["mean_test_score"].rank(ascending=False)
    pool_df["overfit"] = pool_df["mean_train_score"] - pool_df["mean_test_score"]
    pool_df["rank_overfit"] = pool_df["overfit"].rank(ascending=True)
    pool_df["rank_sum"] = pool_df["rank_mean_test_score"] + pool_df["rank_overfit"]

    pool_df["rank_sum_rank"] = pool_df["rank_sum"].rank(ascending=False)
    return pool_df["rank_sum_rank"].to_list()


def rank_mean_test_score_fitness(pool):
    """Define the fitness function as the rank of the mean test score.

    The rank of the mean test score is calculated by ranking the mean test score in ascending order.

    Parameters
    ----------

    pool : list of dict
        Each element in the list is a dictionary with the following keys:
        - features: list of int
            The features selected for this element.
        - mean_test_score: float
            The mean test score of the element.
        - mean_train_score: float
            The mean train score of the element.

    Returns
    -------
    fitness : list of float
        The fitness of each element in the pool.
    """
    pool_df = pd.DataFrame(pool)
    pool_df["rank_mean_test_score"] = pool_df["mean_test_score"].rank(ascending=True)
    return pool_df["rank_mean_test_score"].to_list()


class HybridImportanceGACVFeatureSelector(
    SelectorMixin, MetaEstimatorMixin, BaseEstimator
):
    """Feature selection using Hybrid Genetic Algorithm-Importance with Cross-Validation.

    This feature selector uses a genetic algorithm to select features. The genetic algorithm
    is hybridized with feature importance. The feature importance is calculated using a
    cross-validation scheme. The algorithm works as follows:

    **Pool initialization:** The pool is initialized with random features. The number of features is
    randomly generated using a normal distribution with the average number of features to select and
    the standard deviation of the number of features to select as parameters. The number of features
    is clipped to be between the minimum number of features to select and the number of features in
    the dataset.

    **Cross Over:** The cross over is done by combining the features of the parents. The features
    are sorted by importance and the children are created by combining the features of the parents
    in a round-robin fashion. The number of features of the children is the average of the number of
    features of the parents. In this way, the children will have the most important features of the
    parents.

    **Mutation:** The mutation is done by randomly changing the number of features and replacing the
    least important features with random features.

    **Selection:** The selection is done by selecting the top `pool_size` solutions based on the
    fitness function.


    Parameters
    ----------
    estimator : object
        An estimator that follows the scikit-learn API and has a `fit` method.
    cv : int, cross-validation generator or an iterable, default=5
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
        - None, to use the default 5-fold cross-validation,
        - int, to specify the number of folds in a (Stratified)KFold,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.
    scoring : str, callable or None, default=None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.
    random_state : int or None, default=None
        Controls the random seed given at the beginning of the algorithm.
    n_jobs : int or None, default=None
        The number of jobs to run in parallel. None means 1 unless in a
        joblib.parallel_backend context. -1 means using all processors.
    importance_getter : str or callable, default='auto'
        If 'auto', uses the feature importance either through a `coef_`
        or `feature_importances_` attributes of estimator.

        Also accepts a string that specifies an attribute name/path
        for extracting feature importance.
        For example, give `regressor_.coef_` in case of
        `~sklearn.compose.TransformedTargetRegressor`  or
        `named_steps.clf.feature_importances_` in case of
        `~sklearn.pipeline.Pipeline` with its last step named `clf`.

        If `callable`, overrides the default feature importance getter.
        The callable is passed with the fitted estimator and the validation set
        (X_val, y_val, estimator) and it should return importance for each feature.
    min_features_to_select : int or float, default=1
        The minimum number of features to select. If float, it represents the
        fraction of features to select.
    init_avg_features_num : float, default=15
        The average number of features to select in the initial pool of solutions.
    init_std_features_num : float, default=5
        The standard deviation of the number of features to select in the initial pool of solutions.
    pool_size : int, default=20
        The number of solutions in the pool.
    n_children_cross_over : int, default=5
        The number of children to create by cross-over.
    is_parent_selection_chance_proportional_to_fitness : bool, default=True
        If True, the probability of selecting a parent is proportional to its fitness. This means
        that the fittest parents are more likely to be selected during crossover.
    n_parents_cross_over : int, default=2
        The number of parents to select in each crossover. More than 2 parents can be selected during
        crossover. In that case, the top features of each parent are combined in a round-robin
        fashion to create a children. The number of features of the children is the average of the
        number of features of the parents.
    n_mutations : int, default=5
        The number of mutations to apply to the pool.
    range_change_n_features_mutation : tuple, default=(-2, 3)
        The range of the number of features to change during mutation. The first element is the
        minimum number of features to change and the second element is the maximum number of features
        to change. The right limit is exclusive.
    range_randomly_swapped_features_mutation : tuple, default=(1, 4)
        The range of the number of features to replace during mutation. The first element is the
        minimum number of features to replace and the second element is the maximum number of
        features to replace. The right limit is exclusive.
    mutation_candidate_scorer : callable or None, default=None
        Optional scoring function used to rank candidate features (those not already in the mutated
        element) when selecting replacements during mutation. Signature:
        ``scorer(X, y, selected_features) -> array-like of shape (n_features,)``, where higher
        scores indicate more desirable candidates and ``selected_features`` is the list of features
        currently in the pool element being mutated (same type as the feature identifiers used
        throughout — integer column indices for arrays, column names for DataFrames). When ``None``,
        replacement features are chosen uniformly at random (original behaviour). The scorer is
        called once per mutation with the full training data and the element's current feature set.
        :class:`~felimination.mrmr.MRMRRanker` can be passed directly — it
        auto-initialises on the first call and caches per-feature redundance
        vectors across subsequent calls within the same ``fit``.
    mutation_candidate_selection : {'best', 'sample'}, default='sample'
        How to pick replacement features from the scored candidates. Only used when
        ``mutation_candidate_scorer`` is not ``None``:

        - ``'best'``: deterministically select the highest-scored candidates.
        - ``'sample'``: sample without replacement with probability proportional to the score,
          preserving diversity while favouring high-scoring features.
    max_generations : int, default=100
        The maximum number of generations to run the genetic algorithm.
    patience : int, default=5
        The number of generations without improvement to wait before stopping the algorithm.
    callbacks : list of callable, default=None
        A list of callables that are called after each generation. Each callable should accept
        the selector and the pool as arguments.
    fitness_function : str or callable, default="mean_test_score"
        The fitness function to use. Possible string values are: `'mean_test_score'`,
        `'mean_train_score'`, If a callable is passed, it should accept a list of dictionaries where
        each dictionary has the following keys 'features', 'mean_test_score', 'mean_train_score' and
        return a list of floats with the fitness of each element in the pool. Defaults to
        [rank_mean_test_score_overfit_fitness](./#felimination.ga.rank_mean_test_score_fitness)

    Attributes
    ----------

    estimator_ : object
        The fitted estimator.

    support_ : array of shape (n_features,)
        The mask of selected features.

    best_solution_ : dict
        The best solution found by the genetic algorithm. It is a dictionary with the following keys
        - features: list of int
            The features selected for this element.
        - mean_test_score: float
            The mean test score of the element.
        - mean_train_score: float
            The mean train score of the element.
        - train_scores_per_fold: list of float
            The train score of each fold.
        - test_scores_per_fold: list of float
            The test score of each fold.
        - cv_importances: list of array
            The importances of each fold.
        - mean_cv_importances: array
            The mean importances of each fold.

    best_solutions_ : list of dict
        The best solutions found by the genetic algorithm at each generation. Each element is
        defined as in `best_solution_`.

    evaluation_cache_ : dict
        Cache mapping ``frozenset(features)`` to evaluation results (scores and importances).
        Populated during ``fit``; allows ``_evaluate_calculate_importances`` to skip CV for
        feature sets that have already been evaluated in the same ``fit`` call.

    Examples
    --------

    >>> from felimination.ga import HybridImportanceGACVFeatureSelector
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.linear_model import LogisticRegression
    >>> X, y = make_classification(
        n_samples=sample_size,
        n_features=2,
        n_classes=2,
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=random_state,
    )
    >>> estimator = LogisticRegression(random_state=42)
    >>> selector = selector = HybridImportanceGACVFeatureSelector(
        random_state=random_state,
        init_avg_features_num=2,
        init_std_features_num=1,
    )
    >>> selector = selector.fit(X, y)
    >>> selector.support_
    array([ True,  True,  True,  True,  True, False, False, False, False,
           False])
    """

    _parameter_constraints: dict = {
        "estimator": [HasMethods(["fit"])],
        "min_features_to_select": [
            None,
            Interval(RealNotInt, 0, 1, closed="right"),
            Interval(Integral, 0, None, closed="neither"),
        ],
        "init_avg_features_num": [
            Interval(Real, 0, None, closed="neither"),
        ],
        "init_std_features_num": [
            Interval(Real, 0, None, closed="neither"),
        ],
        "pool_size": [Interval(Integral, 1, None, closed="neither")],
        "n_children_cross_over": [Interval(Integral, 0, None, closed="neither")],
        "is_parent_selection_chance_proportional_to_fitness": [bool],
        "n_parents_cross_over": [Interval(Integral, 2, None, closed="left")],
        "n_mutations": [Interval(Integral, 0, None, closed="neither")],
        "max_generations": [Interval(Integral, 1, None, closed="neither")],
        "patience": [Interval(Integral, 1, None, closed="neither")],
        "importance_getter": [str, callable],
    }

    def __init__(
        self,
        estimator: BaseEstimator | LogisticRegression,
        *,
        cv=5,
        scoring=None,
        random_state=None,
        n_jobs=None,
        importance_getter="auto",
        min_features_to_select=1,
        init_avg_features_num=15,
        init_std_features_num=5,
        pool_size=20,
        is_parent_selection_chance_proportional_to_fitness=True,
        n_children_cross_over=5,
        n_parents_cross_over=2,
        n_mutations=5,
        range_change_n_features_mutation=(-2, 3),
        range_randomly_swapped_features_mutation=(1, 4),
        max_generations=100,
        patience=5,
        callbacks=None,
        fitness_function="mean_test_score",
        mutation_candidate_scorer=None,
        mutation_candidate_selection="sample",
    ) -> None:
        self.estimator = estimator
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.importance_getter = importance_getter
        self.min_features_to_select = min_features_to_select
        self.init_avg_features_num = init_avg_features_num
        self.init_std_features_num = init_std_features_num
        self.pool_size = pool_size
        self.n_children_cross_over = n_children_cross_over
        self.is_parent_selection_chance_proportional_to_fitness = (
            is_parent_selection_chance_proportional_to_fitness
        )
        self.n_parents_cross_over = n_parents_cross_over
        self.n_mutations = n_mutations
        self.range_change_n_features_mutation = range_change_n_features_mutation
        self.range_randomly_swapped_features_mutation = (
            range_randomly_swapped_features_mutation
        )
        self.max_generations = max_generations
        self.patience = patience
        self.callbacks = callbacks
        self.fitness_function = fitness_function
        self.mutation_candidate_scorer = mutation_candidate_scorer
        self.mutation_candidate_selection = mutation_candidate_selection

    @property
    def best_solution_(self):
        check_is_fitted(self)
        return self.best_solutions_[-1]

    def _evaluate_calculate_importances(self, pool, X, y, cv, scorer, routed_params):
        if effective_n_jobs(self.n_jobs) == 1:
            parallel, func = list, _train_score_get_importance
        else:
            parallel = Parallel(n_jobs=self.n_jobs)
            func = delayed(_train_score_get_importance)

        splits = list(cv.split(X, y, **routed_params.splitter.split))
        n_splits = len(splits)

        # Deduplicate by frozenset so each unique feature set appears exactly
        # once in uncached. Duplicates in pool would break the
        # [i*n_splits:(i+1)*n_splits] result-slicing below.
        seen_fs: set = set()
        uncached = []
        for el in pool:
            fs = frozenset(el["features"])
            if fs not in self.evaluation_cache_ and fs not in seen_fs:
                uncached.append(el)
                seen_fs.add(fs)

        if uncached:
            scores_importances_1d_array = parallel(
                func(
                    self.estimator,
                    _select_X_with_features(X, element["features"]),
                    y,
                    train,
                    test,
                    scorer,
                    self.importance_getter,
                    routed_params=routed_params,
                )
                for element in uncached
                for train, test in splits
            )
            for i, element in enumerate(uncached):
                fold_results = scores_importances_1d_array[
                    i * n_splits : (i + 1) * n_splits
                ]
                train_scores = [r[0] for r in fold_results]
                test_scores = [r[1] for r in fold_results]
                # Store per-feature importances as a dict so retrieval is
                # order-independent when the same feature set is reused.
                fold_importances = [
                    dict(zip(element["features"], np.vstack([r[2]]).mean(axis=0)))
                    for r in fold_results
                ]
                self.evaluation_cache_[frozenset(element["features"])] = {
                    "train_scores_per_fold": train_scores,
                    "test_scores_per_fold": test_scores,
                    "fold_importances": fold_importances,
                }

        for element in pool:
            cached = self.evaluation_cache_[frozenset(element["features"])]
            element["train_scores_per_fold"] = cached["train_scores_per_fold"]
            element["test_scores_per_fold"] = cached["test_scores_per_fold"]
            element["cv_importances"] = [
                np.array([fold_imp[f] for f in element["features"]])
                for fold_imp in cached["fold_importances"]
            ]
            element["mean_train_score"] = np.mean(cached["train_scores_per_fold"])
            element["mean_test_score"] = np.mean(cached["test_scores_per_fold"])
            element["mean_cv_importances"] = np.mean(
                np.vstack(element["cv_importances"]), axis=0
            )
        return pool

    def _calculate_fitness(self, pool):
        if isinstance(self.fitness_function, str):
            fitness = [el[self.fitness_function] for el in pool]
        else:
            fitness = self.fitness_function(pool)

        return fitness

    def _combine_parents(self, parents):

        sorted_features = [
            np.array(parent["features"])[
                np.argsort(-np.array(parent["mean_cv_importances"]))
            ]
            for parent in parents
        ]
        combined_features = _deduplicate(_roundrobin(*sorted_features))
        n_features = int(
            round(np.mean([len(parent["features"]) for parent in parents]))
        )

        return {
            "features": combined_features[:n_features],
        }

    def _cross_over(self, pool):
        fitness = self._calculate_fitness(pool)
        children = []
        if self.is_parent_selection_chance_proportional_to_fitness:
            parent_selection_proba = fitness / np.sum(fitness)
        else:
            parent_selection_proba = None
        for _ in range(self.n_children_cross_over):
            parents = np.random.choice(
                pool,
                size=self.n_parents_cross_over,
                replace=False,
                p=parent_selection_proba,
            )
            children.append(self._combine_parents(parents))

        return children

    def _mutate(self, pool, all_features, X, y):
        mutated_pool = []
        for _ in range(self.n_mutations):
            element = np.random.choice(pool)

            number_of_features = max(
                len(element["features"])
                + np.random.randint(*self.range_change_n_features_mutation),
                self.min_features_to_select,
            )

            n_of_features_to_replace = np.random.randint(
                *self.range_randomly_swapped_features_mutation
            )
            if n_of_features_to_replace <= 0:
                continue

            candidates = [
                feat for feat in all_features if feat not in element["features"]
            ]

            # Preserve original feature type (list(np.array(feats)[argsort]) wraps
            # ints as np.int64 which breaks dict lookups downstream).
            orig_feats = element["features"]
            sort_order = np.argsort(-np.array(element["mean_cv_importances"]))
            sorted_features = [orig_feats[i] for i in sort_order]

            if self.mutation_candidate_scorer is not None and len(candidates) > 0:
                all_scores = np.asarray(
                    self.mutation_candidate_scorer(X, y, element["features"]),
                    dtype=float,
                )
                feature_to_score = {
                    feat: all_scores[i] for i, feat in enumerate(all_features)
                }
                candidate_scores = np.array([feature_to_score[c] for c in candidates])
                if self.mutation_candidate_selection == "best":
                    # -inf scores sort naturally to the end via argsort
                    ordered_candidates = [
                        candidates[i] for i in np.argsort(-candidate_scores)
                    ]
                else:  # 'sample'
                    # Separate finite-scored from -inf candidates so probability
                    # normalisation stays well-defined (MRMRRanker emits -inf for
                    # low-relevance features).
                    finite_mask = np.isfinite(candidate_scores)
                    eligible = [c for c, m in zip(candidates, finite_mask) if m]
                    ineligible = [c for c, m in zip(candidates, finite_mask) if not m]
                    if eligible:
                        eligible_scores = candidate_scores[finite_mask]
                        # Shift so min→0, then add eps so every eligible candidate
                        # has nonzero probability even when all scores are equal.
                        shifted = eligible_scores - eligible_scores.min() + 1e-10
                        probs = shifted / shifted.sum()
                        sampled = np.random.choice(
                            len(eligible), size=len(eligible), replace=False, p=probs
                        )
                        ordered_eligible = [eligible[i] for i in sampled]
                    else:
                        ordered_eligible = []
                    np.random.shuffle(ineligible)
                    ordered_candidates = ordered_eligible + ineligible
            else:
                ordered_candidates = candidates[:]
                np.random.shuffle(ordered_candidates)

            mutated_features = (
                sorted_features[:-n_of_features_to_replace] + ordered_candidates
            )[:number_of_features]

            mutated_pool.append({"features": mutated_features})
        return mutated_pool

    def _get_support_mask(self):
        check_is_fitted(self)
        return self.support_

    @_fit_context(
        # estimator not validated yet
        prefer_skip_nested_validation=False
    )
    def fit(self, X, y, groups=None, **params):
        """Fit the selector and then the underlying estimator on the selected features.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values.
        **params : dict
            Additional parameters passed to the `fit` method of the underlying
            estimator.

        Returns
        -------
        self : object
            Fitted estimator.
        """
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
            accept_sparse="csc",
            ensure_min_features=2,
            ensure_all_finite=not get_tags(self.estimator).input_tags.allow_nan,
            multi_output=True,
            dtype=None,
        )

        # Initialization
        cv = check_cv(self.cv, y, classifier=is_classifier(self.estimator))
        scorer = check_scoring(self.estimator, scoring=self.scoring)
        n_features = X.shape[1]
        if self.min_features_to_select is None:
            min_features_to_select = n_features // 2
        elif isinstance(self.min_features_to_select, Integral):  # int
            min_features_to_select = self.min_features_to_select
        else:  # float
            min_features_to_select = int(n_features * self.min_features_to_select)

        if isinstance(X, pd.DataFrame):
            all_features = X.columns.to_list()
        else:
            all_features = list(range(n_features))

        np.random.seed(self.random_state)
        self.evaluation_cache_ = {}

        # Create the initial pool of solutions
        pool = [
            {
                "features": list(
                    np.random.choice(
                        all_features,
                        min(
                            max(
                                int(
                                    np.random.normal(
                                        self.init_avg_features_num,
                                        self.init_std_features_num,
                                    )
                                ),
                                min_features_to_select,
                            ),
                            n_features,
                        ),
                        replace=False,
                    )
                ),
            }
            for _ in range(self.pool_size)
        ]

        # Evaluate the initial pool of solutions
        pool = self._evaluate_calculate_importances(
            pool, X, y, cv, scorer, routed_params
        )
        self.best_solutions_ = []
        for _ in range(1, self.max_generations):
            children = self._cross_over(pool)
            children = self._evaluate_calculate_importances(
                children, X, y, cv, scorer, routed_params
            )
            pool.extend(children)
            mutations = self._mutate(pool, all_features, X, y)
            mutations = self._evaluate_calculate_importances(
                mutations, X, y, cv, scorer, routed_params
            )
            pool.extend(mutations)
            pool_sorted = [
                element
                for _, element in sorted(
                    zip(self._calculate_fitness(pool), pool),
                    reverse=True,
                    key=itemgetter(0),
                )
            ]
            pool = pool_sorted[: self.pool_size]
            self.best_solutions_.append(pool[0])

            if self.callbacks:
                for callback in self.callbacks:
                    callback(self, pool)

            if len(self.best_solutions_) > self.patience:
                if all(
                    [
                        self.best_solutions_[-1]["features"] == solution["features"]
                        for solution in self.best_solutions_[-self.patience :]
                    ]
                ):
                    break

        self.estimator_ = clone(self.estimator)
        X_remaining_features = _select_X_with_features(
            X, self.best_solution_["features"]
        )
        self.estimator_.fit(X_remaining_features, y, **routed_params.estimator.fit)
        self.support_ = np.array(
            [
                True if feature in self.best_solution_["features"] else False
                for feature in all_features
            ]
        )

        return self

    def plot(self, **kwargs):
        """Plot the mean test score and mean train score of the best solution at each generation.

        Parameters
        ----------
        **kwargs : dict
            Additional parameters passed to seaborn.lineplot. For a list
            of possible options, please visit
            [seaborn.lineplot](https://seaborn.pydata.org/generated/seaborn.lineplot.html)  # noqa

        Returns
        -------
        matplotlib.axes.Axes
            The axis where the plot has been plotted.
        """
        data_points_to_plot_long_form = []
        for generation, best_solution in enumerate(self.best_solutions_, start=1):
            for set, scores in zip(
                ["validation", "train"],
                [
                    best_solution["test_scores_per_fold"],
                    best_solution["train_scores_per_fold"],
                ],
            ):
                for score in scores:
                    data_points_to_plot_long_form.append(
                        {"generation": generation, "score": score, "set": set}
                    )
        df_plot = pd.DataFrame(data_points_to_plot_long_form)
        lineplot_kwargs = dict(
            x="generation",
            y="score",
            hue="set",
            markers=True,
            style="set",
            hue_order=["validation", "train"],
            style_order=["validation", "train"],
            seed=self.random_state,
        )
        lineplot_kwargs.update(**kwargs)
        return sns.lineplot(data=df_plot, **lineplot_kwargs)

    @available_if(_estimator_has("predict"))
    def predict(self, X):
        """Reduce X to the selected features and predict using the estimator.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of shape [n_samples]
            The predicted target values.
        """
        check_is_fitted(self)
        return self.estimator_.predict(self.transform(X))

    @available_if(_estimator_has("score"))
    def score(self, X, y, **fit_params):
        """Reduce X to the selected features and return the score of the estimator.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The input samples.

        y : array of shape [n_samples]
            The target values.

        **fit_params : dict
            Parameters to pass to the `score` method of the underlying
            estimator.

            .. versionadded:: 1.0

        Returns
        -------
        score : float
            Score of the underlying base estimator computed with the selected
            features returned by `rfe.transform(X)` and `y`.
        """
        check_is_fitted(self)
        return self.estimator_.score(self.transform(X), y, **fit_params)

    @available_if(_estimator_has("decision_function"))
    def decision_function(self, X):
        """Compute the decision function of ``X``.

        Parameters
        ----------
        X : {array-like or sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        score : array, shape = [n_samples, n_classes] or [n_samples]
            The decision function of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
            Regression and binary classification produce an array of shape
            [n_samples].
        """
        check_is_fitted(self)
        return self.estimator_.decision_function(self.transform(X))

    @available_if(_estimator_has("predict_proba"))
    def predict_proba(self, X):
        """Predict class probabilities for X.

        Parameters
        ----------
        X : {array-like or sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        p : array of shape (n_samples, n_classes)
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
        """
        check_is_fitted(self)
        return self.estimator_.predict_proba(self.transform(X))

    @available_if(_estimator_has("predict_log_proba"))
    def predict_log_proba(self, X):
        """Predict class log-probabilities for X.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The input samples.

        Returns
        -------
        p : array of shape (n_samples, n_classes)
            The class log-probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
        """
        check_is_fitted(self)
        return self.estimator_.predict_log_proba(self.transform(X))

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        sub_estimator_tags = get_tags(self.estimator)
        tags.target_tags.required = False
        tags.input_tags.allow_nan = sub_estimator_tags.input_tags.allow_nan
        return tags

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        sub_estimator_tags = get_tags(self.estimator)
        tags.target_tags.required = False
        tags.input_tags.allow_nan = sub_estimator_tags.input_tags.allow_nan
        return tags
