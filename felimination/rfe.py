from collections import defaultdict
from inspect import signature
from numbers import Integral
from operator import attrgetter

import numpy as np
from joblib import effective_n_jobs
from sklearn.base import BaseEstimator, clone, is_classifier
from sklearn.feature_selection import RFE
from sklearn.linear_model._logistic import LogisticRegression
from sklearn.metrics import check_scoring
from sklearn.model_selection import check_cv
from sklearn.model_selection._validation import _score
from sklearn.utils import safe_sqr
from sklearn.utils.metaestimators import _safe_split
from sklearn.utils.parallel import Parallel, delayed

from felimination.importance import PermutationImportance


def _train_score_get_importance(
    estimator, X, y, train, test, scorer, importance_getter
):
    """
    Return the score for a fit across one fold.
    """
    estimator = clone(estimator)
    X_train, y_train = _safe_split(estimator, X, y, train)
    X_test, y_test = _safe_split(estimator, X, y, test, train)
    estimator = estimator.fit(X_train, y_train)
    train_score = _score(estimator, X_train, y_train, scorer)
    test_score = _score(estimator, X_test, y_test, scorer)
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

    X : {array-like, sparse matrix} of shape (n_samples, n_features), default=None
        The training input samples.

    y : array-like of shape (n_samples,), default=None
        The target values.

    getter : "auto", str or callable
        An attribute or a callable to get the feature importance. If `"auto"`,
        `estimator` is expected to expose `coef_` or `feature_importances`.

    transform_func : {"norm", "square"}, default=None
        The transform to apply to the feature importances. By default (`None`)
        no transformation is applied.

    norm_order : int, default=1
        The norm order to apply when `transform_func="norm"`. Only applied
        when `importances.ndim > 1`.

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


class FeliminationRFECV(RFE):
    def __init__(
        self,
        estimator: BaseEstimator | LogisticRegression,
        *,
        step=1,
        n_features_to_select=1,
        cv=None,
        scoring=None,
        verbose=0,
        n_jobs=None,
        importance_getter="auto",
    ) -> None:
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        super().__init__(
            estimator,
            n_features_to_select=n_features_to_select,
            step=step,
            verbose=verbose,
            importance_getter=importance_getter,
        )

    def fit(self, X, y, groups=None, **fit_params):
        # Parameter step_score controls the calculation of self.scores_
        # step_score is not exposed to users
        # and is used when implementing RFECV
        # self.scores_ will not be calculated when calling _fit through fit
        self._validate_params()
        tags = self._get_tags()
        X, y = self._validate_data(
            X,
            y,
            accept_sparse="csc",
            ensure_min_features=2,
            force_all_finite=not tags.get("allow_nan", True),
            multi_output=True,
        )

        # Initialization
        cv = check_cv(self.cv, y, classifier=is_classifier(self.estimator))
        scorer = check_scoring(self.estimator, scoring=self.scoring)
        n_features = X.shape[1]

        if self.n_features_to_select is None:
            n_features_to_select = n_features // 2
        elif isinstance(self.n_features_to_select, Integral):  # int
            n_features_to_select = self.n_features_to_select
        else:  # float
            n_features_to_select = int(n_features * self.n_features_to_select)

        support_ = np.ones(n_features, dtype=bool)
        ranking_ = np.ones(n_features, dtype=int)

        current_number_of_features = n_features
        self.cv_results_ = defaultdict(list)
        # Elimination
        while current_number_of_features > n_features_to_select:
            # Remaining features
            features = np.arange(n_features)[support_]
            X_remaining_features = X[:, features]

            if self.verbose > 0:
                print(
                    "Fitting estimator with %d features." % current_number_of_features
                )

            if effective_n_jobs(self.n_jobs) == 1:
                parallel, func = list, _train_score_get_importance
            else:
                parallel = Parallel(n_jobs=self.n_jobs)
                func = delayed(_train_score_get_importance)

            scores_importances = parallel(
                func(
                    self.estimator,
                    X_remaining_features,
                    y,
                    train,
                    test,
                    scorer,
                    self.importance_getter,
                )
                for train, test in cv.split(X_remaining_features, y, groups)
            )
            train_scores_per_fold = [
                score_importance[0] for score_importance in scores_importances
            ]
            test_scores_per_fold = [
                score_importance[1] for score_importance in scores_importances
            ]
            cv_importances = [
                score_importance[2] for score_importance in scores_importances
            ]
            mean_importances = np.mean(np.vstack(cv_importances), axis=0)
            ranks = np.argsort(mean_importances)

            # Update cv scores
            for train_or_test, scores_per_fold in zip(
                ["train", "test"], [train_scores_per_fold, test_scores_per_fold]
            ):
                for i, score in enumerate(test_scores_per_fold):
                    self.cv_results_[f"split{i}_{train_or_test}_score"].append(score)
                self.cv_results_[f"mean_{train_or_test}_score"].append(
                    np.mean(scores_per_fold)
                )
                self.cv_results_[f"std_{train_or_test}_score"].append(
                    np.std(scores_per_fold)
                )
            self.cv_results_["number_of_features"].append(current_number_of_features)

            # for sparse case ranks is matrix
            ranks = np.ravel(ranks)

            if 0.0 < self.step < 1.0:
                step = int(max(1, self.step * current_number_of_features))
            else:
                step = int(self.step)

            # Eliminate the worse features
            threshold = min(step, current_number_of_features - n_features_to_select)

            support_[features[ranks][:threshold]] = False
            ranking_[np.logical_not(support_)] += 1
            current_number_of_features = np.sum(support_)
        # Set final attributes
        features = np.arange(n_features)[support_]
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X[:, features], y, **fit_params)

        self.n_features_ = support_.sum()
        self.support_ = support_
        self.ranking_ = ranking_
        self.cv_results_ = dict(self.cv_results_)
        return self


class PermutationImportanceRFECV(FeliminationRFECV):
    def __init__(
        self,
        estimator: BaseEstimator | LogisticRegression,
        *,
        step=1,
        n_features_to_select=1,
        cv=None,
        scoring=None,
        verbose=0,
        n_jobs=None,
        n_repeats=5,
        random_state=None,
        sample_weight=None,
        max_samples=1.0,
    ) -> None:
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.sample_weight = sample_weight
        self.max_samples = max_samples
        super().__init__(
            estimator,
            step=step,
            n_features_to_select=n_features_to_select,
            cv=cv,
            scoring=scoring,
            verbose=verbose,
            n_jobs=n_jobs,
            importance_getter=PermutationImportance(
                scoring=scoring,
                n_repeats=n_repeats,
                n_jobs=n_jobs,
                random_state=random_state,
                sample_weight=sample_weight,
                max_samples=max_samples,
            ),
        )
