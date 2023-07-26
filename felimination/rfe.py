"""Module with tools to perform feature selection.

This module contains the following classes:

- `FeliminationRFECV`: base class for feature selection.
- `PermutationImportanceRFECV`: recursive feature elimination with
    cross-validation based on permutation importance.
"""
from collections import defaultdict
from inspect import signature
from numbers import Integral
from operator import attrgetter

import numpy as np
import pandas as pd
import seaborn as sns
from joblib import effective_n_jobs
from sklearn.base import BaseEstimator, clone, is_classifier
from sklearn.feature_selection import RFE
from sklearn.linear_model._logistic import LogisticRegression
from sklearn.metrics import check_scoring
from sklearn.model_selection import check_cv, cross_validate
from sklearn.model_selection._validation import _score
from sklearn.utils import safe_sqr
from sklearn.utils.metaestimators import _safe_split
from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils.validation import check_is_fitted

from felimination.importance import PermutationImportance


def _train_score_get_importance(
    estimator, X, y, train, test, scorer, importance_getter
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


class FeliminationRFECV(RFE):
    """Perform recursive feature elimination with cross-validation
    following scikit-learn standards.

    It has the following differences with RFECV from scikit-learn:

    - It supports an `importance_getter` function that also uses a validation
    set to compute the feature importances. This allows to use importance measures
    like permutation importance or shap.
    - Instead of using Cross Validation to select the number of features, it
    uses cross validation to get a more accurate estimate of the feature
    importances. This means that the number of features to select has to be
    set during initialization, similarly to RFE.
    - When `step` is a float value it is removes a percentage of the number
    of **remaining** features, not total like in RFE/RFECV. This allows to
    drop big chunks of feature at the beginning of the RFE process and to slow
    down towards the end of the process.
    - Has a plotting function
    - Adds information about the number of features selected at each step in the
    attribute `cv_results_`
    - Allows to change the number of features to be selected after fitting.

    Rater than that, it is a copy-paste of RFE, so credit goes to scikit-learn.

    The algorithm of feature selection goes as follows:
    ```
    while n_features > n_features_to_select:
        - The estimator is trained on the selected features and the score is
          computed using cross validation.
        - feature importance is computed for each validation fold on the validation
          set and then averaged.
        - The least important features are pruned.
        - The pruned features are removed from the dataset.
    ```

    Parameters
    ----------
    estimator : ``Estimator`` instance
        A supervised learning estimator with a ``fit`` method.
    step : int or float, default=1
        If greater than or equal to 1, then ``step`` corresponds to the
        (integer) number of features to remove at each iteration.
        If within (0.0, 1.0), then ``step`` corresponds to the percentage
        (rounded down) of **remaining** features to remove at each iteration.
        Note that the last iteration may remove fewer than ``step`` features in
        order to reach ``min_features_to_select``.
    n_features_to_select : int or float, default=None
        The number of features to select. If `None`, half of the features are
        selected. If integer, the parameter is the absolute number of features
        to select. If float between 0 and 1, it is the fraction of the features to
        select.
    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

            - None, to use the default 5-fold cross-validation,
            - integer, to specify the number of folds.
            - :term:`CV splitter`,
            - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`~sklearn.model_selection.StratifiedKFold` is used. If the
        estimator is a classifier or if ``y`` is neither binary nor multiclass,
        :class:`~sklearn.model_selection.KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.
    scoring : str, callable or None, default=None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.
    verbose : int, default=0
        Controls verbosity of output.
    n_jobs : int or None, default=None
        Number of cores to run in parallel while fitting across folds.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.
    importance_getter : str or callable, default='auto'
        If 'auto', uses the feature importance either through a `coef_`
        or `feature_importances_` attributes of estimator.

        Also accepts a string that specifies an attribute name/path
        for extracting feature importance.
        For example, give `regressor_.coef_` in case of
        :class:`~sklearn.compose.TransformedTargetRegressor`  or
        `named_steps.clf.feature_importances_` in case of
        :class:`~sklearn.pipeline.Pipeline` with its last step named `clf`.

        If `callable`, overrides the default feature importance getter.
        The callable is passed with the fitted estimator and the validation set
        (X_val, y_val, estimator) and it should return importance for each feature.


    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        The classes labels. Only available when `estimator` is a classifier.
    estimator_ : ``Estimator`` instance
        The fitted estimator used to select features.
    cv_results_ : dict of ndarrays
        A dict with keys:
        n_features : ndarray of shape (n_subsets_of_features,)
            The number of features used at that step.
        split(k)_test_score : ndarray of shape (n_subsets_of_features,)
            The cross-validation scores across (k)th fold.
        mean_test_score : ndarray of shape (n_subsets_of_features,)
            Mean of scores over the folds.
        std_test_score : ndarray of shape (n_subsets_of_features,)
            Standard deviation of scores over the folds.
        split(k)_train_score : ndarray of shape (n_subsets_of_features,)
            The cross-validation scores across (k)th fold.
        mean_train_score : ndarray of shape (n_subsets_of_features,)
            Mean of scores over the folds.
        std_train_score : ndarray of shape (n_subsets_of_features,)
            Standard deviation of scores over the folds.
    n_features_ : int
        The number of selected features.
    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if the
        underlying estimator exposes such an attribute when fit.
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.
    ranking_ : ndarray of shape (n_features,)
        The feature ranking, such that ``ranking_[i]`` corresponds to the
        ranking position of the i-th feature. Selected (i.e., estimated
        best) features are assigned rank 1.
    support_ : ndarray of shape (n_features,)
        The mask of selected features.

    Examples
    --------
    The following example shows how to retrieve the 5 most informative
    features in the Friedman #1 dataset.

    >>> from felimination.rfe import FeliminationRFECV
    >>> from felimination.importance import PermutationImportance
    >>> from sklearn.datasets import make_friedman1
    >>> from sklearn.svm import SVR
    >>> X, y = make_friedman1(n_samples=50, n_features=10, random_state=0)
    >>> estimator = SVR(kernel="linear")
    >>> selector = selector = FeliminationRFECV(
        estimator,
        step=1,
        cv=5,
        n_features_to_select=5,
        importance_getter=PermutationImportance()
    )
    >>> selector = selector.fit(X, y)
    >>> selector.support_
    array([ True,  True,  True,  True,  True, False, False, False, False,
           False])
    >>> selector.ranking_
    array([1, 1, 1, 1, 1, 6, 3, 4, 2, 5])
    """

    def __init__(
        self,
        estimator: BaseEstimator | LogisticRegression,
        *,
        step=1,
        n_features_to_select=1,
        cv=None,
        scoring=None,
        random_state=None,
        verbose=0,
        n_jobs=None,
        importance_getter="auto",
    ) -> None:
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.random_state = random_state
        super().__init__(
            estimator,
            n_features_to_select=n_features_to_select,
            step=step,
            verbose=verbose,
            importance_getter=importance_getter,
        )

    def fit(self, X, y, groups=None, **fit_params):
        """Fit the RFE model and then the underlying estimator on the selected features.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values.
        **fit_params : dict
            Additional parameters passed to the `fit` method of the underlying
            estimator.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        self._validate_params()
        tags = self._get_tags()
        X, y = self._validate_data(
            X,
            y,
            accept_sparse="csc",
            ensure_min_features=2,
            force_all_finite=not tags.get("allow_nan", True),
            multi_output=True,
            dtype=None,
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
            # Select remaining features
            features = np.arange(n_features)[support_]
            X_remaining_features = X[:, features]

            if self.verbose > 0:
                print(
                    "Fitting estimator with %d features." % current_number_of_features
                )

            # Train model, score it and get importances
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

            # for sparse case ranks is matrix
            ranks = np.ravel(ranks)

            if 0.0 < self.step < 1.0:
                step = int(max(1, self.step * current_number_of_features))
            else:
                step = int(self.step)

            # Eliminate the worst features
            threshold = min(step, current_number_of_features - n_features_to_select)

            support_[features[ranks][:threshold]] = False
            ranking_[np.logical_not(support_)] += 1

            # Update cv scores
            for train_or_test, scores_per_fold in zip(
                ["train", "test"], [train_scores_per_fold, test_scores_per_fold]
            ):
                for i, score in enumerate(scores_per_fold):
                    self.cv_results_[f"split{i}_{train_or_test}_score"].append(score)
                self.cv_results_[f"mean_{train_or_test}_score"].append(
                    np.mean(scores_per_fold)
                )
                self.cv_results_[f"std_{train_or_test}_score"].append(
                    np.std(scores_per_fold)
                )
            self.cv_results_["n_features"].append(current_number_of_features)

            current_number_of_features = np.sum(support_)
        # Set final attributes

        # Estimate performances of final model
        features = np.arange(n_features)[support_]
        X_remaining_features = X[:, features]
        cv_scores = cross_validate(
            self.estimator,
            X_remaining_features,
            y,
            groups=groups,
            scoring=scorer,
            cv=cv,
            n_jobs=self.n_jobs,
            fit_params=fit_params,
            return_train_score=True,
        )
        self.cv_results_["n_features"].append(current_number_of_features)
        # Update cv scores
        for train_or_test in ["train", "test"]:
            scores_per_fold = cv_scores[f"{train_or_test}_score"]
            for i, score in enumerate(scores_per_fold):
                self.cv_results_[f"split{i}_{train_or_test}_score"].append(score)
            self.cv_results_[f"mean_{train_or_test}_score"].append(
                np.mean(scores_per_fold)
            )
            self.cv_results_[f"std_{train_or_test}_score"].append(
                np.std(scores_per_fold)
            )

        features = np.arange(n_features)[support_]
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X[:, features], y, **fit_params)

        self.n_features_ = support_.sum()
        self.support_ = support_
        self.ranking_ = ranking_
        self.cv_results_ = dict(self.cv_results_)
        return self

    def set_n_features_to_select(self, n_features_to_select):
        """Changes the number of features to select after fitting.

        The underlying estimator **will not be retrained**. So this method will not
        alter the behavior of predict/predict_proba but it will change the behavior
        of transform and get_feature_names_out.

        Parameters
        ----------
        n_features_to_select : int
            The number of features to select. Must be a value among
            `cv_results_["n_features"]`

        Returns
        -------
        self : object
            Fitted estimator.

        Raises
        ------
        ValueError
            When the number of features to select has not been tried during the
            feature selection procedure.
        """
        check_is_fitted(self)
        if n_features_to_select not in self.cv_results_["n_features"]:
            raise ValueError(
                f"This selector has not been fitted up with {n_features_to_select}, "
                f"please select a value in {set(self.cv_results_['n_features'])} or "
                "refit the selector changing the step parameter of the n_features_to_select"
            )
        support_ = np.zeros_like(self.support_, dtype=bool)
        support_[np.argsort(self.ranking_)[:n_features_to_select]] = True
        self.support_ = support_
        return self

    def plot(self, **kwargs):
        """Plot a feature selection plot with number of features

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
        check_is_fitted(self)
        df = pd.DataFrame(self.cv_results_)
        split_score_cols = [col for col in df if "split" in col]
        df_long_form = df[split_score_cols + ["n_features"]].melt(
            id_vars=["n_features"],
            value_vars=split_score_cols,
            var_name="split",
            value_name="score",
        )
        df_long_form["set"] = np.where(
            df_long_form["split"].str.contains("train"), "train", "validation"
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
        )
        lineplot_kwargs.update(**kwargs)
        ax = sns.lineplot(data=df_long_form, **lineplot_kwargs)
        ax.set_xticks(df.n_features)
        return ax


class PermutationImportanceRFECV(FeliminationRFECV):
    """Preset of FeliminationRFECV using permutation importance as importance getter.

    It has the following differences with RFECV from scikit-learn:

    - It supports an `importance_getter` function that also uses a validation
      set to compute the feature importances. This allows to use importance measures
      like permutation importance or shap.
    - Instead of using Cross Validation to select the number of features, it
      uses cross validation to get a more accurate estimate of the feature
      importances. This means that the number of features to select has to be
      set during initialization, similarly to RFE.
    - When `step` is a float value it is removes a percentage of the number
      of **remaining** features, not total like in RFE/RFECV. This allows to
      drop big chunks of feature at the beginning of the RFE process and to slow
      down towards the end of the process.
    - Has a plotting function
    - Adds information about the number of features selected at each step in the
      attribute `cv_results_`
    - Allows to change the number of features to be selected after fitting.

    Rater than that, it is a copy-paste of RFE, so credit goes to scikit-learn.

    The algorithm of feature selection goes as follows:
    ```
    while n_features > n_features_to_select:
        - The estimator is trained on the selected features and the score is
          computed using cross validation.
        - feature importance is computed for each validation fold on the validation
          set and then averaged.
        - The least important features are pruned.
        - The pruned features are removed from the dataset.
    ```

    Parameters
    ----------
    estimator : ``Estimator`` instance
        A supervised learning estimator with a ``fit`` method.
    step : int or float, default=1
        If greater than or equal to 1, then ``step`` corresponds to the
        (integer) number of features to remove at each iteration.
        If within (0.0, 1.0), then ``step`` corresponds to the percentage
        (rounded down) of **remaining** features to remove at each iteration.
        Note that the last iteration may remove fewer than ``step`` features in
        order to reach ``min_features_to_select``.
    n_features_to_select : int or float, default=None
        The number of features to select. If `None`, half of the features are
        selected. If integer, the parameter is the absolute number of features
        to select. If float between 0 and 1, it is the fraction of the features to
        select.
    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross-validation,
        - integer, to specify the number of folds.
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`~sklearn.model_selection.StratifiedKFold` is used. If the
        estimator is a classifier or if ``y`` is neither binary nor multiclass,
        :class:`~sklearn.model_selection.KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.
    scoring : str, callable or None, default=None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.
    verbose : int, default=0
        Controls verbosity of output.
    n_jobs : int or None, default=None
        Number of cores to run in parallel while fitting across folds.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.
    n_repeats : int, default=5
        Number of times to permute a feature.
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


    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        The classes labels. Only available when `estimator` is a classifier.
    estimator_ : ``Estimator`` instance
        The fitted estimator used to select features.
    cv_results_ : dict of ndarrays
        A dict with keys:
        n_features : ndarray of shape (n_subsets_of_features,)
            The number of features used at that step.
        split(k)_test_score : ndarray of shape (n_subsets_of_features,)
            The cross-validation scores across (k)th fold.
        mean_test_score : ndarray of shape (n_subsets_of_features,)
            Mean of scores over the folds.
        std_test_score : ndarray of shape (n_subsets_of_features,)
            Standard deviation of scores over the folds.
        split(k)_train_score : ndarray of shape (n_subsets_of_features,)
            The cross-validation scores across (k)th fold.
        mean_train_score : ndarray of shape (n_subsets_of_features,)
            Mean of scores over the folds.
        std_train_score : ndarray of shape (n_subsets_of_features,)
            Standard deviation of scores over the folds.
    n_features_ : int
        The number of selected features.
    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if the
        underlying estimator exposes such an attribute when fit.
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.
    ranking_ : ndarray of shape (n_features,)
        The feature ranking, such that ``ranking_[i]`` corresponds to the
        ranking position of the i-th feature. Selected (i.e., estimated
        best) features are assigned rank 1.
    support_ : ndarray of shape (n_features,)
        The mask of selected features.

    Examples
    --------
    The following example shows how to retrieve the 5 most informative
    features in the Friedman #1 dataset.

    >>> from felimination.rfe import PermutationImportanceRFECV
    >>> from sklearn.datasets import make_friedman1
    >>> from sklearn.svm import SVR
    >>> X, y = make_friedman1(n_samples=50, n_features=10, random_state=0)
    >>> estimator = SVR(kernel="linear")
    >>> selector = selector = PermutationImportanceRFECV(
            estimator,
            step=1,
            cv=5,
            n_features_to_select=5,
        )
    >>> selector = selector.fit(X, y)
    >>> selector.support_
    array([ True,  True,  True,  True,  True, False, False, False, False,
           False])
    >>> selector.ranking_
    array([1, 1, 1, 1, 1, 6, 3, 4, 2, 5])
    """

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
        self.sample_weight = sample_weight
        self.max_samples = max_samples
        super().__init__(
            estimator,
            step=step,
            n_features_to_select=n_features_to_select,
            cv=cv,
            random_state=random_state,
            scoring=scoring,
            verbose=verbose,
            n_jobs=n_jobs,
            importance_getter=PermutationImportance(
                scoring=scoring,
                n_repeats=n_repeats,
                # Better not to do double parallelization
                n_jobs=1,
                random_state=random_state,
                sample_weight=sample_weight,
                max_samples=max_samples,
            ),
        )
