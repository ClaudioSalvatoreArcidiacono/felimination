"""The idea behind this module comes from the conjunction of two concepts:

- [1] [Classifier Two-Sample Test](https://arxiv.org/abs/1610.06545)
- [2] [Recursive Feature Elimination](\
    https://scikit-learn.org/stable/modules/generated/\
    sklearn.feature_selection.RFE.html)

In [1] classifier performances are used to determine how similar two samples are. More
specifically, imagine to have two samples: `reference` and `test`. In order to assess
whether `reference` and `test` have been drawn from the same distribution, we could
train a classifier in classifying which instances belong to which sample. If the
model easily distinguishes instances from the two samples, then the two samples
have been probably drawn from two different distributions. Conversely, if the
classifier struggles to distinguish them, then it is likely that the samples have
been drawn from the same distribution.

In the context of drift detection, the classifier two-sample test can be used to
assess whether drift has happened between the reference and the test set and to
which degree.

The classes of this module take this idea one step further and attempt
to reduce the drift using recursive feature selection. After a classifier
is trained to distinguish between `reference` and `test`, the feature
importance of the classifier is used to determine which features contribute
the most in distinguishing between the two sets. The most important features
are then eliminated and the procedure is repeated until the classifier is not
able anymore to distinguish between the two samples, or until a certain amount
of features has been removed.

This module contains the following classes:
- `SampleSimilarityDriftRFE`: base class for drift-based sample similarity
    feature selection.
"""

from collections import defaultdict
from numbers import Integral

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, effective_n_jobs
from sklearn.base import ClassifierMixin, clone
from sklearn.metrics import check_scoring
from sklearn.model_selection import check_cv
from sklearn.utils._tags import _safe_tags, get_tags
from sklearn.utils.validation import validate_data

from felimination.importance import PermutationImportance
from felimination.rfe import FeliminationRFECV, _train_score_get_importance


class SampleSimilarityDriftRFE(FeliminationRFECV):
    """Recursively discards the features that introduce the highest drift.

    The algorithm of feature selection goes as follows:
    ```
    Split X into two sets using the `split_column`: X1 and X2
    create target array y1 for X1 as an array of zeroes
    create target array y2 for X2 as an array of ones
    vertically concatenate X1, X2 and y1 and y2, obtaining X_ss and y_ss
    Calculate Cross-validation performances of the estimator on X_ss and y_ss.
    while cross-validation-performances > max_score and n_features > min_features_to_select:
        Discard most important features
        Calculate Cross-validation performances of the estimator on X_ss and y_ss using the new feature set.
    ```

    Parameters
    ----------
    clf : ``Classifier`` instance
        A Classifier with a ``fit`` method.
    step : int or float, default=1
        If greater than or equal to 1, then ``step`` corresponds to the
        (integer) number of features to remove at each iteration.
        If within (0.0, 1.0), then ``step`` corresponds to the percentage
        (rounded down) of **remaining** features to remove at each iteration.
        Note that the last iteration may remove fewer than ``step`` features in
        order to reach ``min_features_to_select``.
    max_score : float, default=0.55
        Stops the feature selection procedure when the
        cross-validation score of the sample similarity classifier is
        lower than `max_score`.
    min_features_to_select : int or float, default=1
        The minimum number of features to select. If `None`, half of the features are
        selected. If integer, the parameter is the absolute number of features
        to select. If float between 0 and 1, it is the fraction of the features to
        select.
    split_column : str, default='split'
        The name of the column in the dataset that will be used to split the dataset
        into two sets.
    split_value : Any, default=None
        If defined, this value will be used to split the dataset into two sets.
    split_frac : float, default=0.5
        If split_value, split frac is used to determine a split_value. The split
        frac corresponds to the quantile of the split_column to use as the split_value.
    split_unique_values: bool, default=True
        Whether to calculate the quantile of the split_column to use as the split_value
        based on the unique values of the split_column.
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
        The classes labels.
    clf_ : ``Classifier`` instance
        The fitted classifier used to select features.
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

    # TODO: Add example
    """

    def __init__(
        self,
        clf: ClassifierMixin,
        *,
        step=1,
        max_score=0.55,
        min_features_to_select=1,
        split_col=0,
        split_value=None,
        split_frac=0.5,
        split_unique_values=True,
        cv=None,
        scoring=None,
        random_state=None,
        verbose=0,
        n_jobs=None,
        importance_getter="auto",
    ) -> None:
        self.max_score = max_score
        self.split_col = split_col
        self.split_value = split_value
        self.split_unique_values = split_unique_values
        self.split_frac = split_frac
        self.clf = clf
        super().__init__(
            estimator=clf,
            min_features_to_select=min_features_to_select,
            step=step,
            cv=cv,
            scoring=scoring,
            random_state=random_state,
            verbose=verbose,
            n_jobs=n_jobs,
            importance_getter=importance_getter,
        )

    def _build_sample_similarity_x_y(self, X, split_col_values):
        X1, X2 = self._split_X(X, split_col_values)
        y1 = np.zeros((X1.shape[0],))
        y2 = np.ones((X2.shape[0],))
        if isinstance(X, np.ndarray):
            X = np.vstack((X1, X2))
        else:
            X = pd.concat([X1, X2])
        y = np.concatenate((y1, y2))
        return X, y

    def _split_X(self, X, split_col_values):
        if self.split_value is not None:
            split_value = self.split_value
        else:
            if self.split_unique_values:
                unique_split_col_values = np.unique(split_col_values)
                split_value = np.quantile(unique_split_col_values, self.split_frac)
            split_value = np.quantile(split_col_values, self.split_frac)

        mask = split_col_values >= split_value

        if isinstance(X, np.ndarray):
            X1 = X[mask]
            X2 = X[~mask]
        else:
            X1 = X.loc[mask]
            X2 = X.loc[~mask]
        return X1, X2

    def fit(self, X, y=None, groups=None, **fit_params):
        """Fit the RFE model and then the underlying clf on the selected features.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values. Not used, kept for compatibility.
        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test set. Only used in conjunction with a "Group" :term:`cv`
            instance.
        **fit_params : dict
            Additional parameters passed to the `fit` method of the underlying
            clf.

        Returns
        -------
        self : object
            Fitted selector.
        """
        self._validate_params()
        validate_data(
            self,
            X,
            y,
            accept_sparse="csc",
            ensure_min_features=2,
            ensure_all_finite=not get_tags(self.estimator).input_tags.allow_nan,
            dtype=None,
        )

        if isinstance(self.split_col, str):
            split_col_idx = list(self.feature_names_in_).index(self.split_col)
        else:
            split_col_idx = self.split_col

        if isinstance(X, np.ndarray):
            split_col_values = X[:, split_col_idx]
        else:
            split_col_name = X.columns[split_col_idx]
            split_col_values = X[split_col_name].values

        X, y = self._build_sample_similarity_x_y(X, split_col_values=split_col_values)

        # Initialization
        cv = check_cv(self.cv, y, classifier=True)
        scorer = check_scoring(self.clf, scoring=self.scoring)
        n_features = X.shape[1]

        if self.min_features_to_select is None:
            min_features_to_select = n_features // 2
        elif isinstance(self.min_features_to_select, Integral):  # int
            min_features_to_select = self.min_features_to_select
        else:  # float
            min_features_to_select = int(n_features * self.min_features_to_select)

        support_ = np.ones(n_features, dtype=bool)
        support_[split_col_idx] = False
        ranking_ = np.ones(n_features, dtype=int)

        current_number_of_features = support_.sum()
        self.cv_results_ = defaultdict(list)

        if self.verbose > 0:
            print("Fitting clf with %d features." % current_number_of_features)

        # Train model, score it and get importances
        if effective_n_jobs(self.n_jobs) == 1:
            parallel, func = list, _train_score_get_importance
        else:
            parallel = Parallel(n_jobs=self.n_jobs)
            func = delayed(_train_score_get_importance)

        features = np.arange(n_features)[support_]

        X_remaining_features, features = self._select_X_with_remaining_features(
            X, support=support_, n_features=n_features
        )

        scores_importances = parallel(
            func(
                self.clf,
                X_remaining_features,
                y,
                train,
                test,
                scorer,
                self.importance_getter,
            )
            for train, test in cv.split(X_remaining_features, y, groups)
        )

        test_scores_per_fold = [
            score_importance[1] for score_importance in scores_importances
        ]
        train_scores_per_fold = [
            score_importance[0] for score_importance in scores_importances
        ]

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

        # Elimination
        while (
            np.mean(test_scores_per_fold) > self.max_score
            and current_number_of_features > min_features_to_select
        ):
            features = np.arange(n_features)[support_]
            if 0.0 < self.step < 1.0:
                step = int(max(1, self.step * current_number_of_features))
            else:
                step = int(self.step)
            # Eliminate most important features
            threshold = min(step, current_number_of_features - min_features_to_select)
            cv_importances = [
                score_importance[2] for score_importance in scores_importances
            ]
            mean_importances = np.mean(np.vstack(cv_importances), axis=0)
            ranks = np.argsort(-mean_importances)
            ranks = np.ravel(ranks)
            support_[features[ranks][:threshold]] = False
            ranking_[np.logical_not(support_)] += 1
            current_number_of_features = np.sum(support_)
            # Select remaining features
            features = np.arange(n_features)[support_]
            X_remaining_features, features = self._select_X_with_remaining_features(
                X, support=support_, n_features=n_features
            )

            if self.verbose > 0:
                print("Fitting clf with %d features." % current_number_of_features)

            # Train model, score it and get importances
            if effective_n_jobs(self.n_jobs) == 1:
                parallel, func = list, _train_score_get_importance
            else:
                parallel = Parallel(n_jobs=self.n_jobs)
                func = delayed(_train_score_get_importance)

            scores_importances = parallel(
                func(
                    self.clf,
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

        features = np.arange(n_features)[support_]
        self.clf_ = clone(self.clf)
        X_remaining_features, features = self._select_X_with_remaining_features(
            X, support=support_, n_features=n_features
        )
        self.clf_.fit(X_remaining_features, y, **fit_params)

        self.n_features_ = support_.sum()
        self.support_ = support_
        self.ranking_ = ranking_
        self.cv_results_ = dict(self.cv_results_)
        return self

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        sub_estimator_tags = get_tags(self.estimator)
        tags.target_tags.required = False
        tags.input_tags.allow_nan = sub_estimator_tags.input_tags.allow_nan
        return tags


class PermImpSampleSimilarityDriftRFE(SampleSimilarityDriftRFE):
    """Preset of SampleSimilarityDriftRFE using permutation importance as importance getter.

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
    clf : ``Classifier`` instance
        A Classifier with a ``fit`` method.
    step : int or float, default=1
        If greater than or equal to 1, then ``step`` corresponds to the
        (integer) number of features to remove at each iteration.
        If within (0.0, 1.0), then ``step`` corresponds to the percentage
        (rounded down) of **remaining** features to remove at each iteration.
        Note that the last iteration may remove fewer than ``step`` features in
        order to reach ``min_features_to_select``.
    max_score : float, default=0.55
        Stops the feature selection procedure when the
        cross-validation score of the sample similarity classifier is
        lower than `max_score`.
    min_features_to_select : int or float, default=1
        The minimum number of features to select. If `None`, half of the features are
        selected. If integer, the parameter is the absolute number of features
        to select. If float between 0 and 1, it is the fraction of the features to
        select.
    split_column : str, default='split'
        The name of the column in the dataset that will be used to split the dataset
        into two sets.
    split_value : Any, default=None
        If defined, this value will be used to split the dataset into two sets.
    split_frac : float, default=0.5
        If split_value, split frac is used to determine a split_value. The split
        frac corresponds to the quantile of the split_column to use as the split_value.
    split_unique_values: bool, default=True
        Whether to calculate the quantile of the split_column to use as the split_value
        based on the unique values of the split_column.
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
    """

    def __init__(
        self,
        clf: ClassifierMixin,
        *,
        step=1,
        max_score=0.55,
        min_features_to_select=1,
        split_col=0,
        split_value=None,
        split_frac=0.5,
        split_unique_values=True,
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
            clf=clf,
            max_score=max_score,
            min_features_to_select=min_features_to_select,
            split_col=split_col,
            split_value=split_value,
            split_frac=split_frac,
            split_unique_values=split_unique_values,
            step=step,
            cv=cv,
            scoring=scoring,
            random_state=random_state,
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
