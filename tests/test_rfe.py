import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_classification, make_friedman1
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils._tags import get_tags

from felimination.rfe import PermutationImportanceRFECV


@pytest.fixture(scope="session")
def n_useful_features_classification():
    return 2


@pytest.fixture(scope="session")
def n_useful_features_regression():
    return 5


@pytest.fixture(scope="session")
def n_random_features():
    return 8


@pytest.fixture(scope="session")
def sample_size():
    # TODO: when the sample size changes, 1 random feature is selected in the
    # regression case, maybe good to take a look at it.
    return 1000


@pytest.fixture(scope="session")
def cv(random_state):
    return ShuffleSplit(random_state=random_state)


@pytest.fixture(scope="session")
def x_y_classification_with_rand_columns_arrays(
    n_useful_features_classification, n_random_features, sample_size, random_state
):
    np.random.seed(random_state)
    X, y = make_classification(
        n_samples=sample_size,
        n_features=n_useful_features_classification,
        n_classes=2,
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=random_state,
    )
    X_with_rand = np.hstack(
        (X, np.random.random(size=(sample_size, n_random_features)))
    )
    return X_with_rand, y


@pytest.fixture(scope="session")
def x_y_classification_with_rand_columns_pandas(
    x_y_classification_with_rand_columns_arrays,
    n_useful_features_classification,
    n_random_features,
):
    X_with_rand, y = x_y_classification_with_rand_columns_arrays
    column_names = [f"x{i+1}" for i in range(n_useful_features_classification)] + [
        f"rand{i+1}" for i in range(n_random_features)
    ]
    X_pandas = pd.DataFrame(X_with_rand, columns=column_names)
    y = pd.Series(y, name="target")
    return X_pandas, y


@pytest.fixture(scope="session")
def x_y_regression_with_rand_columns_arrays(
    n_useful_features_regression, n_random_features, sample_size, random_state
):
    X, y = make_friedman1(
        n_samples=sample_size,
        n_features=n_useful_features_regression,
        random_state=random_state,
    )
    X_with_rand = np.hstack(
        (X, np.random.random(size=(sample_size, n_random_features)))
    )
    return X_with_rand, y


@pytest.fixture(scope="session")
def x_y_regression_with_rand_columns_pandas(
    x_y_regression_with_rand_columns_arrays,
    n_useful_features_regression,
    n_random_features,
):
    X_with_rand, y = x_y_regression_with_rand_columns_arrays
    column_names = [f"x{i+1}" for i in range(n_useful_features_regression)] + [
        f"rand{i+1}" for i in range(n_random_features)
    ]
    X_pandas = pd.DataFrame(X_with_rand, columns=column_names)
    y = pd.Series(y, name="target")
    return X_pandas, y


def test_perm_imp_rfecv_classification_base_case_np_arrays(
    x_y_classification_with_rand_columns_arrays,
    n_useful_features_classification,
    n_random_features,
    cv,
    random_state,
):
    X_with_rand, y = x_y_classification_with_rand_columns_arrays
    selector = PermutationImportanceRFECV(
        LogisticRegression(random_state=random_state),
        cv=cv,
        min_features_to_select=n_useful_features_classification,
    )

    selector.fit(X_with_rand, y)
    assert (
        selector.ranking_[:n_useful_features_classification]
        == [1] * n_useful_features_classification
    ).all()
    assert (
        selector.support_
        == [True] * n_useful_features_classification + [False] * n_random_features
    ).all()


def test_perm_imp_rfecv_classification_base_case_pandas(
    x_y_classification_with_rand_columns_pandas,
    n_useful_features_classification,
    n_random_features,
    cv,
    random_state,
):
    X_with_rand, y = x_y_classification_with_rand_columns_pandas
    ct = ColumnTransformer(
        [
            ("scaler", StandardScaler(), ["x1"]),
        ],
        remainder="passthrough",
    )
    estimator = Pipeline([("ct", ct), ("lr", LogisticRegression(random_state=42))])
    selector = PermutationImportanceRFECV(
        estimator,
        cv=cv,
        min_features_to_select=n_useful_features_classification,
    )

    selector.fit(X_with_rand, y)
    assert (
        selector.ranking_[:n_useful_features_classification]
        == [1] * n_useful_features_classification
    ).all()
    assert (
        selector.support_
        == [True] * n_useful_features_classification + [False] * n_random_features
    ).all()


def test_perm_imp_rfecv_n_jobs(
    x_y_classification_with_rand_columns_pandas,
    n_useful_features_classification,
    n_random_features,
    cv,
    random_state,
):
    X_with_rand, y = x_y_classification_with_rand_columns_pandas
    ct = ColumnTransformer(
        [
            ("scaler", StandardScaler(), ["x1"]),
        ],
        remainder="passthrough",
    )
    estimator = Pipeline([("ct", ct), ("lr", LogisticRegression(random_state=42))])
    selector = PermutationImportanceRFECV(
        estimator,
        cv=cv,
        min_features_to_select=n_useful_features_classification,
        n_jobs=-1,
    )

    selector.fit(X_with_rand, y)
    assert (
        selector.ranking_[:n_useful_features_classification]
        == [1] * n_useful_features_classification
    ).all()
    assert (
        selector.support_
        == [True] * n_useful_features_classification + [False] * n_random_features
    ).all()


def test_float_step_param(
    x_y_classification_with_rand_columns_pandas,
    n_useful_features_classification,
    cv,
    random_state,
):
    X_with_rand, y = x_y_classification_with_rand_columns_pandas
    selector = PermutationImportanceRFECV(
        LogisticRegression(random_state=random_state),
        cv=cv,
        min_features_to_select=n_useful_features_classification,
        step=0.3,
    )

    selector.fit(X_with_rand, y)
    assert selector.cv_results_["n_features"] == [10, 7, 5, 4, 3, 2]


def test_perm_imp_rfecv_regression_base_case_np_arrays(
    x_y_regression_with_rand_columns_arrays,
    n_useful_features_regression,
    n_random_features,
    cv,
):
    X_with_rand, y = x_y_regression_with_rand_columns_arrays
    selector = PermutationImportanceRFECV(
        LinearRegression(), cv=cv, min_features_to_select=n_useful_features_regression
    )

    selector.fit(X_with_rand, y)
    assert (
        selector.ranking_[:n_useful_features_regression]
        == [1] * n_useful_features_regression
    ).all()
    assert (
        selector.support_
        == [True] * n_useful_features_regression + [False] * n_random_features
    ).all()


def test_perm_imp_rfecv_regression_base_case_pandas(
    x_y_regression_with_rand_columns_pandas,
    n_useful_features_regression,
    n_random_features,
    cv,
):
    X_with_rand, y = x_y_regression_with_rand_columns_pandas
    selector = PermutationImportanceRFECV(
        LinearRegression(), cv=cv, min_features_to_select=n_useful_features_regression
    )

    selector.fit(X_with_rand, y)
    assert (
        selector.ranking_[:n_useful_features_regression]
        == [1] * n_useful_features_regression
    ).all()
    assert (
        selector.support_
        == [True] * n_useful_features_regression + [False] * n_random_features
    ).all()


def test_rfecv_plotting_exception_raise(
    random_state,
):
    selector = PermutationImportanceRFECV(
        LogisticRegression(random_state=random_state),
        random_state=random_state,
    )

    with pytest.raises(NotFittedError):
        selector.plot()


def test_rfecv_set_min_features_to_select_base_case(
    x_y_classification_with_rand_columns_pandas,
    n_useful_features_classification,
    n_random_features,
    cv,
    random_state,
):
    X_with_rand, y = x_y_classification_with_rand_columns_pandas
    selector = PermutationImportanceRFECV(
        LogisticRegression(random_state=random_state),
        cv=cv,
        min_features_to_select=1,
    )
    selector.set_output(transform="pandas")
    selector.fit(X_with_rand, y)
    selector.set_min_features_to_select(n_useful_features_classification)
    assert (
        selector.support_
        == [True] * n_useful_features_classification + [False] * n_random_features
    ).all()
    columns_to_select = X_with_rand.columns[:n_useful_features_classification]
    assert (selector.get_feature_names_out() == columns_to_select).all()
    pd.testing.assert_frame_equal(
        selector.transform(X_with_rand), X_with_rand[columns_to_select]
    )


def test_rfecv_set_min_features_to_select_exception_cases(
    random_state,
    n_useful_features_classification,
    cv,
    x_y_classification_with_rand_columns_pandas,
):
    X_with_rand, y = x_y_classification_with_rand_columns_pandas
    selector = PermutationImportanceRFECV(
        LogisticRegression(random_state=random_state),
        cv=cv,
        min_features_to_select=n_useful_features_classification,
    )
    selector.set_output(transform="pandas")

    with pytest.raises(NotFittedError):
        selector.set_min_features_to_select(n_useful_features_classification)

    selector.fit(X_with_rand, y)

    with pytest.raises(ValueError):
        selector.set_min_features_to_select(n_useful_features_classification - 1)

    with pytest.raises(ValueError):
        selector.set_min_features_to_select(X_with_rand.shape[0] + 1)
