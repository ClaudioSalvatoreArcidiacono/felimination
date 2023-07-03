from felimination.rfe import PermutationImportanceRFECV
import pytest
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.datasets import make_classification, make_friedman1
import numpy as np
import pandas as pd


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
def x_y_classification_with_rand_columns_arrays(
    n_useful_features_classification, n_random_features, sample_size, random_state
):
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
    random_state,
):
    X_with_rand, y = x_y_classification_with_rand_columns_arrays
    selector = PermutationImportanceRFECV(
        LogisticRegression(random_state=random_state),
        n_features_to_select=n_useful_features_classification,
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
    random_state,
):
    X_with_rand, y = x_y_classification_with_rand_columns_pandas
    selector = PermutationImportanceRFECV(
        LogisticRegression(random_state=random_state),
        n_features_to_select=n_useful_features_classification,
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


def test_perm_imp_rfecv_regression_base_case_np_arrays(
    x_y_regression_with_rand_columns_arrays,
    n_useful_features_regression,
    n_random_features,
):
    X_with_rand, y = x_y_regression_with_rand_columns_arrays
    selector = PermutationImportanceRFECV(
        LinearRegression(), n_features_to_select=n_useful_features_regression
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
):
    X_with_rand, y = x_y_regression_with_rand_columns_pandas
    selector = PermutationImportanceRFECV(
        LinearRegression(), n_features_to_select=n_useful_features_regression
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
