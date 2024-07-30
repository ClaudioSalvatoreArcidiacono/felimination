import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_friedman1
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold

from felimination.importance import PermutationImportance

from felimination.ga import (
    HybridImportanceGACVFeatureSelector,
    rank_mean_test_score_fitness,
    rank_mean_test_score_overfit_fitness,
)


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
    return 10000


@pytest.fixture(scope="session")
def cv_n_splits():
    return 5


@pytest.fixture(scope="session")
def cv_n_repeats():
    return 10


@pytest.fixture(scope="session")
def cv_classification(random_state, cv_n_splits, cv_n_repeats):
    return RepeatedStratifiedKFold(
        random_state=random_state, n_splits=cv_n_splits, n_repeats=cv_n_repeats
    )


@pytest.fixture(scope="session")
def cv_regression(random_state, cv_n_splits, cv_n_repeats):
    return RepeatedKFold(
        random_state=random_state, n_splits=cv_n_splits, n_repeats=cv_n_repeats
    )


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


def make_mock_pool_arrays(n_useful_features):
    return [
        {
            "features": list(range(n_useful_features)) + [n_useful_features],
        },
        {
            "features": list(range(n_useful_features - 1))
            + [n_useful_features, n_useful_features + 1],
        },
    ]


def pool_to_pandas(array_pool, n_useful_features):
    pool_pandas = []
    for element_array in array_pool:
        element_pandas = element_array.copy()
        features_pandas = []
        for feature in element_pandas["features"]:
            if feature < n_useful_features:
                features_pandas.append(f"x{feature+1}")
            else:
                features_pandas.append(f"rand{feature-n_useful_features+1}")
        element_pandas["features"] = features_pandas
        pool_pandas.append(element_pandas)

    return pool_pandas


def add_mock_scores_to_pool(pool):
    for i, element in enumerate(pool):
        element["mean_test_score"] = 0.8 - 0.05 * i
        element["mean_train_score"] = 0.9 - 0.05 * i
    return pool


@pytest.fixture(scope="session")
def pool_arrays_classification_no_score(n_useful_features_classification):
    return make_mock_pool_arrays(n_useful_features_classification)


@pytest.fixture(scope="session")
def pool_arrays_regression_no_score(n_useful_features_regression):
    return make_mock_pool_arrays(n_useful_features_regression)


@pytest.fixture(scope="session")
def pool_pandas_classification_no_score(n_useful_features_classification):
    return pool_to_pandas(make_mock_pool_arrays(n_useful_features_classification))


@pytest.fixture(scope="session")
def pool_pandas_regression_no_score(n_useful_features_regression):
    return pool_to_pandas(make_mock_pool_arrays(n_useful_features_regression))


@pytest.fixture(scope="session")
def pool_arrays_classification_with_scores(n_useful_features_classification):
    return add_mock_scores_to_pool(
        make_mock_pool_arrays(n_useful_features_classification)
    )


@pytest.fixture(scope="session")
def pool_arrays_regression_with_scores(n_useful_features_regression):
    return add_mock_scores_to_pool(make_mock_pool_arrays(n_useful_features_regression))


@pytest.fixture(scope="session")
def pool_pandas_classification_with_scores(n_useful_features_classification):
    return add_mock_scores_to_pool(
        pool_to_pandas(make_mock_pool_arrays(n_useful_features_classification))
    )


@pytest.fixture(scope="session")
def pool_pandas_regression_with_scores(n_useful_features_regression):
    return add_mock_scores_to_pool(
        pool_to_pandas(make_mock_pool_arrays(n_useful_features_regression))
    )


def test_rank_mean_test_score_overfit_fitness():
    pool = [
        {  # Best test score but highest overfit
            # fitness=2, second best solution, second highest fitness
            "features": ["a", "b", "c"],
            "mean_test_score": 0.8,
            "mean_train_score": 0.9,
        },
        {  # Second best test score but lowest overfit
            # fitness=3, best solution, highest fitness
            "features": ["d", "e", "f"],
            "mean_test_score": 0.75,
            "mean_train_score": 0.75,
        },
        {
            # fitness=1, third best solution, lowest fitness
            "features": ["g", "h", "i"],
            "mean_test_score": 0.7,
            "mean_train_score": 0.71,
        },
    ]
    fitness = rank_mean_test_score_overfit_fitness(pool)
    assert fitness == [2, 3, 1]


def test_rank_mean_test_score_fitness():
    pool = [
        {  # Best test score
            # fitness=3, best solution, highest fitness
            "features": ["a", "b", "c"],
            "mean_test_score": 0.8,
            "mean_train_score": 0.9,
        },
        {  # Second best test score
            # fitness=2, second best solution, second highest fitness
            "features": ["d", "e", "f"],
            "mean_test_score": 0.75,
            "mean_train_score": 0.75,
        },
        {
            # fitness=1, third best solution, lowest fitness
            "features": ["g", "h", "i"],
            "mean_test_score": 0.7,
            "mean_train_score": 0.71,
        },
    ]
    fitness = rank_mean_test_score_fitness(pool)
    assert fitness == [3, 2, 1]


@pytest.mark.parametrize(("n_jobs"), [1, -1])
def test_find_best_features_classification_n_jobs_arrays(
    x_y_classification_with_rand_columns_arrays,
    n_useful_features_classification,
    n_random_features,
    cv_classification,
    random_state,
    n_jobs,
):
    X, y = x_y_classification_with_rand_columns_arrays
    selector = HybridImportanceGACVFeatureSelector(
        LogisticRegression(random_state=random_state),
        random_state=random_state,
        cv=cv_classification,
        n_jobs=n_jobs,
        init_avg_features_num=n_useful_features_classification,
        init_std_features_num=1,
    )
    selector.fit(X, y)

    assert (
        selector.support_
        == [True] * n_useful_features_classification + [False] * n_random_features
    ).all()


def test_find_best_features_regression_n_jobs_arrays(
    x_y_regression_with_rand_columns_arrays,
    n_useful_features_regression,
    cv_regression,
    random_state,
    n_jobs=-1,
):
    X, y = x_y_regression_with_rand_columns_arrays
    selector = HybridImportanceGACVFeatureSelector(
        LinearRegression(),
        random_state=random_state,
        cv=cv_regression,
        n_jobs=n_jobs,
        init_avg_features_num=n_useful_features_regression,
        init_std_features_num=1,
        importance_getter=PermutationImportance(),
    )
    selector.fit(X, y)
    assert selector.support_[:n_useful_features_regression].sum() >= 4
    assert selector.support_[n_useful_features_regression:].sum() <= 1


@pytest.mark.parametrize(("n_jobs"), [1, -1])
def test_find_best_features_classification_n_jobs_pandas(
    x_y_classification_with_rand_columns_pandas,
    n_useful_features_classification,
    n_random_features,
    cv_classification,
    random_state,
    n_jobs,
):
    X, y = x_y_classification_with_rand_columns_pandas
    selector = HybridImportanceGACVFeatureSelector(
        LogisticRegression(random_state=random_state),
        random_state=random_state,
        cv=cv_classification,
        n_jobs=n_jobs,
        init_avg_features_num=n_useful_features_classification,
        init_std_features_num=1,
    )
    selector.fit(X, y)

    assert (
        selector.support_
        == [True] * n_useful_features_classification + [False] * n_random_features
    ).all()


def test_find_best_features_regression_n_jobs_pandas(
    x_y_regression_with_rand_columns_pandas,
    n_useful_features_regression,
    cv_regression,
    random_state,
    n_jobs=-1,
):
    X, y = x_y_regression_with_rand_columns_pandas
    selector = HybridImportanceGACVFeatureSelector(
        LinearRegression(),
        random_state=random_state,
        cv=cv_regression,
        n_jobs=n_jobs,
        init_avg_features_num=n_useful_features_regression,
        init_std_features_num=1,
        importance_getter=PermutationImportance(),
    )
    selector.fit(X, y)
    assert selector.support_[:n_useful_features_regression].sum() >= 4
    assert selector.support_[n_useful_features_regression:].sum() <= 1
