from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit

from felimination.drift import SampleSimilarityDriftRFE, PermImpSampleSimilarityDriftRFE
from felimination.importance import PermutationImportance


@pytest.fixture(scope="session")
def n_features():
    return 10


@pytest.fixture(scope="session")
def cv(random_state):
    return ShuffleSplit(random_state=random_state)


@pytest.fixture(scope="session")
def sample_size():
    return 1000


@pytest.fixture(scope="session")
def x_with_drifty_columns_arrays(n_features, sample_size, random_state):
    np.random.seed(random_state)
    X = np.random.random((sample_size, n_features))
    drifty_feature = np.reshape(np.linspace(-3, 3, sample_size), (sample_size, 1))
    date_column = np.reshape(
        np.arange(
            datetime(2023, 1, 1),
            datetime(2023, 1, 1) + timedelta(days=sample_size),
            step=timedelta(days=1),
        ),
        (sample_size, 1),
    )
    X_with_drifty_columns = np.concatenate(
        [X, drifty_feature, date_column], axis=1, dtype=object
    )
    return X_with_drifty_columns


@pytest.fixture(scope="session")
def x_with_drifty_columns_pandas(
    x_with_drifty_columns_arrays,
    n_features,
):
    column_names = [f"x{i+1}" for i in range(n_features)] + ["drifty_feature", "date"]
    X_pandas = pd.DataFrame(x_with_drifty_columns_arrays, columns=column_names)
    return X_pandas


def test_base_case_arrays(
    x_with_drifty_columns_arrays,
    n_features,
    cv,
    random_state,
):
    selector = SampleSimilarityDriftRFE(
        LogisticRegression(random_state=random_state),
        cv=cv,
        split_col=-1,
        importance_getter=PermutationImportance(),
    )

    selector.fit(x_with_drifty_columns_arrays)
    assert (selector.ranking_[:n_features] == [1] * n_features).all()
    assert (selector.support_ == [True] * n_features + [False] * 2).all()


def test_base_case_pandas(
    x_with_drifty_columns_pandas,
    n_features,
    cv,
    random_state,
):
    selector = SampleSimilarityDriftRFE(
        LogisticRegression(random_state=random_state),
        cv=cv,
        split_col="date",
        importance_getter=PermutationImportance(),
    )

    selector.fit(x_with_drifty_columns_pandas)
    assert (selector.ranking_[:n_features] == [1] * n_features).all()
    assert (selector.support_ == [True] * n_features + [False] * 2).all()


def test_perm_imp_base_case_arrays(
    x_with_drifty_columns_arrays,
    n_features,
    cv,
    random_state,
):
    selector = PermImpSampleSimilarityDriftRFE(
        LogisticRegression(random_state=random_state),
        cv=cv,
        split_col=-1,
    )

    selector.fit(x_with_drifty_columns_arrays)
    assert (selector.ranking_[:n_features] == [1] * n_features).all()
    assert (selector.support_ == [True] * n_features + [False] * 2).all()


def test_perm_imp_base_case_pandas(
    x_with_drifty_columns_pandas,
    n_features,
    cv,
    random_state,
):
    selector = PermImpSampleSimilarityDriftRFE(
        LogisticRegression(random_state=random_state),
        cv=cv,
        split_col="date",
    )

    selector.fit(x_with_drifty_columns_pandas)
    assert (selector.ranking_[:n_features] == [1] * n_features).all()
    assert (selector.support_ == [True] * n_features + [False] * 2).all()
