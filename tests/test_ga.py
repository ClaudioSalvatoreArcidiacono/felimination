import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_friedman1
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import GroupKFold, RepeatedKFold, RepeatedStratifiedKFold

from felimination.ga import (
    HybridImportanceGACVFeatureSelector,
    rank_mean_test_score_fitness,
    rank_mean_test_score_overfit_fitness,
)


_OVERFIT_FITNESS = rank_mean_test_score_overfit_fitness
from felimination.importance import PermutationImportance
from felimination.mrmr import MRMRRanker


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


def test_find_best_features_classification_n_jobs_arrays(
    x_y_classification_with_rand_columns_arrays,
    n_useful_features_classification,
    n_random_features,
    cv_classification,
    random_state,
    n_jobs=-1,
):
    X, y = x_y_classification_with_rand_columns_arrays
    selector = HybridImportanceGACVFeatureSelector(
        LogisticRegression(random_state=random_state),
        random_state=random_state,
        cv=cv_classification,
        n_jobs=n_jobs,
        init_avg_features_num=n_useful_features_classification,
        init_std_features_num=1,
        fitness_function=_OVERFIT_FITNESS,
    )
    selector.fit(X, y)

    assert (
        selector.support_
        == [True] * n_useful_features_classification + [False] * n_random_features
    ).all()


def test_find_best_features_classification_n_jobs_string_fitness(
    x_y_classification_with_rand_columns_arrays,
    n_useful_features_classification,
    cv_classification,
    random_state,
    n_jobs=-1,
):
    X, y = x_y_classification_with_rand_columns_arrays
    selector = HybridImportanceGACVFeatureSelector(
        LogisticRegression(random_state=random_state),
        random_state=random_state,
        cv=cv_classification,
        n_jobs=n_jobs,
        init_avg_features_num=n_useful_features_classification,
        init_std_features_num=1,
        fitness_function="mean_test_score",
    )
    selector.fit(X, y)

    assert (
        selector.support_[:n_useful_features_classification].sum()
        >= n_useful_features_classification - 1
    )
    assert selector.support_[n_useful_features_classification:].sum() <= 2


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
        fitness_function=_OVERFIT_FITNESS,
    )
    selector.fit(X, y)
    assert selector.support_[:n_useful_features_regression].sum() >= 4
    assert selector.support_[n_useful_features_regression:].sum() <= 1


@pytest.mark.parametrize(("n_jobs"), [1, -1])
def test_find_best_features_classification_n_jobs_pandas(  # parallel:  73s; sequential: 2m 8s
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
        fitness_function=_OVERFIT_FITNESS,
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
        fitness_function=_OVERFIT_FITNESS,
    )
    selector.fit(X, y)
    assert selector.support_[:n_useful_features_regression].sum() >= 4
    assert selector.support_[n_useful_features_regression:].sum() <= 1


# ── mutation_candidate_scorer helpers ────────────────────────────────────────


def _make_pool_element(features, importances):
    return {
        "features": features,
        "mean_cv_importances": importances,
        "mean_test_score": 0.8,
        "mean_train_score": 0.9,
    }


def _fixed_scorer(n_features, high_indices):
    """Return a scorer giving 100.0 to ``high_indices``, 0.0 to the rest."""

    def scorer(X, y, selected_features):
        scores = np.zeros(n_features)
        scores[list(high_indices)] = 100.0
        return scores

    return scorer


def _make_selector_for_mutate(scorer, selection, n_mutations=20):
    """Minimal selector configured for deterministic _mutate unit tests."""
    return HybridImportanceGACVFeatureSelector(
        LogisticRegression(),
        mutation_candidate_scorer=scorer,
        mutation_candidate_selection=selection,
        n_mutations=n_mutations,
        # keep element size fixed: delta always 0, replace exactly 1
        range_change_n_features_mutation=(0, 1),
        range_randomly_swapped_features_mutation=(1, 2),
        min_features_to_select=1,
    )


# ── _mutate unit tests ───────────────────────────────────────────────────────


def test_mutate_best_selection_prefers_highest_scored_candidate(random_state):
    """'best' mode always picks the highest-scored candidate as the replacement."""
    np.random.seed(random_state)
    n_features = 5
    # pool element holds features 1,2,3; candidates are 0 (score=100) and 4 (score=0)
    pool = [_make_pool_element([1, 2, 3], [0.3, 0.2, 0.1])]
    all_features = list(range(n_features))
    X = np.random.rand(50, n_features)
    y = np.random.randint(0, 2, 50)

    selector = _make_selector_for_mutate(_fixed_scorer(n_features, [0]), "best")
    mutations = selector._mutate(pool, all_features, X, y)

    # kept=[1,2], ordered_candidates=[0,4]; result is [1,2,0] for every mutation.
    for m in mutations:
        assert 0 in m["features"], f"Expected feature 0 in {m['features']}"


def test_mutate_best_selection_inf_scored_candidates_placed_last(random_state):
    """Features with -inf score are never picked before finite-scored ones in 'best' mode."""
    np.random.seed(random_state)
    n_features = 5

    def scorer_with_inf(X, y, selected_features):
        scores = np.full(n_features, -np.inf)
        scores[4] = 100.0
        return scores

    pool = [_make_pool_element([1, 2, 3], [0.3, 0.2, 0.1])]
    all_features = list(range(n_features))
    X = np.random.rand(50, n_features)
    y = np.random.randint(0, 2, 50)

    selector = _make_selector_for_mutate(scorer_with_inf, "best")
    mutations = selector._mutate(pool, all_features, X, y)

    for m in mutations:
        assert 4 in m["features"], f"Expected feature 4 in {m['features']}"
        assert (
            0 not in m["features"]
        ), f"Feature 0 (-inf) should not appear in {m['features']}"


def test_mutate_sample_selection_only_picks_valid_candidates(random_state):
    """'sample' mode never includes features already in the element, no duplicates."""
    np.random.seed(random_state)
    n_features = 6
    pool = [_make_pool_element([0, 1], [0.5, 0.5])]
    all_features = list(range(n_features))
    X = np.random.rand(50, n_features)
    y = np.random.randint(0, 2, 50)

    # features 4 and 5 have score 0; ensure they can still be sampled (eps prevents prob=0)
    scorer = _fixed_scorer(n_features, [2, 3])
    selector = _make_selector_for_mutate(scorer, "sample", n_mutations=30)
    mutations = selector._mutate(pool, all_features, X, y)

    for m in mutations:
        assert len(m["features"]) == len(set(m["features"])), "Duplicate features found"
        assert all(f in all_features for f in m["features"]), "Invalid feature found"


def test_mutate_sample_selection_excludes_inf_scored_candidates(random_state):
    """'sample' mode never selects -inf scored candidates when finite ones exist."""
    np.random.seed(random_state)
    n_features = 5

    def scorer_with_inf(X, y, selected_features):
        scores = np.full(n_features, -np.inf)
        scores[4] = 100.0
        return scores

    pool = [_make_pool_element([1, 2, 3], [0.3, 0.2, 0.1])]
    all_features = list(range(n_features))
    X = np.random.rand(50, n_features)
    y = np.random.randint(0, 2, 50)

    selector = _make_selector_for_mutate(scorer_with_inf, "sample")
    mutations = selector._mutate(pool, all_features, X, y)

    for m in mutations:
        assert 0 not in m["features"], f"Feature 0 (-inf) appeared in {m['features']}"


def test_mutate_best_selection_pandas(random_state):
    """'best' mode resolves column names correctly for DataFrame inputs."""
    np.random.seed(random_state)
    columns = ["x1", "x2", "rand1", "rand2", "rand3"]
    X = pd.DataFrame(np.random.rand(50, 5), columns=columns)
    y = pd.Series(np.random.randint(0, 2, 50))

    def scorer(X_df, y_, selected_features):
        scores = np.zeros(X_df.shape[1])
        scores[0] = 100.0  # x1 gets highest score
        return scores

    # candidates are x1 (score=100) and x2 (score=0)
    pool = [_make_pool_element(["rand1", "rand2", "rand3"], [0.3, 0.2, 0.1])]

    selector = _make_selector_for_mutate(scorer, "best")
    mutations = selector._mutate(pool, columns, X, y)

    for m in mutations:
        assert "x1" in m["features"], f"Expected 'x1' in {m['features']}"


# ── end-to-end: MRMRRanker as mutation_candidate_scorer ─────────────────────


def test_find_best_features_classification_mrmr_mutation_scorer_best(
    x_y_classification_with_rand_columns_arrays,
    n_useful_features_classification,
    cv_classification,
    random_state,
):
    """End-to-end: MRMRRanker relevance scores guide mutation candidate selection ('best')."""
    X, y = x_y_classification_with_rand_columns_arrays
    mrmr_ranker = MRMRRanker(regression=False, random_state=random_state)

    selector = HybridImportanceGACVFeatureSelector(
        LogisticRegression(random_state=random_state),
        random_state=random_state,
        cv=cv_classification,
        init_avg_features_num=n_useful_features_classification,
        init_std_features_num=1,
        mutation_candidate_scorer=mrmr_ranker,
        mutation_candidate_selection="best",
        fitness_function=_OVERFIT_FITNESS,
    )
    selector.fit(X, y)

    assert (
        selector.support_[:n_useful_features_classification].sum()
        >= n_useful_features_classification - 1
    )
    assert selector.support_[n_useful_features_classification:].sum() <= 2


def test_find_best_features_classification_mrmr_mutation_scorer_sample(
    x_y_classification_with_rand_columns_arrays,
    n_useful_features_classification,
    cv_classification,
    random_state,
):
    """End-to-end: MRMRRanker relevance scores guide mutation candidate selection ('sample')."""
    X, y = x_y_classification_with_rand_columns_arrays
    mrmr_ranker = MRMRRanker(regression=False, random_state=random_state)

    selector = HybridImportanceGACVFeatureSelector(
        LogisticRegression(random_state=random_state),
        random_state=random_state,
        cv=cv_classification,
        init_avg_features_num=n_useful_features_classification,
        init_std_features_num=1,
        mutation_candidate_scorer=mrmr_ranker,
        mutation_candidate_selection="sample",
    )
    selector.fit(X, y)

    assert (
        selector.support_[:n_useful_features_classification].sum()
        >= n_useful_features_classification - 1
    )
    assert selector.support_[n_useful_features_classification:].sum() <= 2


def test_find_best_features_classification_mrmr_mutation_scorer_pandas(
    x_y_classification_with_rand_columns_pandas,
    n_useful_features_classification,
    cv_classification,
    random_state,
):
    """MRMRRanker must handle pandas DataFrames (string column names as selected_idx)."""
    X, y = x_y_classification_with_rand_columns_pandas
    mrmr_ranker = MRMRRanker(regression=False, random_state=random_state)

    selector = HybridImportanceGACVFeatureSelector(
        LogisticRegression(random_state=random_state),
        random_state=random_state,
        cv=cv_classification,
        init_avg_features_num=n_useful_features_classification,
        init_std_features_num=1,
        mutation_candidate_scorer=mrmr_ranker,
        mutation_candidate_selection="sample",
        fitness_function=_OVERFIT_FITNESS,
    )
    selector.fit(X, y)

    assert (
        selector.support_[:n_useful_features_classification].sum()
        >= n_useful_features_classification - 1
    )
    assert selector.support_[n_useful_features_classification:].sum() <= 2


def test_groups_forwarded_to_cv_split(random_state):
    """groups passed to fit() must reach cv.split() — not silently ignored."""
    X, y = make_classification(n_samples=50, n_features=4, random_state=random_state)
    groups = np.tile(np.arange(5), 10)  # 5 groups of 10 samples each

    splits_received = []

    class _CapturingGroupKFold(GroupKFold):
        def split(self, X, y=None, groups=None):
            splits_received.append(groups)
            return super().split(X, y, groups=groups)

    selector = HybridImportanceGACVFeatureSelector(
        LogisticRegression(random_state=random_state),
        random_state=random_state,
        cv=_CapturingGroupKFold(n_splits=5),
        max_generations=2,
        init_avg_features_num=2,
        init_std_features_num=1,
    )
    selector.fit(X, y, groups=groups)

    assert len(splits_received) > 0
    for received in splits_received:
        np.testing.assert_array_equal(received, groups)


def test_generation_stored_in_solutions_and_cache(random_state):
    """generation must be recorded on best_solutions_ entries and in evaluation_cache_."""
    X, y = make_classification(n_samples=50, n_features=4, random_state=random_state)
    max_generations = 4

    selector = HybridImportanceGACVFeatureSelector(
        LogisticRegression(random_state=random_state),
        random_state=random_state,
        max_generations=max_generations,
        init_avg_features_num=2,
        init_std_features_num=1,
    )
    selector.fit(X, y)

    # Each best_solutions_ entry must carry the 1-based generation it was recorded at.
    for i, solution in enumerate(selector.best_solutions_, start=1):
        assert "generation" in solution
        assert solution["generation"] == i

    # Every cache entry must have a generation: 0 for initial pool, ≥1 for loop.
    for entry in selector.evaluation_cache_.values():
        assert "generation" in entry
        assert entry["generation"] >= 0
