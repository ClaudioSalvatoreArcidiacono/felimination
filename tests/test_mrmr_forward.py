"""Extensive tests for felimination.forward.ForwardSelectorCV and
felimination.mrmr.{MRMRRanker, MRMRCV, abs_pearson_correlation}."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression, LogisticRegression

from felimination.forward import ForwardSelectorCV
from felimination.mrmr import MRMRCV, MRMRRanker, abs_pearson_correlation


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_classification():
    X, y = make_classification(
        n_samples=100,
        n_features=6,
        n_informative=3,
        n_redundant=0,
        n_repeated=0,
        random_state=0,
    )
    return X, y


@pytest.fixture
def small_regression():
    X, y = make_regression(
        n_samples=100,
        n_features=6,
        n_informative=3,
        noise=0.1,
        random_state=0,
    )
    return X, y


@pytest.fixture
def tiny_X():
    """5×3 float array for unit-level ranker tests."""
    return np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [2.0, 1.0, 4.0],
            [3.0, 6.0, 2.0],
        ]
    )


@pytest.fixture
def tiny_y():
    return np.array([0, 1, 0, 1, 0])


# ---------------------------------------------------------------------------
# Helpers for deterministic ranker unit tests
# ---------------------------------------------------------------------------


def _const_rel(vals):
    return lambda X, y: np.array(vals, dtype=float)


def _const_red(vals):
    return lambda X, y_feat: np.array(vals, dtype=float)


def _make_ranker(**kw):
    """MRMRRanker with a dummy relevance_func so real MI is never called."""
    kw.setdefault("relevance_func", lambda X, y: np.zeros(X.shape[1]))
    return MRMRRanker(**kw)


# ===========================================================================
# abs_pearson_correlation
# ===========================================================================


def test_abs_pearson_output_shape():
    rng = np.random.default_rng(0)
    X = rng.random((20, 5))
    y = rng.random(20)
    assert abs_pearson_correlation(X, y).shape == (5,)


def test_abs_pearson_perfect_positive_correlation():
    x = np.arange(20, dtype=float)
    result = abs_pearson_correlation(x.reshape(-1, 1), 3 * x + 1)
    np.testing.assert_allclose(result, [1.0], atol=1e-10)


def test_abs_pearson_perfect_negative_correlation():
    x = np.arange(20, dtype=float)
    result = abs_pearson_correlation(x.reshape(-1, 1), -2 * x + 5)
    np.testing.assert_allclose(result, [1.0], atol=1e-10)


def test_abs_pearson_zero_variance_feature_no_crash():
    X = np.ones((10, 3))
    y = np.arange(10, dtype=float)
    result = abs_pearson_correlation(X, y)
    assert result.shape == (3,)
    assert np.all(np.isfinite(result))


def test_abs_pearson_accepts_dataframe():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [4.0, 3.0, 2.0, 1.0]})
    y = np.array([1.0, 2.0, 3.0, 4.0])
    result = abs_pearson_correlation(X, y)
    assert result.shape == (2,)
    np.testing.assert_allclose(result[0], 1.0, atol=1e-10)
    np.testing.assert_allclose(result[1], 1.0, atol=1e-10)


# ===========================================================================
# MRMRRanker – _resolve_discrete_mask
# ===========================================================================


def test_discrete_mask_auto_numpy_all_continuous():
    ranker = _make_ranker()
    mask = ranker._resolve_discrete_mask(np.zeros((5, 4)))
    assert mask.shape == (4,)
    assert not mask.any()


def test_discrete_mask_auto_dataframe_object_is_discrete():
    ranker = _make_ranker()
    X = pd.DataFrame({"num": [1.0, 2.0, 3.0], "obj": ["a", "b", "c"]})
    mask = ranker._resolve_discrete_mask(X)
    assert mask.tolist() == [False, True]


def test_discrete_mask_auto_dataframe_categorical_is_discrete():
    ranker = _make_ranker()
    X = pd.DataFrame({"num": [1.0, 2.0, 3.0], "cat": pd.Categorical(["a", "b", "a"])})
    mask = ranker._resolve_discrete_mask(X)
    assert mask.tolist() == [False, True]


def test_discrete_mask_auto_dataframe_string_dtype_is_discrete():
    ranker = _make_ranker()
    X = pd.DataFrame(
        {
            "num": [1.0, 2.0, 3.0],
            "s": pd.array(["a", "b", "c"], dtype=pd.StringDtype()),
        }
    )
    mask = ranker._resolve_discrete_mask(X)
    assert mask.tolist() == [False, True]


def test_discrete_mask_auto_dataframe_numeric_only_all_continuous():
    ranker = _make_ranker()
    X = pd.DataFrame({"a": [1.0, 2.0], "b": [3, 4], "c": [5.0, 6.0]})
    assert not ranker._resolve_discrete_mask(X).any()


def test_discrete_mask_auto_dataframe_mixed_types():
    ranker = _make_ranker()
    X = pd.DataFrame(
        {
            "f_float": [1.0, 2.0, 3.0],
            "f_obj": ["a", "b", "c"],
            "f_cat": pd.Categorical(["x", "y", "x"]),
            "f_str": pd.array(["p", "q", "r"], dtype=pd.StringDtype()),
            "f_int": [10, 20, 30],
        }
    )
    assert ranker._resolve_discrete_mask(X).tolist() == [False, True, True, True, False]


def test_discrete_mask_bool_true_all_discrete():
    ranker = _make_ranker(discrete_features=True)
    mask = ranker._resolve_discrete_mask(np.zeros((4, 5)))
    assert mask.shape == (5,)
    assert mask.all()


def test_discrete_mask_bool_false_all_continuous():
    ranker = _make_ranker(discrete_features=False)
    assert not ranker._resolve_discrete_mask(np.zeros((4, 5))).any()


def test_discrete_mask_bool_array_passed_through():
    bool_arr = np.array([True, False, True, False])
    ranker = _make_ranker(discrete_features=bool_arr)
    np.testing.assert_array_equal(
        ranker._resolve_discrete_mask(np.zeros((4, 4))), bool_arr
    )


def test_discrete_mask_int_array_indices():
    ranker = _make_ranker(discrete_features=np.array([0, 2]))
    assert ranker._resolve_discrete_mask(np.zeros((4, 4))).tolist() == [
        True,
        False,
        True,
        False,
    ]


def test_discrete_mask_wrong_shape_raises():
    ranker = _make_ranker(discrete_features=np.array([True, False]))
    with pytest.raises(ValueError, match="discrete_features mask"):
        ranker._resolve_discrete_mask(np.zeros((4, 4)))


# ===========================================================================
# MRMRRanker – parameter validation
# ===========================================================================


def test_scheme_invalid_raises():
    with pytest.raises(ValueError, match="scheme"):
        MRMRRanker(scheme="invalid")


@pytest.mark.parametrize("scheme", ["ratio", "difference"])
def test_scheme_valid_does_not_raise(scheme):
    MRMRRanker(scheme=scheme)


# ===========================================================================
# MRMRRanker – __call__ (unit tests with deterministic custom functions)
# ===========================================================================


def test_first_call_returns_relevance(tiny_X, tiny_y):
    relevances = np.array([0.8, 0.5, 0.3])
    ranker = MRMRRanker(relevance_func=_const_rel(relevances))
    np.testing.assert_array_equal(ranker(tiny_X, tiny_y, []), relevances)


def test_first_call_sets_relevance_attr(tiny_X, tiny_y):
    relevances = np.array([0.8, 0.5, 0.3])
    ranker = MRMRRanker(relevance_func=_const_rel(relevances))
    ranker(tiny_X, tiny_y, [])
    np.testing.assert_array_equal(ranker.relevance_, relevances)


def test_first_call_resets_state_on_repeat(tiny_X, tiny_y):
    call_count = [0]

    def relevance_func(X, y):
        call_count[0] += 1
        return np.full(X.shape[1], float(call_count[0]))

    ranker = MRMRRanker(relevance_func=relevance_func)
    ranker(tiny_X, tiny_y, [])
    ranker(tiny_X, tiny_y, [0])
    ranker(tiny_X, tiny_y, [])  # second reset
    assert ranker.relevance_[0] == 2.0
    assert ranker._seen == set()


def test_scheme_difference_combines_correctly(tiny_X, tiny_y):
    relevances = np.array([0.8, 0.5, 0.3])
    redundances = np.array([0.1, 0.2, 0.05])
    ranker = MRMRRanker(
        scheme="difference",
        relevance_func=_const_rel(relevances),
        redundance_func=_const_red(redundances),
    )
    ranker(tiny_X, tiny_y, [])
    scores = ranker(tiny_X, tiny_y, [0])
    np.testing.assert_allclose(scores, relevances - redundances)


def test_scheme_ratio_combines_correctly(tiny_X, tiny_y):
    relevances = np.array([0.8, 0.5, 0.3])
    redundances = np.array([0.2, 0.5, 0.1])
    ranker = MRMRRanker(
        scheme="ratio",
        relevance_func=_const_rel(relevances),
        redundance_func=_const_red(redundances),
    )
    ranker(tiny_X, tiny_y, [])
    scores = ranker(tiny_X, tiny_y, [0])
    np.testing.assert_allclose(scores, relevances / redundances)


def test_scheme_ratio_near_zero_denominator_no_inf(tiny_X, tiny_y):
    ranker = MRMRRanker(
        scheme="ratio",
        relevance_func=_const_rel([0.8, 0.5, 0.3]),
        redundance_func=_const_red([0.0, 0.0, 0.0]),
    )
    ranker(tiny_X, tiny_y, [])
    scores = ranker(tiny_X, tiny_y, [0])
    assert np.all(np.isfinite(scores))


def test_redundancy_mean_over_multiple_selected(tiny_X, tiny_y):
    redundances = np.array([0.4, 0.2, 0.1])
    ranker = MRMRRanker(
        scheme="difference",
        relevance_func=_const_rel([1.0, 1.0, 1.0]),
        redundance_func=_const_red(redundances),
    )
    ranker(tiny_X, tiny_y, [])
    # Both selected features return the same redundances vector,
    # so mean_red = (redundances + redundances) / 2 = redundances.
    scores = ranker(tiny_X, tiny_y, [0, 1])
    np.testing.assert_allclose(scores, 1.0 - redundances)


def test_seen_set_prevents_double_counting(tiny_X, tiny_y):
    call_count = [0]

    def redundance_func(X, y_feat):
        call_count[0] += 1
        return np.array([0.1, 0.2, 0.05])

    ranker = MRMRRanker(
        relevance_func=_const_rel([0.8, 0.5, 0.3]),
        redundance_func=redundance_func,
    )
    ranker(tiny_X, tiny_y, [])
    ranker(tiny_X, tiny_y, [0])
    assert call_count[0] == 1
    ranker(tiny_X, tiny_y, [0])  # feature 0 already seen
    assert call_count[0] == 1
    ranker(tiny_X, tiny_y, [0, 1])  # feature 1 is new
    assert call_count[0] == 2


def test_min_relevance_masks_on_first_call(tiny_X, tiny_y):
    ranker = MRMRRanker(
        relevance_func=_const_rel([0.8, 0.3, 0.5]),
        min_relevance=0.4,
    )
    scores = ranker(tiny_X, tiny_y, [])
    assert scores[0] == pytest.approx(0.8)
    assert scores[2] == pytest.approx(0.5)
    assert scores[1] == -np.inf


def test_min_relevance_masks_on_subsequent_call(tiny_X, tiny_y):
    ranker = MRMRRanker(
        scheme="difference",
        relevance_func=_const_rel([0.8, 0.3, 0.5]),
        redundance_func=_const_red([0.1, 0.1, 0.1]),
        min_relevance=0.4,
    )
    ranker(tiny_X, tiny_y, [])
    scores = ranker(tiny_X, tiny_y, [0])
    assert scores[0] == pytest.approx(0.7)
    assert scores[1] == -np.inf


def test_max_redundancy_not_applied_on_first_call(tiny_X, tiny_y):
    ranker = MRMRRanker(
        relevance_func=_const_rel([0.8, 0.5, 0.3]),
        redundance_func=_const_red([999.0, 999.0, 999.0]),
        max_redundancy=0.01,
    )
    scores = ranker(tiny_X, tiny_y, [])
    assert not np.any(scores == -np.inf)


def test_max_redundancy_masks_on_subsequent_call(tiny_X, tiny_y):
    ranker = MRMRRanker(
        scheme="difference",
        relevance_func=_const_rel([0.8, 0.5, 0.3]),
        redundance_func=_const_red([0.1, 0.9, 0.3]),
        max_redundancy=0.5,
    )
    ranker(tiny_X, tiny_y, [])
    scores = ranker(tiny_X, tiny_y, [0])
    assert scores[0] == pytest.approx(0.7)
    assert scores[1] == -np.inf
    assert scores[2] == pytest.approx(0.0)


def test_both_thresholds_combined(tiny_X, tiny_y):
    ranker = MRMRRanker(
        scheme="difference",
        relevance_func=_const_rel([0.8, 0.2, 0.6]),
        redundance_func=_const_red([0.1, 0.1, 0.9]),
        min_relevance=0.3,
        max_redundancy=0.5,
    )
    ranker(tiny_X, tiny_y, [])
    scores = ranker(tiny_X, tiny_y, [0])
    assert scores[0] == pytest.approx(0.7)  # passes both thresholds
    assert scores[1] == -np.inf  # fails min_relevance (0.2 < 0.3)
    assert scores[2] == -np.inf  # fails max_redundancy (0.9 > 0.5)


def test_custom_relevance_func_is_called(tiny_X, tiny_y):
    calls = []

    def relevance_func(X, y):
        calls.append(X.shape)
        return np.ones(X.shape[1])

    ranker = MRMRRanker(relevance_func=relevance_func)
    ranker(tiny_X, tiny_y, [])
    assert len(calls) == 1


def test_custom_redundance_func_is_called(tiny_X, tiny_y):
    calls = []

    def redundance_func(X, y_feat):
        calls.append(y_feat.shape)
        return np.ones(X.shape[1])

    ranker = MRMRRanker(
        relevance_func=_const_rel([0.8, 0.5, 0.3]),
        redundance_func=redundance_func,
    )
    ranker(tiny_X, tiny_y, [])
    ranker(tiny_X, tiny_y, [0])
    assert len(calls) == 1


# ===========================================================================
# MRMRRanker – integration (with real mutual information)
# ===========================================================================


def test_mrmr_ranker_classif_returns_valid_scores(tiny_X, tiny_y):
    ranker = MRMRRanker(regression=False, n_neighbors=2)
    scores = ranker(tiny_X, tiny_y, [])
    assert scores.shape == (3,)
    assert np.all(scores >= 0)


def test_mrmr_ranker_regression_returns_valid_scores(tiny_X, tiny_y):
    ranker = MRMRRanker(regression=True, n_neighbors=2)
    scores = ranker(tiny_X, tiny_y, [])
    assert scores.shape == (3,)
    assert np.all(scores >= 0)


def test_mrmr_ranker_dataframe_input_no_crash(tiny_y):
    X_df = pd.DataFrame(
        {
            "f0": [1.0, 2.0, 7.0, 2.0, 3.0],
            "f1": [4.0, 5.0, 1.0, 3.0, 2.0],
            "f2": [2.0, 1.0, 3.0, 5.0, 4.0],
        }
    )
    ranker = MRMRRanker(regression=False, n_neighbors=2)
    scores = ranker(X_df, tiny_y, [])
    assert scores.shape == (3,)


def test_mrmr_ranker_subsequent_call_difference_scheme(tiny_X, tiny_y):
    ranker = MRMRRanker(regression=False, scheme="difference", n_neighbors=2)
    ranker(tiny_X, tiny_y, [])
    scores = ranker(tiny_X, tiny_y, [0])
    assert scores.shape == (3,)


# ===========================================================================
# ForwardSelectorCV
# ===========================================================================


def test_forward_selector_fit_numpy_classification(small_classification):
    X, y = small_classification
    selector = ForwardSelectorCV(
        LogisticRegression(random_state=0),
        min_features_to_select=2,
        max_features_to_select=4,
        cv=2,
    ).fit(X, y)
    assert 2 <= selector.n_features_ <= 4
    assert selector.support_.sum() == selector.n_features_


def test_forward_selector_fit_numpy_regression(small_regression):
    X, y = small_regression
    selector = ForwardSelectorCV(
        LinearRegression(),
        min_features_to_select=2,
        max_features_to_select=4,
        cv=2,
    ).fit(X, y)
    assert hasattr(selector, "cv_results_")
    assert selector.support_.shape == (6,)


def test_forward_selector_fit_pandas_preserves_feature_names(small_classification):
    X_arr, y = small_classification
    col_names = [f"feat_{i}" for i in range(X_arr.shape[1])]
    X = pd.DataFrame(X_arr, columns=col_names)
    selector = ForwardSelectorCV(
        LogisticRegression(random_state=0),
        min_features_to_select=2,
        max_features_to_select=4,
        cv=2,
    ).fit(X, y)
    np.testing.assert_array_equal(selector.feature_names_in_, col_names)


def test_forward_selector_cv_results_expected_keys(small_classification):
    X, y = small_classification
    selector = ForwardSelectorCV(
        LogisticRegression(random_state=0),
        min_features_to_select=2,
        max_features_to_select=4,
        cv=2,
    ).fit(X, y)
    expected = {
        "n_features",
        "mean_test_score",
        "std_test_score",
        "mean_train_score",
        "std_train_score",
    }
    assert expected <= set(selector.cv_results_.keys())


def test_forward_selector_cv_results_n_features_integer_step(small_classification):
    X, y = small_classification
    selector = ForwardSelectorCV(
        LogisticRegression(random_state=0),
        min_features_to_select=2,
        max_features_to_select=6,
        step=2,
        cv=2,
    ).fit(X, y)
    assert selector.cv_results_["n_features"] == [2, 4, 6]


def test_forward_selector_cv_results_n_features_float_step(small_classification):
    X, y = small_classification
    selector = ForwardSelectorCV(
        LogisticRegression(random_state=0),
        min_features_to_select=2,
        max_features_to_select=6,
        step=0.5,
        cv=2,
    ).fit(X, y)
    # n_features=6, step=0.5 (based on n_selected):
    # pre-eval → 2; int(0.5*2)=1 → 3; int(0.5*3)=1 → 4; int(0.5*4)=2 → 6
    assert selector.cv_results_["n_features"] == [2, 3, 4, 6]


def test_forward_selector_ranking_first_feature_is_rank_one(small_classification):
    X, y = small_classification
    selector = ForwardSelectorCV(
        LogisticRegression(random_state=0),
        max_features_to_select=4,
        cv=2,
    ).fit(X, y)
    assert (selector.ranking_ == 1).sum() == 1


def test_forward_selector_all_features_have_nonzero_rank(small_classification):
    X, y = small_classification
    selector = ForwardSelectorCV(
        LogisticRegression(random_state=0),
        max_features_to_select=4,
        cv=2,
    ).fit(X, y)
    assert np.all(selector.ranking_ > 0)


def test_forward_selector_set_n_features_changes_support(small_classification):
    X, y = small_classification
    selector = ForwardSelectorCV(
        LogisticRegression(random_state=0),
        min_features_to_select=2,
        max_features_to_select=5,
        step=1,
        cv=2,
    ).fit(X, y)
    selector.set_n_features_to_select(3)
    assert selector.support_.sum() == 3


def test_forward_selector_set_n_features_invalid_value_raises(small_classification):
    X, y = small_classification
    selector = ForwardSelectorCV(
        LogisticRegression(random_state=0),
        min_features_to_select=2,
        max_features_to_select=5,
        step=1,
        cv=2,
    ).fit(X, y)
    with pytest.raises(ValueError):
        selector.set_n_features_to_select(1)  # 1 was never evaluated


def test_forward_selector_set_n_features_not_fitted_raises():
    selector = ForwardSelectorCV(LogisticRegression())
    with pytest.raises(NotFittedError):
        selector.set_n_features_to_select(2)


def test_forward_selector_transform_not_fitted_raises():
    selector = ForwardSelectorCV(LogisticRegression())
    with pytest.raises(NotFittedError):
        selector.transform(np.zeros((5, 3)))


def test_forward_selector_transform_reduces_columns(small_classification):
    X, y = small_classification
    selector = ForwardSelectorCV(
        LogisticRegression(random_state=0),
        min_features_to_select=2,
        max_features_to_select=4,
        step=1,
        cv=2,
    ).fit(X, y)
    selector.set_n_features_to_select(2)
    assert selector.transform(X).shape == (X.shape[0], 2)


def test_forward_selector_get_feature_names_out(small_classification):
    X_arr, y = small_classification
    col_names = [f"feat_{i}" for i in range(X_arr.shape[1])]
    X = pd.DataFrame(X_arr, columns=col_names)
    selector = ForwardSelectorCV(
        LogisticRegression(random_state=0),
        min_features_to_select=2,
        max_features_to_select=4,
        step=1,
        cv=2,
    ).fit(X, y)
    selector.set_n_features_to_select(2)
    out_names = selector.get_feature_names_out()
    assert len(out_names) == 2
    assert all(n in col_names for n in out_names)


def test_forward_selector_min_gt_max_raises(small_classification):
    X, y = small_classification
    selector = ForwardSelectorCV(
        LogisticRegression(),
        min_features_to_select=5,
        max_features_to_select=3,
        cv=2,
    )
    with pytest.raises(ValueError, match="min_features_to_select"):
        selector.fit(X, y)


def test_forward_selector_custom_importance_getter_receives_empty_first(
    small_classification,
):
    X, y = small_classification
    calls = []

    def getter(X, y, selected_idx):
        calls.append(list(selected_idx))
        return np.arange(X.shape[1], dtype=float)

    ForwardSelectorCV(
        LogisticRegression(random_state=0),
        importance_getter=getter,
        min_features_to_select=2,
        max_features_to_select=3,
        cv=2,
    ).fit(X, y)
    assert [] in calls


def test_forward_selector_importance_getter_wrong_shape_raises():
    X, y = make_classification(n_samples=50, n_features=4, random_state=0)

    def bad_getter(X, y, selected_idx):
        return np.array([1.0, 2.0])

    with pytest.raises(ValueError, match="importance_getter"):
        ForwardSelectorCV(
            LogisticRegression(),
            importance_getter=bad_getter,
            cv=2,
        ).fit(X, y)


def test_forward_selector_callbacks_called_at_each_evaluation(small_classification):
    X, y = small_classification
    log = []

    def callback(selector, scores):
        log.append(len(selector.cv_results_["n_features"]))

    ForwardSelectorCV(
        LogisticRegression(random_state=0),
        min_features_to_select=2,
        max_features_to_select=4,
        step=1,
        cv=2,
        callbacks=[callback],
    ).fit(X, y)
    # Evaluated at n_features = 2, 3, 4 → callback fires 3 times
    assert log == [1, 2, 3]


def test_forward_selector_multiple_callbacks(small_classification):
    X, y = small_classification
    log_a, log_b = [], []
    ForwardSelectorCV(
        LogisticRegression(random_state=0),
        min_features_to_select=2,
        max_features_to_select=3,
        step=1,
        cv=2,
        callbacks=[
            lambda sel, sc: log_a.append(1),
            lambda sel, sc: log_b.append(1),
        ],
    ).fit(X, y)
    assert len(log_a) == len(log_b) == 2


def test_forward_selector_plot_not_fitted_raises():
    with pytest.raises(NotFittedError):
        ForwardSelectorCV(LogisticRegression(), random_state=0).plot()


def test_forward_selector_best_iteration_callable(small_classification):
    X, y = small_classification
    selector = ForwardSelectorCV(
        LogisticRegression(random_state=0),
        min_features_to_select=2,
        max_features_to_select=5,
        step=1,
        cv=2,
        best_iteration_selection_criteria=lambda cv_results: 3,
    ).fit(X, y)
    assert selector.n_features_ == 3
    assert selector.support_.sum() == 3


def test_forward_selector_auto_getter_classifier(small_classification):
    X, y = small_classification
    selector = ForwardSelectorCV(
        LogisticRegression(random_state=0),
        importance_getter="auto",
        min_features_to_select=2,
        max_features_to_select=3,
        cv=2,
    ).fit(X, y)
    assert hasattr(selector, "cv_results_")


def test_forward_selector_auto_getter_regressor(small_regression):
    X, y = small_regression
    selector = ForwardSelectorCV(
        LinearRegression(),
        importance_getter="auto",
        min_features_to_select=2,
        max_features_to_select=3,
        cv=2,
    ).fit(X, y)
    assert hasattr(selector, "cv_results_")


def test_forward_selector_max_features_limits_selection(small_classification):
    X, y = small_classification
    selector = ForwardSelectorCV(
        LogisticRegression(random_state=0),
        max_features_to_select=3,
        cv=2,
    ).fit(X, y)
    assert selector.n_features_ <= 3


def test_forward_selector_n_features_in(small_classification):
    X, y = small_classification
    selector = ForwardSelectorCV(
        LogisticRegression(random_state=0),
        max_features_to_select=3,
        cv=2,
    ).fit(X, y)
    assert selector.n_features_in_ == X.shape[1]


# ===========================================================================
# MRMRCV
# ===========================================================================


def test_mrmrcv_fit_classification(small_classification):
    X, y = small_classification
    selector = MRMRCV(
        LogisticRegression(random_state=0),
        min_features_to_select=2,
        max_features_to_select=4,
        cv=2,
        random_state=0,
    ).fit(X, y)
    assert hasattr(selector, "cv_results_")
    assert 2 <= selector.n_features_ <= 4
    assert selector.support_.sum() == selector.n_features_


def test_mrmrcv_fit_regression(small_regression):
    X, y = small_regression
    selector = MRMRCV(
        LinearRegression(),
        min_features_to_select=2,
        max_features_to_select=4,
        cv=2,
        random_state=0,
    ).fit(X, y)
    assert hasattr(selector, "cv_results_")


def test_mrmrcv_regression_flag_from_is_classifier():
    assert MRMRCV(LogisticRegression()).importance_getter.regression is False
    assert MRMRCV(LinearRegression()).importance_getter.regression is True


def test_mrmrcv_params_forwarded_to_ranker():
    selector = MRMRCV(
        LogisticRegression(),
        scheme="ratio",
        n_neighbors=5,
        min_relevance=0.1,
        max_redundancy=0.8,
    )
    ranker = selector.importance_getter
    assert ranker.scheme == "ratio"
    assert ranker.n_neighbors == 5
    assert ranker.min_relevance == 0.1
    assert ranker.max_redundancy == 0.8


def test_mrmrcv_default_scheme_is_difference():
    assert MRMRCV(LogisticRegression()).importance_getter.scheme == "difference"


def test_mrmrcv_invalid_scheme_raises():
    with pytest.raises(ValueError, match="scheme"):
        MRMRCV(LogisticRegression(), scheme="bad")


def test_mrmrcv_stores_params_on_self():
    selector = MRMRCV(
        LogisticRegression(),
        scheme="ratio",
        min_relevance=0.05,
        max_redundancy=0.9,
        discrete_features=True,
    )
    assert selector.scheme == "ratio"
    assert selector.min_relevance == 0.05
    assert selector.max_redundancy == 0.9
    assert selector.discrete_features is True
