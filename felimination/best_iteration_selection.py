"""Criteria for picking the best number of features from a fitted
selector's ``cv_results_``.

Each function in this module accepts the ``cv_results_`` dict produced by
`FeliminationRFECV`, `ForwardSelectorCV`, or any of their presets, and
returns one of the values in ``cv_results_["n_features"]``. Pass any of
them to the ``best_iteration_selection_criteria`` argument of those
selectors.
"""

import pandas as pd


def select_best_by_mean_test_score_and_overfit(cv_results):
    """Balance high test score against low overfit.

    Ranks each iteration by descending ``mean_test_score`` and by ascending
    ``overfit = mean_train_score - mean_test_score``, then picks the
    iteration that minimises the rank sum. Ties are broken by
    ``mean_test_score``.

    Parameters
    ----------
    cv_results : dict
        Must contain keys ``mean_test_score``, ``mean_train_score`` and
        ``n_features``.

    Returns
    -------
    int
        The chosen number of features. Guaranteed to be one of the values
        in ``cv_results["n_features"]``.
    """
    cv_df = pd.DataFrame(cv_results)
    cv_df["rank_mean_test_score"] = cv_df["mean_test_score"].rank(ascending=False)
    cv_df["overfit"] = cv_df["mean_train_score"] - cv_df["mean_test_score"]
    cv_df["rank_overfit"] = cv_df["overfit"].rank(ascending=True)
    cv_df["rank_sum"] = cv_df["rank_mean_test_score"] + cv_df["rank_overfit"]
    return cv_df.sort_values(["rank_sum", "mean_test_score"], ascending=[True, False])[
        "n_features"
    ].iloc[0]


def select_best_by_n_features_and_score(cv_results):
    """Balance high test score against a small number of features.

    Ranks each iteration by descending ``mean_test_score`` and by ascending
    ``n_features``, then picks the iteration that minimises the rank sum.
    Useful when you want a parsimonious model and are willing to give up a
    bit of score to drop features. Ties are broken by ``mean_test_score``.

    Parameters
    ----------
    cv_results : dict
        Must contain keys ``mean_test_score`` and ``n_features``.

    Returns
    -------
    int
        The chosen number of features. Guaranteed to be one of the values
        in ``cv_results["n_features"]``.
    """
    cv_df = pd.DataFrame(cv_results)
    cv_df["rank_mean_test_score"] = cv_df["mean_test_score"].rank(ascending=False)
    cv_df["rank_n_features"] = cv_df["n_features"].rank(ascending=True)
    cv_df["rank_sum"] = cv_df["rank_mean_test_score"] + cv_df["rank_n_features"]
    return cv_df.sort_values(["rank_sum", "mean_test_score"], ascending=[True, False])[
        "n_features"
    ].iloc[0]
