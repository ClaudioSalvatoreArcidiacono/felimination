from typing import Any
from sklearn.inspection import permutation_importance


class PermutationImportance:
    def __init__(
        self,
        scoring=None,
        n_repeats=5,
        n_jobs=None,
        random_state=None,
        sample_weight=None,
        max_samples=1.0,
    ):
        self.scoring = scoring
        self.n_repeats = n_repeats
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.sample_weight = sample_weight
        self.max_samples = max_samples

    def __call__(self, estimator, X, y) -> Any:
        return permutation_importance(
            estimator,
            X,
            y,
            scoring=self.scoring,
            n_repeats=self.n_repeats,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            sample_weight=self.sample_weight,
            max_samples=self.max_samples,
        ).importances_mean
