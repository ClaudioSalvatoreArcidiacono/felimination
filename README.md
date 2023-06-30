# felimination

## Basic Usage

```python
from felimination.rfe import PermutationImportanceRFECV
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=1000,
    n_features=2,
    n_classes=2,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=42,
)

# Add random features at the end, so the first 2 features are
# relevant and the remaining 8 features are random
X_with_rand = np.hstack((X, np.random.random(size=(X.shape[0], 8))))

selector = PermutationImportanceRFECV(LogisticRegression())

selector.fit(X_with_rand, y)

selector.support_
# array([False,  True, False, False, False, False, False, False, False, False])

selector.ranking_
# array([ 2,  1,  9, 10,  4,  5,  6,  3,  7,  8])
```