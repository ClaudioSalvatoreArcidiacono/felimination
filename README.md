# felimination

This library contains some useful scikit-learn compatible classes for feature selection.

## [Check out documentation here](https://claudiosalvatorearcidiacono.github.io/felimination/)

## Features

- [Recursive Feature Elimination with Cross Validation using Permutation Importance](reference/RFE.md#felimination.rfe.PermutationImportanceRFECV)

## Requirements

- Python 3.7+
- NumPy
- Scikit-learn
- Pandas

## Installation

In a terminal shell run the following command
```
pip install felimination
```

## Usage

```python

from felimination.rfe import PermutationImportanceRFECV
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
import numpy as np


X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=6,
    n_redundant=10,
    n_clusters_per_class=1,
    random_state=42,
)

# Add random features at the end, so the first 2 features are
# relevant and the remaining 8 features are random
X_with_rand = np.hstack((X, np.random.random(size=(X.shape[0], 8))))

selector = PermutationImportanceRFECV(LogisticRegression(), step=0.3)

selector.fit(X_with_rand, y)

selector.support_
# array([False, False, False, False, False, False, False, False, False,
#        False, False, False, False, True, False, False, False, False,
#        False, False, False, False, False, False, False, False, False,
#        False])

selector.ranking_
# array([10,  3,  9, 10,  8,  9,  7,  7, 10,  6,  8,  2,  8,  1,  9, 10, 10,
#         4,  5,  6,  9, 10,  9,  8, 10,  9,  7, 10])

selector.plot()
```
![example of plot](./docs/assets/example_plot.png)

It looks like `3` is a good number of features, we can set the number of features to select to 3 without need of retraining

```python
selector.set_n_features_to_select(3)
selector.support_
# array([False,  True, False, False, False, False, False, False, False,
#        False, False,  True, False,  True, False, False, False, False,
#        False, False, False, False, False, False, False, False, False,
#        False])
```

## License

This project is licensed under the BSD 3-Clause License - see the LICENSE.md file for details

## Acknowledgments

- [scikit-learn](https://scikit-learn.org/)
