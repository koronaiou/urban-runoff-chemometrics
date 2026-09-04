"""
preprocessing.py
================
Fold-safe implementation of the preprocessing recipe used for the PCA and
PLS-DA in the chemometrics chapter, so that the machine-learning case study
operates on an identically prepared matrix.

The recipe (as described in the chemometrics section)
-----------------------------------------------------
1. Drop compounds detected in fewer than four samples.
2. Replace zeros with half the minimum detected value of that compound.
3. log10 transform.
4. Max-scale each compound to [0, 1].

Why this is reimplemented rather than applied once to the whole matrix
---------------------------------------------------------------------
Steps 1, 2 and 4 all estimate a quantity *from the data*: which compounds pass
the occurrence filter, the per-compound imputation constant, and the per-
compound maximum. Computing these once over all 40 samples is perfectly
appropriate for a descriptive ordination, where no held-out set exists.

Inside cross-validation it is not appropriate: the held-out samples would have
contributed to the filter, the imputation constants and the scaling, so the
model would have been prepared using information from the samples it is then
scored on. The effect is usually small, but it is exactly the class of problem
the validation design exists to exclude, and it is cheap to avoid.

`ChapterPreprocessor` therefore learns all three quantities in `fit()` from the
training partition alone and applies them unchanged in `transform()`. Held-out
values may fall outside [0, 1] after scaling; this is correct and expected, and
they are not clipped, because clipping would discard genuine information about
samples more extreme than anything seen in training.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class ChapterPreprocessor(BaseEstimator, TransformerMixin):
    """
    Occurrence filter -> half-minimum imputation -> log10 -> max-scaling.

    Parameters
    ----------
    min_detect_frac : float
        Minimum fraction of *training* samples in which a compound must be
        detected to be retained. The default of 0.10 reproduces the chapter's
        "fewer than four samples" rule at n = 40.
    """

    def __init__(self, min_detect_frac=0.10):
        self.min_detect_frac = min_detect_frac

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        detected = X > 0
        n_required = max(1, int(np.ceil(self.min_detect_frac * n)))
        self.keep_ = detected.sum(axis=0) >= n_required

        Xk = X[:, self.keep_]
        dk = detected[:, self.keep_]
        # half the minimum detected value, per compound, from training only
        self.half_min_ = np.array([
            0.5 * Xk[dk[:, j], j].min() if dk[:, j].any() else 1.0
            for j in range(Xk.shape[1])
        ])

        L = np.log10(np.where(dk, Xk, self.half_min_))
        self.max_ = L.max(axis=0)
        self.max_[self.max_ == 0] = 1.0
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        Xk = X[:, self.keep_]
        L = np.log10(np.where(Xk > 0, Xk, self.half_min_))
        return L / self.max_

    def get_kept_indices(self):
        """Indices, in the original 175-compound space, of retained compounds."""
        return np.where(self.keep_)[0]
