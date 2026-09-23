"""Fast multilingual text classifier: character + word TF-IDF and rule features -> 12 logistic heads.

Character n-grams on accent-folded text are largely language-agnostic ("menisc", "rupt", "fractu"),
so this model is a strong, CPU-only baseline that trains in seconds and runs offline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion

from .constants import LABELS
from .text import fold_text


class TfidfLabeler:
    def __init__(self, C: float = 4.0, max_features: int = 200_000, use_rule_features: bool = True):
        self.C = C
        self.max_features = max_features
        self.use_rule_features = use_rule_features
        self.vectorizer: FeatureUnion | None = None
        self.heads: dict[str, LogisticRegression | float] = {}

    def _features(self, texts: pd.Series, rule_feats: pd.DataFrame | None, fit: bool) -> sp.csr_matrix:
        folded = [fold_text(t) for t in texts]
        if fit:
            self.vectorizer = FeatureUnion([
                ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                                         sublinear_tf=True, max_features=self.max_features)),
                ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2,
                                         sublinear_tf=True, max_features=self.max_features // 2)),
            ])
            X = self.vectorizer.fit_transform(folded)
        else:
            X = self.vectorizer.transform(folded)
        if self.use_rule_features:
            if rule_feats is None:
                raise ValueError("rule_feats is required when use_rule_features=True")
            # Hit counts are small integers; clip and scale to be comparable with TF-IDF weights.
            R = np.clip(rule_feats.to_numpy(dtype=np.float32), 0, 3) / 3.0
            X = sp.hstack([X, sp.csr_matrix(R)], format="csr")
        return X

    def fit(self, texts: pd.Series, Y: np.ndarray, rule_feats: pd.DataFrame | None = None) -> "TfidfLabeler":
        """Fit one head per label on the rows where that label is known (Y may contain NaN)."""
        X = self._features(texts, rule_feats, fit=True)
        for j, label in enumerate(LABELS):
            known = ~np.isnan(Y[:, j])
            y = Y[known, j]
            if len(np.unique(y)) < 2:
                # Can't learn a ranking from one class; predict the observed prevalence.
                self.heads[label] = float(y.mean()) if len(y) else 0.0
                continue
            clf = LogisticRegression(C=self.C, class_weight="balanced", solver="liblinear", max_iter=2000)
            self.heads[label] = clf.fit(X[known], y)
        return self

    def predict_proba(self, texts: pd.Series, rule_feats: pd.DataFrame | None = None) -> np.ndarray:
        X = self._features(texts, rule_feats, fit=False)
        P = np.zeros((X.shape[0], len(LABELS)))
        for j, label in enumerate(LABELS):
            head = self.heads[label]
            P[:, j] = head if isinstance(head, float) else head.predict_proba(X)[:, 1]
        return P
