"""LightGBM classifier wrapper (the primary model).

Why LightGBM: native handling of missing values (essential when B/C features are absent
for thin-file applicants - "no history" is itself informative), native categoricals,
monotone-free non-linear interactions, and fast exact TreeSHAP for per-applicant reasons.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


class LGBMModel:
    model_type = "lgbm"

    def __init__(self, seed: int = 42, early_stopping_rounds: int = 200, **params):
        self.seed = seed
        self.early_stopping_rounds = early_stopping_rounds
        self.params = {"objective": "binary", "random_state": seed, "n_jobs": -1, "verbose": -1, **params}

    def _prep(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X[self.features_].copy()
        for c, levels in self.categories_.items():
            X[c] = pd.Categorical(X[c].astype("object"), categories=levels)
        return X

    def fit(self, X: pd.DataFrame, y, eval_set: tuple | None = None) -> LGBMModel:
        self.features_ = list(X.columns)
        self.categories_ = {c: sorted(X[c].dropna().astype(str).unique().tolist())
                            for c in X.columns if not is_numeric_dtype(X[c])}
        self.model_ = lgb.LGBMClassifier(**self.params)
        kw = {}
        if eval_set is not None:
            Xv, yv = eval_set
            kw = dict(eval_X=(self._prep(Xv),), eval_y=(yv,), eval_metric="auc",
                      callbacks=[lgb.early_stopping(self.early_stopping_rounds, verbose=False)])
        self.model_.fit(self._prep(X), y, **kw)
        self.best_iteration_ = self.model_.best_iteration_ or self.params.get("n_estimators")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(self._prep(X))[:, 1]

    def contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Exact TreeSHAP values in log-odds space (last column is the base value)."""
        c = self.model_.booster_.predict(self._prep(X), pred_contrib=True)
        return pd.DataFrame(c, columns=self.features_ + ["__base__"], index=X.index)

    def explain(self, X: pd.DataFrame, k: int = 3) -> list[list[dict]]:
        """Top-k features pushing this applicant's default risk *up*."""
        contrib = self.contributions(X).drop(columns="__base__")
        out = []
        for i in range(len(X)):
            row = contrib.iloc[i].sort_values(ascending=False)
            out.append([{"feature": f, "impact": round(float(v), 4), "value": _jsonable(X.iloc[i][f])}
                        for f, v in row.head(k).items() if v > 0])
        return out

    @property
    def feature_importance(self) -> pd.Series:
        b = self.model_.booster_
        return pd.Series(b.feature_importance("gain"), index=b.feature_name()).sort_values(ascending=False)


def _jsonable(v):
    if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA:
        return None
    if isinstance(v, (np.integer, np.floating)):
        return v.item()
    return v
