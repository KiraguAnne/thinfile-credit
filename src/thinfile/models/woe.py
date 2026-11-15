"""Weight-of-Evidence binning and logistic-regression scorecard (the baseline).

Implemented from scratch (~200 lines) rather than with a binning library so every step a
credit-risk reviewer would ask about is visible: quantile pre-binning, minimum bin size,
monotonic WoE, Information Value filtering, correlation pruning, sign checks, and the
points scaling (PDO).

WoE convention: WoE = ln(%goods / %bads). Higher WoE = lower risk.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.linear_model import LogisticRegression

MISSING, OTHER = "__MISSING__", "__OTHER__"


@dataclass
class FeatureBins:
    name: str
    numeric: bool
    edges: list[float] = field(default_factory=list)      # numeric: inner cut points
    categories: dict[str, int] = field(default_factory=dict)  # categorical: level -> bin id
    woe: dict = field(default_factory=dict)               # bin id (or MISSING/OTHER) -> WoE
    labels: dict = field(default_factory=dict)            # bin id -> readable label
    counts: dict = field(default_factory=dict)            # bin id -> (n, bad rate)
    iv: float = 0.0

    def bin_ids(self, x: pd.Series) -> np.ndarray:
        if self.numeric:
            v = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
            ids = np.searchsorted(np.asarray(self.edges), v, side="right").astype(object)
            ids[np.isnan(v)] = MISSING
            return ids
        s = x.astype("object").where(x.notna(), MISSING)
        return np.array([self.categories.get(v, MISSING if v == MISSING else OTHER) for v in s], dtype=object)

    def transform(self, x: pd.Series) -> np.ndarray:
        default = 0.0
        return np.array([self.woe.get(b, self.woe.get(OTHER, default)) for b in self.bin_ids(x)], dtype=float)


def _woe_table(ids: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    t = pd.DataFrame({"bin": ids, "y": y}).groupby("bin", sort=False)["y"].agg(["count", "sum"])
    t.columns = ["n", "bad"]
    t["good"] = t["n"] - t["bad"]
    G, B = t["good"].sum(), t["bad"].sum()
    pg, pb = (t["good"] + 0.5) / (G + 0.5), (t["bad"] + 0.5) / (B + 0.5)  # +0.5 smoothing avoids log(0)
    t["woe"] = np.log(pg / pb)
    t["iv"] = (pg - pb) * t["woe"]
    return t


def _monotone_numeric_bins(x: np.ndarray, y: np.ndarray, max_bins: int, min_frac: float) -> list[float]:
    """Quantile pre-bins, then merge adjacent bins until each is big enough and bad rates are monotone."""
    qs = np.unique(np.nanquantile(x, np.linspace(0, 1, max_bins + 1)[1:-1]))
    edges = list(qs)
    n_min = max(int(min_frac * len(x)), 1)

    def stats(edges_):
        ids = np.searchsorted(np.asarray(edges_), x, side="right")
        k = len(edges_) + 1
        n = np.bincount(ids, minlength=k)
        bad = np.bincount(ids, weights=y, minlength=k)
        return n, bad / np.maximum(n, 1)

    while edges:
        n, rate = stats(edges)
        small = np.where(n < n_min)[0]
        if len(small):  # merge the smallest bin into its neighbour
            i = small[np.argmin(n[small])]
            del edges[min(i, len(edges) - 1) if i > 0 else 0]
            continue
        direction = np.sign(np.corrcoef(np.arange(len(rate)), rate)[0, 1]) if len(rate) > 2 else 0
        diffs = np.diff(rate) * (direction or 1)
        bad_steps = np.where(diffs < 0)[0]
        if not len(bad_steps):
            break
        # merge the pair with the smallest violation
        j = bad_steps[np.argmin(np.abs(diffs[bad_steps]))]
        del edges[j]
    return edges


def fit_feature_bins(x: pd.Series, y: np.ndarray, max_bins: int, min_frac: float) -> FeatureBins:
    numeric = is_numeric_dtype(x)
    fb = FeatureBins(name=x.name, numeric=bool(numeric))
    if fb.numeric:
        v = x.to_numpy(dtype=float)
        mask = ~np.isnan(v)
        fb.edges = _monotone_numeric_bins(v[mask], y[mask], max_bins, min_frac) if mask.sum() else []
        e = [-np.inf] + fb.edges + [np.inf]
        fb.labels = {i: f"[{e[i]:.4g}, {e[i + 1]:.4g})" for i in range(len(e) - 1)}
    else:
        s = x.astype("object").where(x.notna(), MISSING)
        freq = s.value_counts(normalize=True)
        keep = [c for c in freq.index if freq[c] >= min_frac and c != MISSING]
        fb.categories = {c: i for i, c in enumerate(keep)}
        fb.labels = {i: c for c, i in fb.categories.items()}
    fb.labels[MISSING], fb.labels[OTHER] = "missing", "other"
    t = _woe_table(fb.bin_ids(x), y)
    # Tiny missing/other bins get a neutral WoE rather than a noisy one.
    for b, row in t.iterrows():
        fb.woe[b] = 0.0 if (b in (MISSING, OTHER) and row["n"] < min_frac * len(x) / 5) else float(row["woe"])
        fb.counts[b] = (int(row["n"]), float(row["bad"] / row["n"]))
    fb.iv = float(t["iv"].sum())
    return fb


class WoEScorecard:
    """WoE-binned logistic regression, scaled to points."""

    model_type = "scorecard"

    def __init__(self, max_bins=8, min_bin_frac=0.05, min_iv=0.02, max_corr=0.75, C=0.1,
                 base_score=600, base_odds=50, pdo=20, seed=42):
        self.params = dict(max_bins=max_bins, min_bin_frac=min_bin_frac, min_iv=min_iv, max_corr=max_corr, C=C)
        self.base_score, self.base_odds, self.pdo, self.seed = base_score, base_odds, pdo, seed

    # -------------------------------------------------------------------------- fitting
    def fit(self, X: pd.DataFrame, y: pd.Series, **_) -> WoEScorecard:
        y = np.asarray(y, dtype=float)
        p = self.params
        self.bins_ = {c: fit_feature_bins(X[c], y, p["max_bins"], p["min_bin_frac"]) for c in X.columns}
        self.iv_ = pd.Series({c: b.iv for c, b in self.bins_.items()}).sort_values(ascending=False)
        cand = [c for c in self.iv_.index if self.iv_[c] >= p["min_iv"]]
        W = pd.DataFrame({c: self.bins_[c].transform(X[c]) for c in cand})
        # Greedy correlation pruning in IV order: keep the stronger of any highly correlated pair.
        corr = W.corr().abs()
        selected: list[str] = []
        for c in cand:
            if all(corr.loc[c, s] < p["max_corr"] for s in selected):
                selected.append(c)
        # Every coefficient on WoE must be negative (higher WoE = safer = lower PD); a positive
        # one means collinearity is distorting the fit, so drop that feature and refit.
        while True:
            lr = LogisticRegression(C=p["C"], max_iter=2000, random_state=self.seed)
            lr.fit(W[selected], y)
            coef = pd.Series(lr.coef_[0], index=selected)
            wrong = coef[coef > 0]
            if wrong.empty:
                break
            selected.remove(wrong.idxmax())
        self.features_ = selected
        self.lr_ = lr
        self.coef_ = coef
        self._build_points()
        return self

    def _build_points(self):
        factor = self.pdo / np.log(2)
        offset = self.base_score - factor * np.log(self.base_odds)
        n = len(self.features_)
        b0 = self.lr_.intercept_[0]
        self.points_ = {}
        for c in self.features_:
            b = self.coef_[c]
            # log-odds(good) = -(b0 + sum b_i * woe_i); spread intercept and offset evenly over features
            self.points_[c] = {k: offset / n - factor * (b * w + b0 / n) for k, w in self.bins_[c].woe.items()}
        self.factor_, self.offset_ = factor, offset

    # ------------------------------------------------------------------------ inference
    def woe_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({c: self.bins_[c].transform(X[c]) for c in self.features_}, index=X.index)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.lr_.predict_proba(self.woe_frame(X))[:, 1]

    def score(self, X: pd.DataFrame) -> np.ndarray:
        pd_ = np.clip(self.predict_proba(X), 1e-9, 1 - 1e-9)
        return self.offset_ + self.factor_ * np.log((1 - pd_) / pd_)

    def explain(self, X: pd.DataFrame, k: int = 3) -> list[list[dict]]:
        """Reason codes: features where the applicant lost the most points vs. the best attainable bin."""
        out = []
        for i in range(len(X)):
            row = X.iloc[[i]]
            lost = []
            for c in self.features_:
                b = self.bins_[c].bin_ids(row[c])[0]
                pts = self.points_[c].get(b, self.points_[c].get(OTHER, 0.0))
                lost.append((c, max(self.points_[c].values()) - pts, row[c].iloc[0]))
            lost.sort(key=lambda t: -t[1])
            out.append([{"feature": c, "impact": round(float(v), 2), "value": _jsonable(val)}
                        for c, v, val in lost[:k] if v > 0])
        return out

    def scorecard_table(self) -> pd.DataFrame:
        rows = []
        for c in self.features_:
            fb = self.bins_[c]
            for b, pts in self.points_[c].items():
                n, br = fb.counts.get(b, (0, np.nan))
                rows.append({"feature": c, "bin": fb.labels.get(b, str(b)), "n": n, "bad_rate": br,
                             "woe": fb.woe[b], "points": round(pts, 1), "iv": fb.iv})
        return pd.DataFrame(rows)


def _jsonable(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (np.integer, np.floating)):
        return v.item()
    return v
