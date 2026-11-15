import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from thinfile.metrics import cost_optimal_threshold, decide
from thinfile.models.lgbm import LGBMModel
from thinfile.models.woe import MISSING, WoEScorecard, fit_feature_bins


def _data(n=5000, seed=0):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    cat = rng.choice(["low", "mid", "high"], n)
    logit = -2 + 1.2 * x1 - 0.5 * x2 + np.select([cat == "high"], [0.8], 0)
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    x1[rng.random(n) < 0.1] = np.nan
    return pd.DataFrame({"x1": x1, "x2": x2, "noise": rng.normal(size=n), "cat": cat}), y


def test_woe_bins_are_monotone():
    X, y = _data()
    fb = fit_feature_bins(X["x1"], y, max_bins=10, min_frac=0.05)
    woes = [fb.woe[i] for i in range(len(fb.edges) + 1)]
    d = np.diff(woes)
    assert (d <= 1e-9).all() or (d >= -1e-9).all()
    assert MISSING in fb.woe  # missing values get their own bin


def test_scorecard_learns_signal_and_drops_noise():
    X, y = _data()
    sc = WoEScorecard(min_iv=0.02).fit(X, y)
    assert "noise" not in sc.features_
    assert (sc.coef_ < 0).all()  # WoE convention: higher WoE = safer
    p = sc.predict_proba(X)
    assert roc_auc_score(y, p) > 0.7
    # higher score = lower risk
    s = sc.score(X)
    assert np.corrcoef(s, p)[0, 1] < -0.9
    reasons = sc.explain(X.head(3))
    assert len(reasons) == 3 and all(len(r) <= 3 for r in reasons)


def test_lgbm_handles_missing_and_categoricals():
    X, y = _data()
    m = LGBMModel(n_estimators=200, learning_rate=0.1, verbose=-1).fit(X, y)
    Xn = X.head(5).copy()
    Xn.loc[:, "x1"] = np.nan
    Xn.loc[Xn.index[0], "cat"] = "never-seen"
    p = m.predict_proba(Xn)
    assert np.isfinite(p).all()
    c = m.contributions(X.head(5))
    # TreeSHAP additivity: contributions + base value = raw log-odds
    raw = np.log(m.predict_proba(X.head(5)) / (1 - m.predict_proba(X.head(5))))
    assert np.allclose(c.sum(axis=1), raw, atol=1e-6)


def test_cost_threshold_and_decision_bands():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 0.6, 4000)
    y = (rng.random(4000) < p).astype(int)
    t = cost_optimal_threshold(y, p, cost_fn=5, cost_fp=1)
    # for calibrated scores the optimum is near cost_fp / (cost_fp + cost_fn) = 1/6
    assert 0.1 < t["threshold"] < 0.25
    assert t["expected_cost"] < t["baseline_cost_approve_all"]
    assert decide(0.01, 0.2, 0.5) == "approve"
    assert decide(0.2, 0.2, 0.5) == "review"
    assert decide(0.5, 0.2, 0.5) == "decline"
