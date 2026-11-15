"""Leakage audit: can the supposedly bureau-free feature sets tell who has a bureau file?

If set A (or B) predicted THIN_FILE almost perfectly, some feature would be acting as a
proxy for bureau data, and the A/B vs. C comparison would be contaminated. Two checks:

1. A LightGBM classifier predicts THIN_FILE from each set (3-fold CV AUC). Moderate AUC is
   expected from genuine demographic differences; near 1.0 would signal a proxy.
2. Single-feature AUC for THIN_FILE: any one feature above 0.80 is flagged for review.

Also lists each feature's univariate AUC for TARGET, to spot suspiciously strong
(possibly post-outcome) features.

    python -m thinfile.audit
"""
from __future__ import annotations

import argparse
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

from thinfile.config import get_logger, load_config

log = get_logger("audit")


def _auc(y, x) -> float:
    m = x.notna()
    if m.sum() < 1000 or x[m].nunique() < 2 or y[m].nunique() < 2:
        return np.nan
    a = roc_auc_score(y[m], x[m])
    return float(max(a, 1 - a))


def audit(cfg: dict, sample: int = 80000) -> dict:
    proc = cfg["paths"]["processed_dir"]
    feats = pd.read_parquet(proc / "features.parquet")
    sets = json.loads((proc / "feature_sets.json").read_text())
    s = feats.sample(min(sample, len(feats)), random_state=cfg["seed"])
    out = {"thin_file_predictability": {}, "flagged_proxies": {}}
    for name in ("A", "B"):
        X = s[sets[name]].copy()
        for c in X.columns:
            if X[c].dtype.kind not in "fiub":
                X[c] = X[c].astype("category")
        m = lgb.LGBMClassifier(n_estimators=200, verbose=-1, random_state=cfg["seed"])
        p = cross_val_predict(m, X, s["THIN_FILE"], cv=3, method="predict_proba")[:, 1]
        auc = float(roc_auc_score(s["THIN_FILE"], p))
        single = {c: _auc(s["THIN_FILE"], s[c]) for c in sets[name] if s[c].dtype.kind in "fiub"}
        flagged = {c: round(v, 3) for c, v in single.items() if v > 0.80}
        out["thin_file_predictability"][name] = round(auc, 4)
        out["flagged_proxies"][name] = flagged
        log.info("Set %s predicts THIN_FILE with AUC %.3f; single features > 0.80: %s", name, auc, flagged or "none")
    num = [c for c in sets["C"] if feats[c].dtype.kind in "fiub"]
    uni = pd.Series({c: _auc(feats["TARGET"], feats[c]) for c in num}).dropna().sort_values(ascending=False)
    out["target_univariate_top10"] = uni.head(10).round(4).to_dict()
    out["target_univariate_max"] = round(float(uni.max()), 4)
    rep = cfg["paths"]["reports_dir"]
    rep.mkdir(parents=True, exist_ok=True)
    (rep / "leakage_audit.json").write_text(json.dumps(out, indent=1))
    log.info("Strongest single feature for TARGET: %s (AUC %.3f)", uni.index[0], uni.iloc[0])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config")
    audit(load_config(ap.parse_args().config))


if __name__ == "__main__":
    main()
