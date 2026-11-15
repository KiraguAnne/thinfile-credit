"""Drift monitoring with the Population Stability Index (PSI), the standard credit-risk check.

Compares the feature and score distributions of a new batch of applicants against the
training reference. Conventional reading: PSI < 0.10 stable, 0.10-0.25 watch, > 0.25 act
(investigate / consider retraining).

    python -m thinfile.monitor --batch new_applicants_features.parquet
    python -m thinfile.monitor --simulate     # demo: thin-file-heavy batch vs. training mix
"""
from __future__ import annotations

import argparse

import joblib
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from thinfile.config import get_logger, load_config

log = get_logger("monitor")


def psi(ref: pd.Series, new: pd.Series, bins: int = 10) -> float:
    eps = 1e-4
    if is_numeric_dtype(ref):
        r = ref.to_numpy(dtype=float)
        edges = np.unique(np.nanquantile(r, np.linspace(0, 1, bins + 1)[1:-1]))

        def dist(x):
            x = x.to_numpy(dtype=float)
            ids = np.where(np.isnan(x), -1, np.searchsorted(edges, x, side="right"))
            cats = np.arange(-1, len(edges) + 1)
            return np.array([(ids == c).mean() for c in cats])
    else:
        cats = pd.concat([ref, new]).astype("object").fillna("__MISSING__").unique()

        def dist(x):
            vc = x.astype("object").fillna("__MISSING__").value_counts(normalize=True)
            return np.array([vc.get(c, 0.0) for c in cats])
    p, q = dist(ref) + eps, dist(new) + eps
    return float(np.sum((q - p) * np.log(q / p)))


def drift_report(ref: pd.DataFrame, new: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = [{"feature": f, "psi": psi(ref[f], new[f])} for f in features if f in new]
    df = pd.DataFrame(rows).sort_values("psi", ascending=False)
    df["status"] = pd.cut(df["psi"], [-np.inf, 0.10, 0.25, np.inf], labels=["stable", "watch", "act"])
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config")
    ap.add_argument("--batch", help="parquet of new applicants' features")
    ap.add_argument("--simulate", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    name = f"{cfg['serving']['model_type']}_{cfg['serving']['model_set']}"
    bundle = joblib.load(cfg["paths"]["artifacts_dir"] / "models" / f"{name}.joblib")
    feats = pd.read_parquet(cfg["paths"]["processed_dir"] / "features.parquet")
    test_ids = set(pd.read_parquet(cfg["paths"]["artifacts_dir"] / "predictions" / "test.parquet").SK_ID_CURR)
    ref = feats[~feats.SK_ID_CURR.isin(test_ids)]
    if a.simulate:
        # A lender expanding into the thin-file segment: batch is 60% thin-file vs ~14% in training.
        pool = feats[feats.SK_ID_CURR.isin(test_ids)]
        thin, thick = pool[pool.THIN_FILE == 1], pool[pool.THIN_FILE == 0]
        new = pd.concat([thin.sample(6000, replace=True, random_state=0), thick.sample(4000, random_state=0)])
    else:
        new = pd.read_parquet(a.batch)
    ref = ref.assign(__score__=bundle["model"].predict_proba(ref[bundle["features"]]))
    new = new.assign(__score__=bundle["model"].predict_proba(new[bundle["features"]]))
    rep = drift_report(ref, new, ["__score__"] + bundle["features"])
    out = cfg["paths"]["reports_dir"] / ("drift_simulated.csv" if a.simulate else "drift_report.csv")
    rep.to_csv(out, index=False)
    log.info("Score PSI %.3f | features: %s | written to %s", rep.set_index("feature").loc["__score__", "psi"],
             rep["status"].value_counts().to_dict(), out.name)
    print(rep.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
