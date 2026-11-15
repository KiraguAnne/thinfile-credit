"""Stage 5 - prediction.

`Scorer` is the single inference entry point, shared by the batch CLI and the API:
raw records in -> features (same SQL as training) -> PD, score band, decision, reasons.

    # batch-score applicants from the processed feature table
    python -m thinfile.predict --ids 100002 100003 --model lgbm_C
    python -m thinfile.predict --input applicants.parquet --output scored.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from thinfile.config import get_logger, load_config
from thinfile.explain import describe_feature
from thinfile.features.build import features_for_request
from thinfile.metrics import decide

log = get_logger("predict")


class Scorer:
    def __init__(self, bundle_path: str | Path):
        self.path = Path(bundle_path)
        self.bundle = joblib.load(self.path)
        self.model = self.bundle["model"]
        self.features = self.bundle["features"]
        self.threshold = self.bundle["threshold"]
        self.review_band = self.bundle["review_band"]

    @property
    def info(self) -> dict:
        b = self.bundle
        return {"model_type": b["model_type"], "feature_set": b["feature_set"], "version": b["version"],
                "trained_at": b["trained_at"], "n_features": len(self.features),
                "threshold": round(self.threshold, 4), "review_band": self.review_band,
                "metrics": b["metrics"]}

    def score_features(self, feats: pd.DataFrame, k_reasons: int = 3) -> list[dict]:
        X = feats.reindex(columns=self.features)
        pd_ = self.model.predict_proba(X)
        reasons = self.model.explain(X, k_reasons)
        out = []
        for i in range(len(X)):
            p = float(pd_[i])
            odds = np.log((1 - min(max(p, 1e-9), 1 - 1e-9)) / min(max(p, 1e-9), 1 - 1e-9))
            out.append({
                "SK_ID_CURR": int(feats["SK_ID_CURR"].iloc[i]) if "SK_ID_CURR" in feats else None,
                "probability_of_default": round(p, 5),
                "credit_score": int(round(600 - 20 / np.log(2) * np.log(50) + 20 / np.log(2) * odds)),
                "decision": decide(p, self.threshold, self.review_band),
                "thin_file": bool(feats["THIN_FILE"].iloc[i]) if "THIN_FILE" in feats else None,
                "top_reasons": [{**r, "description": describe_feature(r["feature"])} for r in reasons[i]],
            })
        return out

    def score_raw(self, tables: dict[str, pd.DataFrame | None], k_reasons: int = 3) -> list[dict]:
        feats = features_for_request(tables, self.bundle["table_schemas"], self.bundle["windows_months"])
        return self.score_features(feats, k_reasons)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config")
    ap.add_argument("--model", help="artifact name, e.g. lgbm_C (default: serving model in config)")
    ap.add_argument("--ids", nargs="*", type=int, help="SK_ID_CURR values from the feature table")
    ap.add_argument("--input", help="parquet/csv of pre-built features")
    ap.add_argument("--output", help="where to write scored rows (parquet/csv)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    name = a.model or f"{cfg['serving']['model_type']}_{cfg['serving']['model_set']}"
    scorer = Scorer(cfg["paths"]["artifacts_dir"] / "models" / f"{name}.joblib")
    if a.input:
        feats = pd.read_parquet(a.input) if a.input.endswith(".parquet") else pd.read_csv(a.input)
    else:
        feats = pd.read_parquet(cfg["paths"]["processed_dir"] / "features.parquet")
        if a.ids:
            feats = feats[feats.SK_ID_CURR.isin(a.ids)]
    res = pd.DataFrame(scorer.score_features(feats))
    if a.output:
        (res.to_parquet if a.output.endswith(".parquet") else res.to_csv)(a.output, index=False)
        log.info("Scored %d applicants -> %s", len(res), a.output)
    else:
        pd.set_option("display.max_colwidth", 120)
        print(res.to_string(index=False))


if __name__ == "__main__":
    main()
