"""Stage 3 - training.

For every (feature set, model) pair: stratified 5-fold CV on the training split for honest
out-of-fold (OOF) estimates, then a refit on the full training split and one prediction on
the untouched test split. Artifacts, predictions and MLflow runs are written per pair.

    python -m thinfile.train                     # all sets x both models
    python -m thinfile.train --sets C --models lgbm
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from thinfile import __version__
from thinfile.config import get_logger, load_config
from thinfile.metrics import cost_optimal_threshold, score_metrics
from thinfile.models.lgbm import LGBMModel
from thinfile.models.woe import WoEScorecard

log = get_logger("train")


def make_model(kind: str, cfg: dict):
    if kind == "lgbm":
        return LGBMModel(seed=cfg["seed"], **cfg["lgbm"])
    if kind == "scorecard":
        return WoEScorecard(seed=cfg["seed"], **cfg["scorecard"])
    raise ValueError(kind)


def load_data(cfg: dict) -> tuple[pd.DataFrame, dict]:
    proc = cfg["paths"]["processed_dir"]
    feats = pd.read_parquet(proc / "features.parquet")
    sets = json.loads((proc / "feature_sets.json").read_text())
    return feats, sets


def split_ids(feats: pd.DataFrame, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Hold out a test split, stratified on default x thin-file so both are represented evenly."""
    if not feats["SK_ID_CURR"].is_monotonic_increasing:
        raise ValueError("features must be sorted by SK_ID_CURR so the split is reproducible")
    strata = feats["TARGET"].astype(str) + "_" + feats["THIN_FILE"].astype(str)
    tr, te = train_test_split(feats.index.to_numpy(), test_size=cfg["split"]["test_size"],
                              stratify=strata, random_state=cfg["seed"])
    return np.sort(tr), np.sort(te)


def fit_one(kind, X, y, cfg, X_val=None, y_val=None):
    m = make_model(kind, cfg)
    if kind == "lgbm" and X_val is not None:
        m.fit(X, y, eval_set=(X_val, y_val))
    else:
        m.fit(X, y)
    return m


def run_pair(kind: str, fset: str, feats: pd.DataFrame, cols: list[str], tr: np.ndarray, te: np.ndarray,
             cfg: dict) -> dict:
    t0 = time.time()
    Xtr, ytr = feats.loc[tr, cols], feats.loc[tr, "TARGET"].to_numpy()
    Xte, yte = feats.loc[te, cols], feats.loc[te, "TARGET"].to_numpy()
    strata = feats.loc[tr, "TARGET"].astype(str) + "_" + feats.loc[tr, "THIN_FILE"].astype(str)
    skf = StratifiedKFold(cfg["split"]["n_folds"], shuffle=True, random_state=cfg["seed"])
    oof = np.zeros(len(tr))
    fold_ids = np.zeros(len(tr), dtype=int)
    fold_metrics, best_iters = [], []
    for k, (a, b) in enumerate(skf.split(Xtr, strata)):
        m = fit_one(kind, Xtr.iloc[a], ytr[a], cfg, Xtr.iloc[b], ytr[b])
        oof[b] = m.predict_proba(Xtr.iloc[b])
        fold_ids[b] = k
        fold_metrics.append(score_metrics(ytr[b], oof[b]))
        if kind == "lgbm":
            best_iters.append(m.best_iteration_)
        log.info("  [%s/%s] fold %d AUC %.4f", kind, fset, k, fold_metrics[-1]["auc"])

    # Refit on all training data. LightGBM uses the average early-stopped tree count (+10%
    # because the refit sees 25% more data than each CV fold did).
    if kind == "lgbm":
        n_trees = int(np.mean(best_iters) * 1.1)
        final = make_model(kind, {**cfg, "lgbm": {**cfg["lgbm"], "n_estimators": n_trees}})
        final.fit(Xtr, ytr)
    else:
        final = fit_one(kind, Xtr, ytr, cfg)
    p_test = final.predict_proba(Xte)

    aucs = [m["auc"] for m in fold_metrics]
    res = {
        "model": kind, "feature_set": fset, "n_features_in": len(cols),
        "n_features_used": len(final.features_),
        "cv_auc_mean": float(np.mean(aucs)), "cv_auc_std": float(np.std(aucs)),
        "oof": score_metrics(ytr, oof), "test": score_metrics(yte, p_test),
        "best_iterations": best_iters, "train_seconds": round(time.time() - t0, 1),
    }
    threshold = cost_optimal_threshold(ytr, oof, cfg["decision"]["cost_fn"], cfg["decision"]["cost_fp"])
    res["decision"] = threshold

    art = cfg["paths"]["artifacts_dir"] / "models"
    art.mkdir(parents=True, exist_ok=True)
    schemas_path = cfg["paths"]["artifacts_dir"] / "table_schemas.json"
    bundle = {
        "model": final, "model_type": kind, "feature_set": fset, "features": cols,
        "threshold": threshold["threshold"], "review_band": cfg["decision"]["review_band"],
        "windows_months": cfg["windows_months"], "metrics": {k: res[k] for k in ("cv_auc_mean", "cv_auc_std", "test")},
        "table_schemas": json.loads(schemas_path.read_text()) if schemas_path.exists() else {},
        "version": __version__, "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    joblib.dump(bundle, art / f"{kind}_{fset}.joblib", compress=3)
    if kind == "scorecard":
        final.scorecard_table().to_csv(art / f"scorecard_{fset}_table.csv", index=False)
        final.iv_.rename("iv").to_csv(art / f"scorecard_{fset}_iv.csv")

    log.info("[%s/%s] CV AUC %.4f ± %.4f | test AUC %.4f | %d features used | %.0fs", kind, fset,
             res["cv_auc_mean"], res["cv_auc_std"], res["test"]["auc"], res["n_features_used"], res["train_seconds"])
    return res, oof, fold_ids, p_test


def _mlflow_log(cfg, res):
    try:
        import mlflow
    except ImportError:
        return
    mlflow.set_tracking_uri(str(cfg["paths"]["mlflow_uri"]))
    mlflow.set_experiment("thinfile-credit")
    with mlflow.start_run(run_name=f"{res['model']}_{res['feature_set']}"):
        mlflow.log_params({"model": res["model"], "feature_set": res["feature_set"],
                           "n_features_used": res["n_features_used"], "seed": cfg["seed"]})
        mlflow.log_params({f"{res['model']}.{k}": v for k, v in cfg[res["model"]].items()})
        mlflow.log_metrics({"cv_auc_mean": res["cv_auc_mean"], "cv_auc_std": res["cv_auc_std"],
                            **{f"test_{k}": v for k, v in res["test"].items() if isinstance(v, float)}})
        mlflow.log_artifact(str(cfg["paths"]["artifacts_dir"] / "models" / f"{res['model']}_{res['feature_set']}.joblib"))


def train(cfg: dict, sets: list[str] | None = None, models: list[str] | None = None, use_mlflow: bool = True):
    feats, set_cols = load_data(cfg)
    tr, te = split_ids(feats, cfg)
    pred_dir = cfg["paths"]["artifacts_dir"] / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    oof_path, test_path = pred_dir / "oof.parquet", pred_dir / "test.parquet"
    meta = ["SK_ID_CURR", "TARGET", "THIN_FILE", "AGE_BAND", "CODE_GENDER", "NAME_EDUCATION_TYPE"]
    oof_df = pd.read_parquet(oof_path) if oof_path.exists() else feats.loc[tr, meta].reset_index(drop=True)
    test_df = pd.read_parquet(test_path) if test_path.exists() else feats.loc[te, meta].reset_index(drop=True)
    results_path = cfg["paths"]["artifacts_dir"] / "train_results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}

    pairs = [(f, k) for f in (sets or cfg["feature_sets"]) for k in (models or ["scorecard", "lgbm"])]
    if not sets:  # default run also trains the LightGBM sensitivity sets
        pairs += [(f, "lgbm") for f in cfg.get("sensitivity_sets", [])]
    for fset, kind in pairs:
        res, oof, folds, p_test = run_pair(kind, fset, feats, set_cols[fset], tr, te, cfg)
        oof_df[f"{kind}_{fset}"] = oof
        oof_df["fold"] = folds
        test_df[f"{kind}_{fset}"] = p_test
        results[f"{kind}_{fset}"] = res
        oof_df.to_parquet(oof_path, index=False)
        test_df.to_parquet(test_path, index=False)
        results_path.write_text(json.dumps(results, indent=1))
        if use_mlflow:
            try:
                _mlflow_log(cfg, res)
            except Exception as e:  # tracking must never break a training run
                log.warning("MLflow logging skipped: %s", e)
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config")
    ap.add_argument("--sets", nargs="*")
    ap.add_argument("--models", nargs="*", choices=["scorecard", "lgbm"])
    ap.add_argument("--no-mlflow", action="store_true")
    a = ap.parse_args()
    train(load_config(a.config), a.sets, a.models, not a.no_mlflow)


if __name__ == "__main__":
    main()
