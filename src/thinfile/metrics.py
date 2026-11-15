"""Evaluation metrics and the cost-based decision rule."""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score, roc_curve


def fast_auc(y: np.ndarray, p: np.ndarray) -> float:
    """ROC-AUC via the Mann-Whitney rank formula (ties averaged). ~4x faster than sklearn for bootstraps."""
    r = rankdata(p)
    n1 = y.sum()
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ks_stat(y, p) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def score_metrics(y, p) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    auc = roc_auc_score(y, p)
    return {
        "auc": float(auc),
        "gini": float(2 * auc - 1),
        "ks": ks_stat(y, p),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7))),
        "n": int(len(y)),
        "default_rate": float(y.mean()),
    }


def bootstrap_auc(y, p, n_boot: int = 500, seed: int = 42) -> tuple[float, float]:
    """95% percentile bootstrap interval for ROC-AUC."""
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        vals.append(fast_auc(y[idx], p[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def paired_bootstrap_auc_diff(y, p1, p2, n_boot: int = 500, seed: int = 42) -> dict:
    """AUC(p2) - AUC(p1) with a paired bootstrap CI - is model 2 really better on the same applicants?"""
    y, p1, p2 = map(np.asarray, (y, p1, p2))
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        diffs.append(fast_auc(y[idx], p2[idx]) - fast_auc(y[idx], p1[idx]))
    diffs = np.asarray(diffs)
    return {"diff": float(roc_auc_score(y, p2) - roc_auc_score(y, p1)),
            "ci_low": float(np.percentile(diffs, 2.5)), "ci_high": float(np.percentile(diffs, 97.5)),
            "p_le_0": float((diffs <= 0).mean())}


def cost_optimal_threshold(y, p, cost_fn: float, cost_fp: float) -> dict:
    """Decline when PD >= threshold. Choose the threshold minimising expected cost per applicant.

    FN = approved applicant who defaults (lost principal); FP = declined good applicant (lost margin).
    """
    y, p = np.asarray(y), np.asarray(p)
    grid = np.unique(np.quantile(p, np.linspace(0.01, 0.99, 197)))
    costs = []
    for t in grid:
        decline = p >= t
        fn = np.sum(~decline & (y == 1))
        fp = np.sum(decline & (y == 0))
        costs.append((cost_fn * fn + cost_fp * fp) / len(y))
    i = int(np.argmin(costs))
    t = float(grid[i])
    decline = p >= t
    return {"threshold": t, "expected_cost": float(costs[i]),
            "approval_rate": float(1 - decline.mean()),
            "default_rate_approved": float(y[~decline].mean()),
            "recall_defaulters": float(decline[y == 1].mean()),
            "baseline_cost_approve_all": float(cost_fn * y.mean())}


def decide(p: float, threshold: float, review_band: float) -> str:
    if p >= threshold * (1 + review_band):
        return "decline"
    if p < threshold * (1 - review_band):
        return "approve"
    return "review"
