"""Stage 4 - evaluation.

Reads the out-of-fold and test predictions written by training and produces:
  reports/metrics.json, reports/metrics_table.csv     all models x sets x populations
  reports/alt_data_uplift.csv                         paired-bootstrap AUC gains (A->B, B->C)
  reports/fairness.csv                                subgroup metrics for the serving model
  reports/confusion_matrix.csv                        deployed decision rule: approved/declined x repaid/defaulted
  reports/head_to_head.csv                            behaviour vs bureau history on applicants who have both
  reports/figures/*.png                               charts used in the report

The test split is touched only here. Headline numbers are test-set numbers; OOF numbers are
used only for the decision threshold.

    python -m thinfile.evaluate
"""
from __future__ import annotations

import argparse
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve  # noqa: E402

from thinfile.config import get_logger, load_config  # noqa: E402
from thinfile.explain import describe_feature  # noqa: E402
from thinfile.metrics import bootstrap_auc, paired_bootstrap_auc_diff, score_metrics  # noqa: E402

log = get_logger("evaluate")

# Validated reference palette (light surface). Slots 1-3 pass all-pairs CVD checks.
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SLOT = ["#2a78d6", "#eb6834", "#1baf7a"]
MODEL_COLOR = {"scorecard": SLOT[0], "lgbm": SLOT[1]}
MODEL_NAME = {"scorecard": "WoE scorecard", "lgbm": "LightGBM"}
SET_NAME = {"A": "A: application", "B": "B: + behavioural", "C": "C: + bureau"}
POPS = {"all": "All applicants", "thin": "Thin-file (no bureau record)", "thick": "Has bureau record"}


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
        "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
    })


def _pop_mask(df: pd.DataFrame, pop: str) -> np.ndarray:
    if pop == "thin":
        return (df["THIN_FILE"] == 1).to_numpy()
    if pop == "thick":
        return (df["THIN_FILE"] == 0).to_numpy()
    return np.ones(len(df), dtype=bool)


def metrics_table(test: pd.DataFrame, results: dict, cols: list[str]) -> pd.DataFrame:
    rows = []
    for c in cols:
        kind, fset = c.split("_", 1)
        for pop in POPS:
            m = _pop_mask(test, pop)
            y, p = test.loc[m, "TARGET"].to_numpy(), test.loc[m, c].to_numpy()
            met = score_metrics(y, p)
            lo, hi = bootstrap_auc(y, p)
            rows.append({"model": kind, "feature_set": fset, "population": pop, **met,
                         "auc_ci_low": lo, "auc_ci_high": hi,
                         "cv_auc_mean": results[c]["cv_auc_mean"], "cv_auc_std": results[c]["cv_auc_std"],
                         "n_features_used": results[c]["n_features_used"]})
    return pd.DataFrame(rows)


def uplift_table(test: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for kind in ("scorecard", "lgbm"):
        for a, b in (("A", "B"), ("B", "C"), ("A", "C"), ("B", "B_ext"), ("C_noext", "C"), ("B", "C_noext")):
            ca, cb = f"{kind}_{a}", f"{kind}_{b}"
            if ca not in cols or cb not in cols:
                continue
            for pop in POPS:
                m = _pop_mask(test, pop)
                d = paired_bootstrap_auc_diff(test.loc[m, "TARGET"], test.loc[m, ca], test.loc[m, cb])
                rows.append({"model": kind, "comparison": f"{a}->{b}", "population": pop, **d})
    # Model comparison at fixed feature set
    for fset in ("A", "B", "C"):
        ca, cb = f"scorecard_{fset}", f"lgbm_{fset}"
        if ca in cols and cb in cols:
            for pop in POPS:
                m = _pop_mask(test, pop)
                d = paired_bootstrap_auc_diff(test.loc[m, "TARGET"], test.loc[m, ca], test.loc[m, cb])
                rows.append({"model": "lgbm_vs_scorecard", "comparison": fset, "population": pop, **d})
    return pd.DataFrame(rows)


def fairness_table(test: pd.DataFrame, col: str, threshold: float, groups: list[str]) -> pd.DataFrame:
    """Subgroup view at the deployed threshold.

    approval_rate_good = share of applicants who did NOT default that the model approves
    (equal-opportunity view: are creditworthy people in each group treated alike?).
    """
    rows = []
    approve = test[col] < threshold
    for g in groups:
        for level, idx in test.groupby(g, observed=True).groups.items():
            sub = test.loc[idx]
            if len(sub) < 200:
                continue
            ap = approve.loc[idx]
            y = sub["TARGET"]
            rows.append({
                "attribute": g, "group": level, "n": len(sub), "default_rate": y.mean(),
                "auc": roc_auc_score(y, sub[col]) if y.nunique() == 2 else np.nan,
                "mean_pd": sub[col].mean(), "approval_rate": ap.mean(),
                "approval_rate_good": ap[y == 0].mean(),
                "default_rate_approved": y[ap].mean() if ap.any() else np.nan,
            })
    df = pd.DataFrame(rows)
    df["approval_ratio_vs_best"] = df["approval_rate"] / df.groupby("attribute")["approval_rate"].transform("max")
    return df


def decision_economics(test: pd.DataFrame, cols: list[str], thresholds: dict, band: float, cost_fn: float,
                       cost_fp: float) -> pd.DataFrame:
    """Test-set consequences of each model's own cost-optimal threshold (set on OOF predictions)."""
    rows = []
    for c in cols:
        threshold = thresholds[c]
        for pop in ("all", "thin"):
            d = test[_pop_mask(test, pop)]
            p, y = d[c].to_numpy(), d["TARGET"].to_numpy()
            decline = p >= threshold
            auto_ok, auto_no = p < threshold * (1 - band), p >= threshold * (1 + band)
            rows.append({
                "model": c, "population": pop, "threshold": threshold, "n": len(d), "approval_rate": 1 - decline.mean(),
                "default_rate_all": y.mean(), "default_rate_approved": y[~decline].mean(),
                "defaulters_declined": decline[y == 1].mean(), "goods_declined": decline[y == 0].mean(),
                "expected_cost": (cost_fn * np.sum(~decline & (y == 1)) + cost_fp * np.sum(decline & (y == 0))) / len(y),
                "cost_approve_all": cost_fn * y.mean(),
                "auto_approve_share": auto_ok.mean(), "review_share": 1 - auto_ok.mean() - auto_no.mean(),
                "auto_decline_share": auto_no.mean(),
            })
    df = pd.DataFrame(rows)
    df["cost_saving_vs_approve_all"] = 1 - df["expected_cost"] / df["cost_approve_all"]
    return df


def shap_by_source(art, feats: pd.DataFrame, seed: int, n: int = 15000) -> dict:
    """Share of mean |SHAP| carried by each data source, per population."""
    from thinfile.features.application import BUREAU_DERIVED

    def group(c):
        if c.startswith("EXT_SOURCE"):
            return "External scores"
        if c.startswith(("BUR_", "BB_")) or c in BUREAU_DERIVED:
            return "Bureau history"
        if c.startswith(("INST_", "POS_", "CC_", "PREV_")):
            return "Behavioural"
        return "Application"

    out = {}
    for name in ("lgbm_B", "lgbm_C"):
        p = art / "models" / f"{name}.joblib"
        if not p.exists():
            continue
        b = joblib.load(p)
        for pop in POPS:
            m = (feats["THIN_FILE"] == 1) if pop == "thin" else (feats["THIN_FILE"] == 0) if pop == "thick" \
                else pd.Series(True, index=feats.index)
            X = feats.loc[m, b["features"]]
            X = X.sample(min(n, len(X)), random_state=seed)
            s = b["model"].contributions(X).drop(columns="__base__").abs().mean()
            g = s.groupby(s.index.map(group)).sum()
            out[f"{name}_{pop}"] = (g / g.sum()).round(4).to_dict()
    return out


def head_to_head(test: pd.DataFrame, has_history: np.ndarray) -> pd.DataFrame:
    """Behaviour vs bureau history on the same people: test applicants with a bureau file AND repayment history.

    External scores are excluded from both sides so that records are compared with records.
    """
    m = (test["THIN_FILE"] == 0).to_numpy() & has_history
    y = test.loc[m, "TARGET"]
    rows = []
    for name, col in (("Application form only", "lgbm_A"), ("Form + behaviour", "lgbm_B"),
                      ("Form + bureau history", "lgbm_A_bureau"), ("Form + behaviour + bureau history", "lgbm_C_noext")):
        if col in test:
            lo, hi = bootstrap_auc(y.to_numpy(), test.loc[m, col].to_numpy())
            rows.append({"row": "auc", "comparison": name, "model": col, "n": int(m.sum()),
                         "value": roc_auc_score(y, test.loc[m, col]), "ci_low": lo, "ci_high": hi})
    for label, a, b in (("behaviour minus bureau history", "lgbm_A_bureau", "lgbm_B"),
                        ("bureau history added to behaviour", "lgbm_B", "lgbm_C_noext")):
        if a in test and b in test:
            d = paired_bootstrap_auc_diff(y, test.loc[m, a], test.loc[m, b])
            rows.append({"row": "difference", "comparison": label, "model": f"{b} - {a}", "n": int(m.sum()), **d})
    return pd.DataFrame(rows)


def confusion(test: pd.DataFrame, col: str, threshold: float) -> pd.DataFrame:
    """Confusion matrix of the deployed decision rule (decline if PD >= threshold), counts and per 1,000."""
    y, decline = test["TARGET"].to_numpy(), test[col].to_numpy() >= threshold
    rows = []
    for pop in ("all", "thin"):
        m = _pop_mask(test, pop)
        cm = confusion_matrix(y[m], decline[m].astype(int), labels=[0, 1])
        for i, actual in enumerate(("repaid", "defaulted")):
            for j, dec in enumerate(("approved", "declined")):
                rows.append({"population": pop, "actual": actual, "decision": dec, "n": int(cm[i, j]),
                             "per_1000": cm[i, j] / cm.sum() * 1000})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------- figures
def fig_confusion(cm: pd.DataFrame, threshold: float, path):
    d = cm[cm.population == "all"]
    mat = d.pivot(index="actual", columns="decision", values="n").reindex(["repaid", "defaulted"])[["approved", "declined"]]
    per_k = mat / mat.values.sum() * 1000
    caught = mat.loc["defaulted", "declined"] / mat.loc["defaulted"].sum()
    lost = mat.loc["repaid", "declined"] / mat.loc["repaid"].sum()
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.imshow(per_k.values, cmap="Blues", vmin=0, vmax=per_k.values.max() * 1.15)
    for i in range(2):
        for j in range(2):
            v = per_k.values[i, j]
            ax.text(j, i, f"{mat.values[i, j]:,}\n({v:.0f} per 1,000)", ha="center", va="center",
                    color="white" if v > 400 else INK, fontsize=11)
    ax.set_xticks([0, 1], [f"Approved\n(PD < {threshold:.1%})", f"Declined\n(PD >= {threshold:.1%})"])
    ax.set_yticks([0, 1], ["Repaid", "Defaulted"])
    ax.set_xlabel("Model decision"); ax.set_ylabel("What actually happened")
    ax.grid(False)
    ax.set_title(f"The model declines {caught:.0%} of defaulters\nwhile turning away {lost:.0%} of good customers",
                 loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_auc_by_set(mt: pd.DataFrame, path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    sets = [s for s in ("A", "B", "C") if s in set(mt["feature_set"])]
    x = np.arange(len(sets))
    w = 0.36
    for ax, pop in zip(axes, POPS):
        for j, kind in enumerate(("scorecard", "lgbm")):
            d = mt[(mt.population == pop) & (mt.model == kind)].set_index("feature_set").reindex(sets)
            xs = x + (j - 0.5) * w
            err = np.vstack([d["auc"] - d["auc_ci_low"], d["auc_ci_high"] - d["auc"]])
            ax.bar(xs, d["auc"], w * 0.92, color=MODEL_COLOR[kind], label=MODEL_NAME[kind], zorder=2)
            ax.errorbar(xs, d["auc"], yerr=err, fmt="none", ecolor=INK2, elinewidth=1, capsize=3, zorder=3)
            for xi, v, top in zip(xs, d["auc"], d["auc_ci_high"]):
                ax.text(xi, top + 0.004, f"{v:.3f}", ha="center", va="bottom", fontsize=8, color=INK)
        ax.set_xticks(x, [SET_NAME[s].replace(": ", "\n") for s in sets])
        ax.set_title(POPS[pop])
        ax.set_ylim(0.5, 0.85)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Test ROC-AUC (95% bootstrap CI)\naxis starts at 0.5 = random ranking")
    axes[0].legend(loc="upper left")
    fig.suptitle("Predictive power by feature set: how much does each data source add?", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_roc_thin(test: pd.DataFrame, kind: str, path):
    m = _pop_mask(test, "thin")
    fig, ax = plt.subplots(figsize=(5.2, 5))
    for i, fset in enumerate(("A", "B", "C")):
        c = f"{kind}_{fset}"
        if c not in test:
            continue
        fpr, tpr, _ = roc_curve(test.loc[m, "TARGET"], test.loc[m, c])
        auc = roc_auc_score(test.loc[m, "TARGET"], test.loc[m, c])
        ax.plot(fpr, tpr, color=SLOT[i], lw=2, label=f"{SET_NAME[fset]}  (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls="--")
    ax.set_xlabel("False positive rate (good applicants declined)")
    ax.set_ylabel("True positive rate (defaulters caught)")
    ax.set_title(f"{MODEL_NAME[kind]} on thin-file test applicants")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_calibration(test: pd.DataFrame, cols: list[str], path):
    fig, ax = plt.subplots(figsize=(5.2, 5))
    for i, c in enumerate(cols):
        q = pd.qcut(test[c], 10, duplicates="drop")
        g = test.groupby(q, observed=True).agg(p=(c, "mean"), y=("TARGET", "mean"))
        kind, fset = c.split("_", 1)
        ax.plot(g["p"], g["y"], marker="o", ms=6, lw=2, color=MODEL_COLOR[kind],
                label=f"{MODEL_NAME[kind]} ({fset})")
    lim = max(test[cols].quantile(0.99).max(), 0.35)
    ax.plot([0, lim], [0, lim], color=INK2, lw=1, ls="--", label="perfect calibration")
    ax.set_xlabel("Mean predicted PD (decile)")
    ax.set_ylabel("Observed default rate")
    ax.set_title("Calibration on the test split")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_shap(bundle: dict, X: pd.DataFrame, title: str, path, top: int = 15):
    contrib = bundle["model"].contributions(X).drop(columns="__base__")
    imp = contrib.abs().mean().sort_values(ascending=True).tail(top)
    fig, ax = plt.subplots(figsize=(7.5, 0.34 * top + 1.2))
    ax.barh([describe_feature(f) for f in imp.index], imp.values, color=SLOT[0], height=0.7, zorder=2)
    ax.set_xlabel("Mean |SHAP| (log-odds contribution)")
    ax.set_title(title, loc="left")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return imp.sort_values(ascending=False)


def fig_fairness(fair: pd.DataFrame, path):
    attrs = fair["attribute"].unique()
    fig, axes = plt.subplots(1, len(attrs), figsize=(4 * len(attrs), 3.8),
                             gridspec_kw={"width_ratios": [max(fair.attribute.eq(a).sum(), 2) for a in attrs]})
    for ax, a in zip(np.atleast_1d(axes), attrs):
        d = fair[fair.attribute == a]
        x = np.arange(len(d))
        ax.bar(x - 0.2, d["approval_rate"], 0.38, color=SLOT[0], label="all applicants", zorder=2)
        ax.bar(x + 0.2, d["approval_rate_good"], 0.38, color=SLOT[1], label="non-defaulters only", zorder=2)
        labels = [str(g).replace("Secondary / secondary special", "Secondary").replace(" / ", "/")
                  .replace("Higher education", "Higher").replace("Incomplete higher", "Incompl. higher")
                  for g in d["group"]]
        ax.set_xticks(x, labels, rotation=20 if len(d) > 3 else 0, ha="right" if len(d) > 3 else "center",
                      fontsize=8.5)
        ax.set_ylim(0, 1.05)
        ax.set_title(a.replace("_", " ").title(), loc="left")
        ax.grid(axis="x", visible=False)
    np.atleast_1d(axes)[0].set_ylabel("Approval rate at deployed threshold")
    np.atleast_1d(axes)[0].legend(loc="lower left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def evaluate(cfg: dict) -> dict:
    _style()
    art, rep = cfg["paths"]["artifacts_dir"], cfg["paths"]["reports_dir"]
    figs = rep / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    test = pd.read_parquet(art / "predictions" / "test.parquet")
    results = json.loads((art / "train_results.json").read_text())
    cols = [c for c in test.columns if c.startswith(("scorecard_", "lgbm_"))]

    mt = metrics_table(test, results, cols)
    mt.to_csv(rep / "metrics_table.csv", index=False)
    up = uplift_table(test, cols)
    up.to_csv(rep / "alt_data_uplift.csv", index=False)
    fig_auc_by_set(mt, figs / "auc_by_feature_set.png")
    for kind in ("scorecard", "lgbm"):
        fig_roc_thin(test, kind, figs / f"roc_thin_file_{kind}.png")

    serve = f"{cfg['serving']['model_type']}_{cfg['serving']['model_set']}"
    bundle = joblib.load(art / "models" / f"{serve}.joblib")
    fig_calibration(test, [c for c in (f"scorecard_{cfg['serving']['model_set']}", serve) if c in cols],
                    figs / "calibration.png")
    fair = fairness_table(test, serve, bundle["threshold"], cfg["fairness"]["groups"])
    fair.to_csv(rep / "fairness.csv", index=False)
    if "lgbm_C_noage" in cols:  # fairness-through-unawareness check
        results_noage = results["lgbm_C_noage"]["decision"]["threshold"]
        fairness_table(test, "lgbm_C_noage", results_noage, cfg["fairness"]["groups"]).to_csv(
            rep / "fairness_without_age.csv", index=False)
    fig_fairness(fair, figs / "fairness.png")
    cm = confusion(test, serve, bundle["threshold"])
    cm.to_csv(rep / "confusion_matrix.csv", index=False)
    fig_confusion(cm, bundle["threshold"], figs / "confusion_matrix.png")

    # Explanations on test applicants (sample for speed).
    feats = pd.read_parquet(cfg["paths"]["processed_dir"] / "features.parquet")
    feats = feats[feats.SK_ID_CURR.isin(test.SK_ID_CURR)].set_index("SK_ID_CURR")
    shap = {}
    for name in ("lgbm_C", "lgbm_B"):
        p = art / "models" / f"{name}.joblib"
        if not p.exists():
            continue
        b = joblib.load(p)
        for pop, mask in (("all", slice(None)), ("thin", feats.THIN_FILE == 1)):
            X = feats.loc[mask, b["features"]].sample(n=min(20000, int(np.sum(mask) if pop == "thin" else len(feats))),
                                                      random_state=cfg["seed"])
            imp = fig_shap(b, X, f"{MODEL_NAME['lgbm']} {name[-1]}: top drivers ({POPS[pop].lower()})",
                           figs / f"shap_{name}_{pop}.png")
            shap[f"{name}_{pop}"] = imp.round(4).to_dict()

    has_history = test["SK_ID_CURR"].map(feats["INST_N_ALL"].fillna(0)).to_numpy() > 0   # has instalment history
    h2h = head_to_head(test, has_history)
    h2h.to_csv(rep / "head_to_head.csv", index=False)
    log.info("Head-to-head (applicants with both behaviour and bureau history):\n%s", h2h.round(4).to_string(index=False))

    dec = cfg["decision"]
    decision_economics(test, cols, {c: results[c]["decision"]["threshold"] for c in cols},
                       dec["review_band"], dec["cost_fn"], dec["cost_fp"]).to_csv(
        rep / "decision_economics.csv", index=False)
    (rep / "shap_by_source.json").write_text(json.dumps(shap_by_source(art, feats, cfg["seed"]), indent=1))

    summary = {
        "serving_model": serve, "threshold": bundle["threshold"],
        "decision_oof": results[serve]["decision"],
        "headline": mt[mt.model.eq(cfg["serving"]["model_type"])].pivot(
            index="feature_set", columns="population", values="auc").round(4).to_dict(),
        "shap_top": shap,
        "confusion_matrix": cm.to_dict(orient="records"),
        "head_to_head": h2h.to_dict(orient="records"),
    }
    (rep / "metrics.json").write_text(json.dumps(summary, indent=1, default=str))
    log.info("Test AUC by set/population (LightGBM):\n%s", mt[mt.model == "lgbm"].pivot(
        index="feature_set", columns="population", values="auc").round(4))
    log.info("Uplift (paired bootstrap):\n%s", up.round(4).to_string(index=False))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config")
    evaluate(load_config(ap.parse_args().config))


if __name__ == "__main__":
    main()
