"""Stage 2 - feature building.

Joins application features with the behavioural (B) and bureau (C) aggregates and writes
one modelling table, data/processed/features.parquet. The same `compute_features` is
called by the API on a single applicant's records.

    python -m thinfile.features.build
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Callable

import duckdb
import numpy as np
import pandas as pd

from thinfile.config import get_logger, load_config
from thinfile.features.application import BUREAU_DERIVED, build_application_features
from thinfile.features.behavioural import (
    add_trends, credit_card_features, installments_features, pos_cash_features, previous_application_features,
)
from thinfile.features.bureau import bureau_balance_features, bureau_features

log = get_logger("features")

# table(s) a module needs -> function
MODULES: list[tuple[tuple[str, ...], Callable]] = [
    (("previous_application",), previous_application_features),
    (("installments",), installments_features),
    (("pos_cash",), pos_cash_features),
    (("credit_card",), credit_card_features),
    (("bureau",), bureau_features),
    (("bureau", "bureau_balance"), bureau_balance_features),
]

AGE_DERIVED = ["DAYS_BIRTH", "AGE_YEARS", "EMPLOYED_TO_AGE", "PHONE_CHANGE_TO_AGE", "ID_PUBLISH_TO_AGE"]
META_COLUMNS = ["SK_ID_CURR", "TARGET", "THIN_FILE", "AGE_BAND", "CODE_GENDER"]
B_PREFIXES = ("PREV_", "INST_", "POS_", "CC_")
C_PREFIXES = ("BUR_", "BB_")


def feature_set_columns(columns: list[str], feature_set: str, excluded: list[str]) -> list[str]:
    """Columns that make up nested feature set A, B or C."""
    drop = set(META_COLUMNS) | set(excluded)
    cols = [c for c in columns if c not in drop]
    is_b = lambda c: c.startswith(B_PREFIXES) or c in {"INST_LATE_TREND", "INST_DPD_TREND"}  # noqa: E731
    is_c = lambda c: c.startswith(C_PREFIXES) or c in BUREAU_DERIVED  # noqa: E731
    if feature_set == "A":
        return [c for c in cols if not is_b(c) and not is_c(c)]
    if feature_set == "B":
        return [c for c in cols if not is_c(c)]
    if feature_set == "C":
        return cols
    # Sensitivity sets: separate bureau *history* from the opaque external scores.
    ext = [c for c in cols if c.startswith("EXT_SOURCE_")]
    if feature_set == "B_ext":   # behavioural + external scores, no bureau history
        return [c for c in cols if not is_c(c)] + ext
    if feature_set == "C_noext":  # everything except the external scores
        return [c for c in cols if c not in ext]
    if feature_set == "A_bureau":  # head-to-head: form + bureau history (no behaviour, no external scores)
        return [c for c in cols if not is_b(c) and c not in ext]
    if feature_set == "C_noage":  # fairness check: everything except age and age-derived ratios
        return [c for c in cols if c not in AGE_DERIVED]
    raise ValueError(feature_set)


def compute_features(con: duckdb.DuckDBPyConnection, app: pd.DataFrame, available: set[str],
                     windows: list[int]) -> pd.DataFrame:
    """Application features + every history aggregate whose source tables are registered on `con`."""
    feats = build_application_features(app)
    for tables, fn in MODULES:
        if not set(tables) <= available:
            continue
        agg = fn(con, windows)
        feats = feats.merge(agg, on="SK_ID_CURR", how="left")
    feats = add_trends(feats, windows)
    # Thin file = no record at the credit bureau. Kept as metadata for evaluation, not as an input.
    if "BUR_N_CREDITS" in feats:
        feats["THIN_FILE"] = feats["BUR_N_CREDITS"].isna().astype("int8")
    else:  # bureau table not supplied: bureau enquiry fields are null exactly for no-file applicants
        feats["THIN_FILE"] = feats["AMT_REQ_CREDIT_BUREAU_YEAR"].isna().astype("int8")
    # Counts are zero, not unknown, for an applicant with no records in a table.
    count_cols = [c for c in feats.columns if c.startswith(B_PREFIXES + C_PREFIXES)
                  and ("_N_" in c or c.endswith("_N")) and "SHARE" not in c]
    feats[count_cols] = feats[count_cols].fillna(0)
    num = feats.select_dtypes("number").columns
    feats[num] = feats[num].replace([np.inf, -np.inf], np.nan)
    # Fixed row order by applicant id, so the train/test split never depends on SQL output order.
    return feats.sort_values("SK_ID_CURR", kind="stable").reset_index(drop=True)


# ------------------------------------------------------------------------------- batch path
def build(cfg: dict) -> pd.DataFrame:
    proc = cfg["paths"]["processed_dir"]
    con = duckdb.connect()
    # Single-threaded aggregation: parallel float sums are added in varying order, which changes
    # results in the last bits and makes reruns non-identical. ~1 minute slower, fully reproducible.
    con.execute("SET threads = 1")
    available = set()
    for name in cfg["tables"]:
        path = proc / f"{name}.parquet"
        if path.exists():
            con.execute(f"CREATE VIEW {name} AS SELECT * FROM '{path}'")
            available.add(name)
    log.info("Tables available: %s", sorted(available))
    app = con.sql("SELECT * FROM application ORDER BY SK_ID_CURR").df()
    feats = compute_features(con, app, available, cfg["windows_months"])

    out = proc / "features.parquet"
    feats.to_parquet(out, index=False)
    names = cfg["feature_sets"] + cfg.get("sensitivity_sets", [])
    sets = {s: feature_set_columns(list(feats.columns), s, cfg["excluded_features"]) for s in names}
    (proc / "feature_sets.json").write_text(json.dumps(sets, indent=1))
    log.info("Wrote %s: %s rows x %s cols | thin-file share %.1f%%", out.name, f"{len(feats):,}",
             feats.shape[1], 100 * feats["THIN_FILE"].mean())
    for s, cols in sets.items():
        log.info("  feature set %s: %d features", s, len(cols))
    return feats


# ------------------------------------------------------------------------------ online path
def features_for_request(tables: dict[str, pd.DataFrame | None], schemas: dict[str, dict[str, str]],
                         windows: list[int]) -> pd.DataFrame:
    """Build features for applicants passed as raw records (used by the API).

    Every table is created with its training-time schema, so an applicant with no history
    still gets correctly typed (empty) tables and the SQL yields missing aggregates - the
    same thing a thin-file applicant looks like in training.
    """
    con = duckdb.connect()
    con.execute("SET threads = 1")  # one applicant's rows: thread start-up costs more than it saves
    for name, schema in schemas.items():
        cols = ", ".join(f'"{c}" {t}' for c, t in schema.items())
        con.execute(f"CREATE TABLE {name} ({cols})")
        df = tables.get(name)
        if df is not None and len(df):
            con.register("_incoming", df)
            con.execute(f"INSERT INTO {name} BY NAME SELECT * FROM _incoming")
            con.unregister("_incoming")
    app = con.sql("SELECT * FROM application").df()
    return compute_features(con, app, set(schemas), windows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config")
    build(load_config(ap.parse_args().config))


if __name__ == "__main__":
    main()
