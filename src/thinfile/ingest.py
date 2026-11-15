"""Stage 1 - ingestion.

Reads the raw Home Credit CSVs, checks each against an expected schema, and writes
typed Parquet copies to data/processed/. Also records the column schema of every table
so the API can build correctly-typed empty tables for applicants with no history.

    python -m thinfile.ingest
"""
from __future__ import annotations

import argparse
import json

import duckdb

from thinfile.config import get_logger, load_config

log = get_logger("ingest")

# Columns the feature code depends on. Ingestion fails loudly if any are missing,
# rather than letting a renamed column silently become all-NaN features.
REQUIRED_COLUMNS = {
    "application": ["SK_ID_CURR", "TARGET", "AMT_CREDIT", "AMT_ANNUITY", "AMT_INCOME_TOTAL",
                    "DAYS_BIRTH", "DAYS_EMPLOYED", "CODE_GENDER"],
    "previous_application": ["SK_ID_PREV", "SK_ID_CURR", "NAME_CONTRACT_STATUS", "DAYS_DECISION",
                             "AMT_APPLICATION", "AMT_CREDIT"],
    "installments": ["SK_ID_PREV", "SK_ID_CURR", "NUM_INSTALMENT_VERSION", "NUM_INSTALMENT_NUMBER",
                     "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT", "AMT_INSTALMENT", "AMT_PAYMENT"],
    "pos_cash": ["SK_ID_PREV", "SK_ID_CURR", "MONTHS_BALANCE", "SK_DPD", "SK_DPD_DEF",
                 "CNT_INSTALMENT_FUTURE", "NAME_CONTRACT_STATUS"],
    "credit_card": ["SK_ID_PREV", "SK_ID_CURR", "MONTHS_BALANCE", "AMT_BALANCE",
                    "AMT_CREDIT_LIMIT_ACTUAL", "AMT_PAYMENT_TOTAL_CURRENT", "AMT_INST_MIN_REGULARITY",
                    "AMT_DRAWINGS_CURRENT", "SK_DPD"],
    "bureau": ["SK_ID_CURR", "SK_ID_BUREAU", "CREDIT_ACTIVE", "DAYS_CREDIT", "AMT_CREDIT_SUM",
               "AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_SUM_OVERDUE", "CREDIT_DAY_OVERDUE"],
    "bureau_balance": ["SK_ID_BUREAU", "MONTHS_BALANCE", "STATUS"],
}


def ingest(cfg: dict, only: list[str] | None = None) -> dict:
    raw, out = cfg["paths"]["raw_dir"], cfg["paths"]["processed_dir"]
    out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    art = cfg["paths"]["artifacts_dir"]
    schema_path, report_path = art / "table_schemas.json", out / "ingest_report.json"
    # When re-ingesting a subset, keep what earlier runs recorded for the other tables.
    schemas = json.loads(schema_path.read_text()) if only and schema_path.exists() else {}
    report = json.loads(report_path.read_text()) if only and report_path.exists() else {}
    for name, spec in cfg["tables"].items():
        if only and name not in only:
            continue
        src = raw / spec["file"]
        if not src.exists():
            if spec["required"]:
                raise FileNotFoundError(f"Required table missing: {src}")
            log.warning("Optional table %s not found (%s) - its features will be left missing", name, src)
            report[name] = {"status": "missing"}
            continue
        rel = con.read_csv(str(src), sample_size=-1)
        cols = rel.columns
        missing = [c for c in REQUIRED_COLUMNS[name] if c not in cols]
        if missing:
            raise ValueError(f"{name}: expected columns not found: {missing}")
        dst = out / f"{name}.parquet"
        rel.write_parquet(str(dst), compression="zstd")
        n = con.execute(f"SELECT count(*) FROM '{dst}'").fetchall()[0][0]
        key = "SK_ID_BUREAU" if name == "bureau_balance" else "SK_ID_CURR"
        null_keys = con.execute(f"SELECT count(*) FROM '{dst}' WHERE {key} IS NULL").fetchall()[0][0]
        if null_keys:
            raise ValueError(f"{name}: {null_keys} rows with null {key}")
        if name == "application":
            dup = n - con.execute(f"SELECT count(DISTINCT SK_ID_CURR) FROM '{dst}'").fetchall()[0][0]
            if dup:
                raise ValueError(f"application: {dup} duplicate SK_ID_CURR")
        schemas[name] = dict(zip(rel.columns, [str(t) for t in rel.types]))
        report[name] = {"status": "ok", "rows": int(n), "columns": len(cols)}
        log.info("%-22s %12s rows  %4d cols -> %s", name, f"{n:,}", len(cols), dst.name)

    art.mkdir(parents=True, exist_ok=True)
    schema_path.write_text(json.dumps(schemas, indent=1))
    report_path.write_text(json.dumps(report, indent=1))
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config")
    ap.add_argument("--only", nargs="*", help="re-ingest just these tables")
    args = ap.parse_args()
    ingest(load_config(args.config), args.only)


if __name__ == "__main__":
    main()
