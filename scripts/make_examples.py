"""Write example API requests from real test-split applicants (one thin-file, one with a bureau file).

    python scripts/make_examples.py
"""
import json
import sys
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from thinfile.config import load_config  # noqa: E402

TABLES = ["previous_application", "installments", "pos_cash", "credit_card", "bureau"]


def records(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.to_json(orient="records"))


def main():
    cfg = load_config()
    proc = cfg["paths"]["processed_dir"]
    test = pd.read_parquet(cfg["paths"]["artifacts_dir"] / "predictions" / "test.parquet")
    con = duckdb.connect()
    picks = {
        "thin_file_applicant": int(test[(test.THIN_FILE == 1)].SK_ID_CURR.iloc[0]),
        "bureau_file_applicant": int(test[(test.THIN_FILE == 0)].SK_ID_CURR.iloc[0]),
    }
    out_dir = ROOT / "examples"
    out_dir.mkdir(exist_ok=True)
    for name, sid in picks.items():
        app = con.sql(f"SELECT * EXCLUDE (TARGET) FROM '{proc}/application.parquet' WHERE SK_ID_CURR={sid}").df()
        req = {"application": records(app)[0]}
        for t in TABLES:
            df = con.sql(f"SELECT * EXCLUDE (SK_ID_CURR) FROM '{proc}/{t}.parquet' WHERE SK_ID_CURR={sid}").df()
            if len(df):
                req[t] = records(df)
        if "bureau" in req:
            ids = ",".join(str(b["SK_ID_BUREAU"]) for b in req["bureau"])
            bb = con.sql(f"SELECT * FROM '{proc}/bureau_balance.parquet' WHERE SK_ID_BUREAU IN ({ids})").df()
            if len(bb):
                req["bureau_balance"] = records(bb)
        # drop nulls to keep the files readable; missing fields are treated as missing by the API
        req["application"] = {k: v for k, v in req["application"].items() if v is not None}
        (out_dir / f"{name}.json").write_text(json.dumps(req, indent=1))
        print(name, sid, {k: len(v) for k, v in req.items() if isinstance(v, list)})


if __name__ == "__main__":
    main()
