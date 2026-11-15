"""Synthetic Home Credit-shaped data so tests run without the real (large, licensed) data."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCHEMAS = json.loads((ROOT / "artifacts" / "table_schemas.json").read_text())


def synthetic_application(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols = {}
    for c, t in SCHEMAS["application"].items():
        if t in ("VARCHAR",):
            cols[c] = rng.choice(["a", "b", "c", None], n)
        elif t in ("BIGINT", "INTEGER"):
            cols[c] = rng.integers(0, 2, n)
        else:
            cols[c] = rng.normal(0.5, 0.2, n)
    df = pd.DataFrame(cols)
    df["SK_ID_CURR"] = np.arange(100000, 100000 + n)
    df["CODE_GENDER"] = rng.choice(["F", "M"], n)
    df["AMT_INCOME_TOTAL"] = rng.lognormal(11.8, 0.5, n)
    df["AMT_CREDIT"] = rng.lognormal(13, 0.6, n)
    df["AMT_ANNUITY"] = df["AMT_CREDIT"] / rng.uniform(10, 40, n)
    df["AMT_GOODS_PRICE"] = df["AMT_CREDIT"] * rng.uniform(0.8, 1.0, n)
    df["DAYS_BIRTH"] = -rng.integers(21 * 365, 68 * 365, n)
    df["DAYS_EMPLOYED"] = np.where(rng.random(n) < 0.15, 365243, -rng.integers(30, 15000, n))
    df["CNT_FAM_MEMBERS"] = rng.integers(1, 5, n).astype(float)
    df["DAYS_LAST_PHONE_CHANGE"] = -rng.integers(0, 3000, n).astype(float)
    df["DAYS_ID_PUBLISH"] = -rng.integers(0, 6000, n)
    # risk signal: younger and higher credit-to-income -> more defaults
    logit = -2.5 + 0.8 * (df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"] > 4) + 0.6 * (df["DAYS_BIRTH"] > -30 * 365)
    df["TARGET"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return df


def installments_for(sk_id: int, rows: list[tuple[int, int | None, float, float]]) -> pd.DataFrame:
    """rows: (days_instalment, days_entry_payment, amt_instalment, amt_payment)."""
    return pd.DataFrame([{
        "SK_ID_PREV": 1, "SK_ID_CURR": sk_id, "NUM_INSTALMENT_VERSION": 1, "NUM_INSTALMENT_NUMBER": i + 1,
        "DAYS_INSTALMENT": d, "DAYS_ENTRY_PAYMENT": e, "AMT_INSTALMENT": a, "AMT_PAYMENT": p,
    } for i, (d, e, a, p) in enumerate(rows)])


@pytest.fixture(scope="session")
def schemas():
    return SCHEMAS


@pytest.fixture(scope="session")
def app_df():
    return synthetic_application(3000)
