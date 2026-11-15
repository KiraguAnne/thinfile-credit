"""Application-form features (feature set A, plus the bureau-derived columns used in C).

Pure pandas, row-wise transformations only, so the same function serves a 300k-row
training frame and a single API request without any fitted state.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DAYS_PLACEHOLDER = 365243  # Home Credit's sentinel for "not applicable" (e.g. pensioners' employment)

# Columns in application_train that carry credit-bureau information. They are held back
# from set A so that A represents a genuine thin-file view of the applicant.
#   EXT_SOURCE_*: normalised scores from external data sources (bureau-type scores).
#   AMT_REQ_CREDIT_BUREAU_*: bureau enquiry counts. They are null only for applicants with
#   no bureau file (94% of those; never for applicants with a file), so even their
#   missingness leaks bureau status into a supposedly application-only model.
BUREAU_DERIVED = [
    "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
    "AMT_REQ_CREDIT_BUREAU_HOUR", "AMT_REQ_CREDIT_BUREAU_DAY", "AMT_REQ_CREDIT_BUREAU_WEEK",
    "AMT_REQ_CREDIT_BUREAU_MON", "AMT_REQ_CREDIT_BUREAU_QRT", "AMT_REQ_CREDIT_BUREAU_YEAR",
]

# Building-information block comes in _AVG, _MODE and _MEDI variants of the same 14 measures
# (correlation > 0.95). Keep _AVG only; also drop the low-information document flags that are
# set for < 0.1% of applicants.
_REDUNDANT_SUFFIXES = ("_MODE", "_MEDI")
_KEEP_MODE = {"FONDKAPREMONT_MODE", "HOUSETYPE_MODE", "WALLSMATERIAL_MODE", "EMERGENCYSTATE_MODE",
              "TOTALAREA_MODE"}
RARE_DOCUMENT_FLAGS = [f"FLAG_DOCUMENT_{i}" for i in (2, 4, 7, 9, 10, 12, 13, 14, 15, 17, 19, 20, 21)]


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace(0, np.nan)


def build_application_features(app: pd.DataFrame) -> pd.DataFrame:
    df = app.copy()

    # --- data-quality fixes -------------------------------------------------------------
    df["DAYS_EMPLOYED_ANOM"] = (df["DAYS_EMPLOYED"] == DAYS_PLACEHOLDER).astype("int8")
    df.loc[df["DAYS_EMPLOYED"] == DAYS_PLACEHOLDER, "DAYS_EMPLOYED"] = np.nan
    for col, bad in (("CODE_GENDER", "XNA"), ("NAME_FAMILY_STATUS", "Unknown"), ("ORGANIZATION_TYPE", "XNA")):
        if col in df:
            df.loc[df[col] == bad, col] = np.nan
    # A handful of applicants report car ages of 60+ years (likely data-entry); cap rather than drop.
    if "OWN_CAR_AGE" in df:
        df["OWN_CAR_AGE"] = df["OWN_CAR_AGE"].clip(upper=40)

    # --- derived features ---------------------------------------------------------------
    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365.25
    df["EMPLOYED_YEARS"] = -df["DAYS_EMPLOYED"] / 365.25
    df["EMPLOYED_TO_AGE"] = _safe_div(df["EMPLOYED_YEARS"], df["AGE_YEARS"])
    df["CREDIT_TO_INCOME"] = _safe_div(df["AMT_CREDIT"], df["AMT_INCOME_TOTAL"])
    df["ANNUITY_TO_INCOME"] = _safe_div(df["AMT_ANNUITY"], df["AMT_INCOME_TOTAL"])
    df["ANNUITY_TO_CREDIT"] = _safe_div(df["AMT_ANNUITY"], df["AMT_CREDIT"])  # ~ inverse of loan term
    df["CREDIT_TO_GOODS"] = _safe_div(df["AMT_CREDIT"], df["AMT_GOODS_PRICE"])  # financing beyond goods value
    df["INCOME_PER_PERSON"] = _safe_div(df["AMT_INCOME_TOTAL"], df["CNT_FAM_MEMBERS"])
    df["PHONE_CHANGE_TO_AGE"] = _safe_div(df["DAYS_LAST_PHONE_CHANGE"], df["DAYS_BIRTH"])
    df["ID_PUBLISH_TO_AGE"] = _safe_div(df["DAYS_ID_PUBLISH"], df["DAYS_BIRTH"])
    doc_cols = [c for c in df.columns if c.startswith("FLAG_DOCUMENT_")]
    df["N_DOCUMENTS"] = df[doc_cols].sum(axis=1)
    contact = [c for c in ("FLAG_MOBIL", "FLAG_EMP_PHONE", "FLAG_WORK_PHONE", "FLAG_CONT_MOBILE",
                           "FLAG_PHONE", "FLAG_EMAIL") if c in df]
    df["N_CONTACT_CHANNELS"] = df[contact].sum(axis=1)
    addr = [c for c in df.columns if c.startswith(("REG_REGION_NOT", "REG_CITY_NOT", "LIVE_REGION_NOT",
                                                    "LIVE_CITY_NOT"))]
    df["N_ADDRESS_MISMATCH"] = df[addr].sum(axis=1)
    # Exclude bureau-derived columns: their missingness alone reveals "no bureau file" (see BUREAU_DERIVED).
    # TARGET is excluded too: it is always filled in training but always blank at serving time.
    skip = [c for c in BUREAU_DERIVED + ["TARGET"] if c in app]
    df["N_MISSING_APP"] = app.drop(columns=skip).isna().sum(axis=1)

    # Age band is used only for the fairness audit (and kept as a readable group label).
    df["AGE_BAND"] = pd.cut(df["AGE_YEARS"], [0, 30, 45, 60, 200], labels=["<30", "30-44", "45-59", "60+"],
                            right=False).astype("object")

    drop = [c for c in df.columns if c.endswith(_REDUNDANT_SUFFIXES) and c not in _KEEP_MODE]
    drop += [c for c in RARE_DOCUMENT_FLAGS if c in df]
    return df.drop(columns=drop)

