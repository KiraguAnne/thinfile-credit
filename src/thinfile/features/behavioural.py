"""Behavioural features from the lender's own history (feature set B).

These four tables are the closest public analogue to mobile-money logs: dated, repeated
records of how a customer handled money owed to the lender. Each feature family is
aggregated over 3-, 6- and 12-month windows before the application, plus all-time.

All aggregation is DuckDB SQL over tables registered on a connection. Training registers
the full Parquet files; the API registers one applicant's records as small DataFrames.
The same SQL therefore runs in both places, which removes training/serving skew.

Leakage guard: every query keeps only records dated strictly before the application
(DAYS_* < 0, MONTHS_BALANCE < 0).
"""
from __future__ import annotations

import duckdb
import pandas as pd

DAYS_PER_MONTH = 30.4375
PLACEHOLDER = 365243


def _windowed(exprs: dict[str, str], cond_col: str, windows: list[int], unit_days: bool, prefix: str) -> str:
    """Build `agg(expr) FILTER (WHERE col >= -w)` columns for every window."""
    parts = []
    for w in windows:
        bound = round(w * DAYS_PER_MONTH) if unit_days else w
        for name, (agg, expr) in exprs.items():
            parts.append(f"{agg}({expr}) FILTER (WHERE {cond_col} >= -{bound}) AS {prefix}_{name}_{w}M")
    return ",\n  ".join(parts)


def installments_features(con: duckdb.DuckDBPyConnection, windows: list[int]) -> pd.DataFrame:
    """Payment timeliness, partial payments and days past due, per scheduled instalment."""
    win = {
        "N": ("count", "*"),
        "LATE_RATE": ("avg", "late"),
        "PARTIAL_RATE": ("avg", "partial"),
        "DPD_MEAN": ("avg", "dpd"),
        "DPD_MAX": ("max", "dpd"),
        "PAY_RATIO_MEAN": ("avg", "pay_ratio"),
    }
    sql = f"""
    WITH inst AS (   -- one row per scheduled instalment; a single instalment can be paid in several rows
      SELECT SK_ID_CURR, SK_ID_PREV, NUM_INSTALMENT_VERSION, NUM_INSTALMENT_NUMBER,
             max(DAYS_INSTALMENT) AS due_day, max(AMT_INSTALMENT) AS due_amt,
             sum(coalesce(AMT_PAYMENT, 0)) AS paid_amt, max(DAYS_ENTRY_PAYMENT) AS last_pay_day
      FROM installments WHERE DAYS_INSTALMENT < 0
      GROUP BY ALL
    ), x AS (
      SELECT *,
        -- an instalment never paid counts as outstanding up to the application date (day 0)
        greatest(coalesce(last_pay_day, 0) - due_day, 0)           AS dpd,
        (coalesce(last_pay_day, 0) > due_day)::INT                 AS late,
        (paid_amt < 0.99 * due_amt)::INT                           AS partial,
        greatest(due_day - last_pay_day, 0)                        AS days_early,
        least(paid_amt / due_amt, 2.0)                             AS pay_ratio,
        greatest(due_amt - paid_amt, 0)                            AS shortfall
      FROM inst WHERE due_amt > 0
    )
    SELECT SK_ID_CURR,
      count(*)                      AS INST_N_ALL,
      count(DISTINCT SK_ID_PREV)    AS INST_N_LOANS,
      avg(late)                     AS INST_LATE_RATE_ALL,
      avg(partial)                  AS INST_PARTIAL_RATE_ALL,
      avg(dpd)                      AS INST_DPD_MEAN_ALL,
      max(dpd)                      AS INST_DPD_MAX_ALL,
      avg(days_early)               AS INST_DAYS_EARLY_MEAN_ALL,
      avg(pay_ratio)                AS INST_PAY_RATIO_MEAN_ALL,
      sum(shortfall)                AS INST_SHORTFALL_SUM_ALL,
      stddev_samp(dpd)              AS INST_DPD_STD_ALL,
      max(due_day)                  AS INST_DAYS_SINCE_LAST_DUE,
      {_windowed(win, "due_day", windows, True, "INST")}
    FROM x GROUP BY SK_ID_CURR
    """
    return con.sql(sql).df()


def pos_cash_features(con: duckdb.DuckDBPyConnection, windows: list[int]) -> pd.DataFrame:
    """Monthly point-of-sale / cash-loan status: DPD incidence and remaining obligations."""
    win = {
        "N_MONTHS": ("count", "*"),
        "DPD_MEAN": ("avg", "SK_DPD"),
        "DPD_MAX": ("max", "SK_DPD"),
        "DPD_SHARE": ("avg", "(SK_DPD > 0)::INT"),
        "DPD_DEF_SHARE": ("avg", "(SK_DPD_DEF > 0)::INT"),
    }
    sql = f"""
    WITH latest AS (
      SELECT SK_ID_CURR, SK_ID_PREV, arg_max(NAME_CONTRACT_STATUS, MONTHS_BALANCE) AS last_status,
             arg_max(CNT_INSTALMENT_FUTURE, MONTHS_BALANCE) AS future_inst
      FROM pos_cash WHERE MONTHS_BALANCE < 0 GROUP BY ALL
    ), lat AS (
      SELECT SK_ID_CURR,
        count(*)                                        AS POS_N_LOANS,
        avg((last_status = 'Completed')::INT)           AS POS_COMPLETED_SHARE,
        sum((last_status = 'Active')::INT)              AS POS_N_ACTIVE,
        sum(future_inst) FILTER (WHERE last_status = 'Active') AS POS_FUTURE_INST_SUM
      FROM latest GROUP BY SK_ID_CURR
    ), agg AS (
      SELECT SK_ID_CURR,
        count(*)                        AS POS_N_MONTHS_ALL,
        avg(SK_DPD)                     AS POS_DPD_MEAN_ALL,
        max(SK_DPD)                     AS POS_DPD_MAX_ALL,
        avg((SK_DPD > 0)::INT)          AS POS_DPD_SHARE_ALL,
        {_windowed(win, "MONTHS_BALANCE", windows, False, "POS")}
      FROM pos_cash WHERE MONTHS_BALANCE < 0 GROUP BY SK_ID_CURR
    )
    SELECT * FROM agg LEFT JOIN lat USING (SK_ID_CURR)
    """
    return con.sql(sql).df()


def credit_card_features(con: duckdb.DuckDBPyConnection, windows: list[int]) -> pd.DataFrame:
    """Revolving behaviour: balance utilisation and its volatility, paying vs. the minimum, cash drawings."""
    win = {
        "UTIL_MEAN": ("avg", "util"),
        "UTIL_MAX": ("max", "util"),
        "UTIL_STD": ("stddev_samp", "util"),
        "PAY_TO_MIN_MEAN": ("avg", "pay_to_min"),
        "BELOW_MIN_SHARE": ("avg", "below_min"),
        "DRAWINGS_MEAN": ("avg", "AMT_DRAWINGS_CURRENT"),
        "DPD_SHARE": ("avg", "(SK_DPD > 0)::INT"),
    }
    sql = f"""
    WITH x AS (
      SELECT *,
        CASE WHEN AMT_CREDIT_LIMIT_ACTUAL > 0 THEN least(AMT_BALANCE / AMT_CREDIT_LIMIT_ACTUAL, 3.0) END AS util,
        CASE WHEN AMT_INST_MIN_REGULARITY > 0
             THEN least(coalesce(AMT_PAYMENT_TOTAL_CURRENT, 0) / AMT_INST_MIN_REGULARITY, 10.0) END AS pay_to_min,
        CASE WHEN AMT_INST_MIN_REGULARITY > 0
             THEN (coalesce(AMT_PAYMENT_TOTAL_CURRENT, 0) < AMT_INST_MIN_REGULARITY)::INT END AS below_min
      FROM credit_card WHERE MONTHS_BALANCE < 0
    )
    SELECT SK_ID_CURR,
      count(DISTINCT SK_ID_PREV)        AS CC_N_CARDS,
      count(*)                          AS CC_N_MONTHS_ALL,
      avg(util)                         AS CC_UTIL_MEAN_ALL,
      max(util)                         AS CC_UTIL_MAX_ALL,
      stddev_samp(util)                 AS CC_UTIL_STD_ALL,
      avg(pay_to_min)                   AS CC_PAY_TO_MIN_MEAN_ALL,
      avg(below_min)                    AS CC_BELOW_MIN_SHARE_ALL,
      max(SK_DPD)                       AS CC_DPD_MAX_ALL,
      {_windowed(win, "MONTHS_BALANCE", windows, False, "CC")}
    FROM x GROUP BY SK_ID_CURR
    """
    # Latest balance and limit summed across all of an applicant's cards (each card's most recent month).
    latest = """
    WITH last AS (
      SELECT SK_ID_CURR, SK_ID_PREV, max(MONTHS_BALANCE) AS m
      FROM credit_card WHERE MONTHS_BALANCE < 0 GROUP BY ALL
    )
    SELECT c.SK_ID_CURR,
           sum(c.AMT_BALANCE)              AS CC_LAST_BALANCE,
           sum(c.AMT_CREDIT_LIMIT_ACTUAL)  AS CC_LAST_LIMIT
    FROM credit_card c JOIN last l
      ON c.SK_ID_PREV = l.SK_ID_PREV AND c.SK_ID_CURR = l.SK_ID_CURR AND c.MONTHS_BALANCE = l.m
    GROUP BY c.SK_ID_CURR
    """
    return con.sql(sql).df().merge(con.sql(latest).df(), on="SK_ID_CURR", how="left")


def previous_application_features(con: duckdb.DuckDBPyConnection, windows: list[int]) -> pd.DataFrame:
    """Past applications to the lender: approval/refusal history and how loan asks were treated."""
    win = {
        "N": ("count", "*"),
        "REFUSED_N": ("sum", "(NAME_CONTRACT_STATUS = 'Refused')::INT"),
        "APPROVED_N": ("sum", "(NAME_CONTRACT_STATUS = 'Approved')::INT"),
    }
    sql = f"""
    WITH p AS (
      SELECT *,
        nullif(DAYS_LAST_DUE, {PLACEHOLDER})               AS last_due,
        nullif(DAYS_LAST_DUE_1ST_VERSION, {PLACEHOLDER})    AS last_due_1st,
        nullif(DAYS_TERMINATION, {PLACEHOLDER})             AS termination
      FROM previous_application WHERE DAYS_DECISION < 0
    )
    SELECT SK_ID_CURR,
      count(*)                                                   AS PREV_N_ALL,
      avg((NAME_CONTRACT_STATUS = 'Approved')::INT)              AS PREV_APPROVED_RATE,
      avg((NAME_CONTRACT_STATUS = 'Refused')::INT)               AS PREV_REFUSED_RATE,
      avg((NAME_CONTRACT_STATUS = 'Canceled')::INT)              AS PREV_CANCELED_RATE,
      avg(AMT_CREDIT / nullif(AMT_APPLICATION, 0))               AS PREV_CREDIT_TO_ASK_MEAN,
      min(AMT_CREDIT / nullif(AMT_APPLICATION, 0))               AS PREV_CREDIT_TO_ASK_MIN,
      avg(AMT_ANNUITY / nullif(AMT_CREDIT, 0))                   AS PREV_ANNUITY_TO_CREDIT_MEAN,
      avg(RATE_DOWN_PAYMENT)                                     AS PREV_DOWN_PAYMENT_RATE_MEAN,
      avg(CNT_PAYMENT)                                           AS PREV_TERM_MEAN,
      max(DAYS_DECISION)                                         AS PREV_DAYS_SINCE_LAST,
      min(DAYS_DECISION)                                         AS PREV_DAYS_SINCE_FIRST,
      avg((NAME_CLIENT_TYPE = 'New')::INT)                       AS PREV_NEW_CLIENT_SHARE,
      -- loans closed earlier than originally scheduled (early repayment is a positive signal)
      avg((last_due < last_due_1st)::INT) FILTER (WHERE NAME_CONTRACT_STATUS = 'Approved') AS PREV_EARLY_CLOSE_SHARE,
      sum((NAME_CONTRACT_STATUS = 'Approved' AND termination IS NULL AND last_due_1st > 0)::INT) AS PREV_N_ACTIVE,
      {_windowed(win, "DAYS_DECISION", windows, True, "PREV")}
    FROM p GROUP BY SK_ID_CURR
    """
    return con.sql(sql).df()


TREND_PAIRS = [  # recent minus longer-run behaviour: is the applicant getting better or worse?
    ("INST_LATE_RATE", "INST_LATE_TREND"),
    ("INST_DPD_MEAN", "INST_DPD_TREND"),
    ("POS_DPD_SHARE", "POS_DPD_TREND"),
    ("CC_UTIL_MEAN", "CC_UTIL_TREND"),
    ("CC_BELOW_MIN_SHARE", "CC_BELOW_MIN_TREND"),
]


def add_trends(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    short, long_ = min(windows), max(windows)
    for base, name in TREND_PAIRS:
        a, b = f"{base}_{short}M", f"{base}_{long_}M"
        if a in df and b in df:
            df[name] = df[a] - df[b]
    return df
