"""Credit-bureau features (feature set C).

Thin-file applicants have no rows here by construction - these features are what a
traditional scorecard relies on, and what the behavioural set B tries to replace.
"""
from __future__ import annotations

import duckdb
import pandas as pd

from thinfile.features.behavioural import DAYS_PER_MONTH


def bureau_features(con: duckdb.DuckDBPyConnection, windows: list[int]) -> pd.DataFrame:
    opened = ",\n      ".join(
        f"sum((DAYS_CREDIT >= -{round(w * DAYS_PER_MONTH)})::INT) AS BUR_N_OPENED_{w}M" for w in windows
    )
    sql = f"""
    WITH b AS (SELECT * FROM bureau WHERE DAYS_CREDIT < 0)
    SELECT SK_ID_CURR,
      count(*)                                                      AS BUR_N_CREDITS,
      sum((CREDIT_ACTIVE = 'Active')::INT)                          AS BUR_N_ACTIVE,
      avg((CREDIT_ACTIVE = 'Closed')::INT)                          AS BUR_CLOSED_SHARE,
      count(DISTINCT CREDIT_TYPE)                                   AS BUR_N_CREDIT_TYPES,
      avg((CREDIT_TYPE = 'Credit card')::INT)                       AS BUR_CARD_SHARE,
      avg((CREDIT_TYPE = 'Microloan')::INT)                         AS BUR_MICROLOAN_SHARE,
      sum(AMT_CREDIT_SUM_DEBT) FILTER (WHERE CREDIT_ACTIVE = 'Active')  AS BUR_ACTIVE_DEBT_SUM,
      sum(AMT_CREDIT_SUM) FILTER (WHERE CREDIT_ACTIVE = 'Active')       AS BUR_ACTIVE_CREDIT_SUM,
      sum(AMT_CREDIT_SUM_DEBT) FILTER (WHERE CREDIT_ACTIVE = 'Active')
        / nullif(sum(AMT_CREDIT_SUM) FILTER (WHERE CREDIT_ACTIVE = 'Active'), 0) AS BUR_ACTIVE_DEBT_RATIO,
      sum(AMT_CREDIT_SUM_OVERDUE)                                   AS BUR_OVERDUE_SUM,
      max(CREDIT_DAY_OVERDUE)                                       AS BUR_DAY_OVERDUE_MAX,
      max(AMT_CREDIT_MAX_OVERDUE)                                   AS BUR_MAX_OVERDUE_AMT,
      sum(CNT_CREDIT_PROLONG)                                       AS BUR_N_PROLONG,
      max(DAYS_CREDIT)                                              AS BUR_DAYS_SINCE_LAST,
      min(DAYS_CREDIT)                                              AS BUR_DAYS_SINCE_FIRST,
      avg(DAYS_CREDIT_ENDDATE) FILTER (WHERE CREDIT_ACTIVE = 'Active') AS BUR_ACTIVE_ENDDATE_MEAN,
      {opened}
    FROM b GROUP BY SK_ID_CURR
    """
    return con.sql(sql).df()


def bureau_balance_features(con: duckdb.DuckDBPyConnection, windows: list[int]) -> pd.DataFrame:
    """Monthly bureau status: STATUS 1-5 = days-past-due buckets, C = closed, X = unknown."""
    win = ",\n      ".join(
        f"avg(dpd_flag) FILTER (WHERE MONTHS_BALANCE >= -{w}) AS BB_DPD_SHARE_{w}M,\n"
        f"      max(status_num) FILTER (WHERE MONTHS_BALANCE >= -{w}) AS BB_WORST_{w}M"
        for w in windows
    )
    sql = f"""
    WITH bb AS (
      SELECT b.SK_ID_CURR, bb.MONTHS_BALANCE,
             CASE WHEN bb.STATUS IN ('1','2','3','4','5') THEN bb.STATUS::INT ELSE 0 END AS status_num,
             (bb.STATUS IN ('1','2','3','4','5'))::INT AS dpd_flag
      FROM bureau_balance bb JOIN bureau b USING (SK_ID_BUREAU)
      WHERE bb.MONTHS_BALANCE < 0 AND bb.STATUS <> 'X'
    )
    SELECT SK_ID_CURR,
      count(*)            AS BB_N_MONTHS,
      avg(dpd_flag)       AS BB_DPD_SHARE_ALL,
      max(status_num)     AS BB_WORST_ALL,
      {win}
    FROM bb GROUP BY SK_ID_CURR
    """
    return con.sql(sql).df()
