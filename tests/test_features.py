import numpy as np
import pandas as pd
import pytest

from thinfile.features.application import BUREAU_DERIVED, build_application_features
from thinfile.features.build import feature_set_columns, features_for_request
from tests.conftest import installments_for, synthetic_application

WINDOWS = [3, 6, 12]


def test_days_employed_placeholder_becomes_flag():
    app = synthetic_application(50)
    app.loc[0, "DAYS_EMPLOYED"] = 365243
    f = build_application_features(app)
    assert np.isnan(f.loc[0, "DAYS_EMPLOYED"])
    assert f.loc[0, "DAYS_EMPLOYED_ANOM"] == 1


def test_feature_sets_are_nested_and_bureau_free(app_df):
    cols = list(build_application_features(app_df).columns) + ["INST_LATE_RATE_ALL", "BUR_N_CREDITS"]
    excl = ["SK_ID_CURR", "TARGET", "CODE_GENDER"]
    a, b, c = (set(feature_set_columns(cols, s, excl)) for s in "ABC")
    assert a < b < c
    assert not (a & set(BUREAU_DERIVED)), "set A must not contain bureau-derived fields"
    assert "INST_LATE_RATE_ALL" in b and "BUR_N_CREDITS" not in b
    assert "CODE_GENDER" not in c, "protected attribute must never be a model input"


def test_installment_aggregates_are_correct(schemas):
    sid = 100000
    app = synthetic_application(1).assign(SK_ID_CURR=sid)
    inst = installments_for(sid, [
        (-30, -30, 100.0, 100.0),   # on time, full
        (-60, -50, 100.0, 100.0),   # 10 days late
        (-90, -95, 100.0, 50.0),    # early but partial
        (-400, -380, 100.0, 100.0),  # 20 days late, outside the 12m window
    ])
    f = features_for_request({"application": app, "installments": inst}, schemas, WINDOWS).iloc[0]
    assert f["INST_N_ALL"] == 4
    assert f["INST_LATE_RATE_ALL"] == pytest.approx(2 / 4)
    assert f["INST_PARTIAL_RATE_ALL"] == pytest.approx(1 / 4)
    assert f["INST_DPD_MAX_ALL"] == 20
    assert f["INST_N_12M"] == 3 and f["INST_DPD_MAX_12M"] == 10
    assert f["INST_N_3M"] == 3  # -90 days falls inside a 3-month (91-day) window


def test_records_after_application_date_are_ignored(schemas):
    """Leakage guard: anything dated on/after the application (day >= 0) must not count."""
    sid = 100000
    app = synthetic_application(1).assign(SK_ID_CURR=sid)
    inst = installments_for(sid, [(-30, -30, 100.0, 100.0), (15, 40, 100.0, 0.0)])
    f = features_for_request({"application": app, "installments": inst}, schemas, WINDOWS).iloc[0]
    assert f["INST_N_ALL"] == 1 and f["INST_LATE_RATE_ALL"] == 0


def test_applicant_with_no_history_is_thin_file(schemas):
    app = synthetic_application(1)
    f = features_for_request({"application": app}, schemas, WINDOWS).iloc[0]
    assert f["THIN_FILE"] == 1
    assert f["BUR_N_CREDITS"] == 0 and f["INST_N_ALL"] == 0  # counts are zero, not missing
    assert pd.isna(f["INST_LATE_RATE_ALL"]) and pd.isna(f["CC_UTIL_MEAN_ALL"])  # rates are unknown


def test_bureau_record_clears_thin_file_flag(schemas):
    sid = 100000
    app = synthetic_application(1).assign(SK_ID_CURR=sid)
    bureau = pd.DataFrame([{"SK_ID_CURR": sid, "SK_ID_BUREAU": 7, "CREDIT_ACTIVE": "Active",
                            "DAYS_CREDIT": -200, "AMT_CREDIT_SUM": 1000.0, "AMT_CREDIT_SUM_DEBT": 400.0}])
    bb = pd.DataFrame([{"SK_ID_BUREAU": 7, "MONTHS_BALANCE": m, "STATUS": s}
                       for m, s in ((-1, "0"), (-2, "1"), (-3, "0"), (-20, "3"))])
    f = features_for_request({"application": app, "bureau": bureau, "bureau_balance": bb},
                             schemas, WINDOWS).iloc[0]
    assert f["THIN_FILE"] == 0
    assert f["BUR_ACTIVE_DEBT_RATIO"] == pytest.approx(0.4)
    assert f["BB_DPD_SHARE_3M"] == pytest.approx(1 / 3)
    assert f["BB_WORST_ALL"] == 3 and f["BB_WORST_12M"] == 1


def test_serving_features_match_training_features(schemas):
    """Training/serving parity: the same applicant gets identical features whether TARGET is known or not."""
    app = synthetic_application(5)
    with_target = features_for_request({"application": app}, schemas, WINDOWS)
    without = features_for_request({"application": app.drop(columns="TARGET")}, schemas, WINDOWS)
    cols = [c for c in with_target.columns if c != "TARGET"]
    pd.testing.assert_frame_equal(with_target[cols], without[cols])
