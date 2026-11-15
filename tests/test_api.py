"""API contract tests against a small model trained on synthetic data (no real data needed)."""
import importlib
import json
from datetime import datetime, timezone

import joblib
import pytest
from fastapi.testclient import TestClient

from thinfile.features.build import feature_set_columns, features_for_request
from thinfile.models.lgbm import LGBMModel
from tests.conftest import synthetic_application

WINDOWS = [3, 6, 12]


@pytest.fixture(scope="module")
def client(tmp_path_factory, schemas, monkeypatch_module):
    app_df = synthetic_application(2000, seed=1)
    feats = features_for_request({"application": app_df}, schemas, WINDOWS)
    cols = feature_set_columns(list(feats.columns), "C", ["SK_ID_CURR", "TARGET", "CODE_GENDER"])
    model = LGBMModel(n_estimators=50, learning_rate=0.1, verbose=-1).fit(feats[cols], feats["TARGET"])
    path = tmp_path_factory.mktemp("m") / "model.joblib"
    joblib.dump({"model": model, "model_type": "lgbm", "feature_set": "C", "features": cols, "threshold": 0.15,
                 "review_band": 0.5, "windows_months": WINDOWS, "metrics": {}, "table_schemas": schemas,
                 "version": "test", "trained_at": datetime.now(timezone.utc).isoformat()}, path)
    monkeypatch_module.setenv("MODEL_PATH", str(path))
    import api.app as api_app
    importlib.reload(api_app)
    with TestClient(api_app.app) as c:
        yield c


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def _application():
    row = synthetic_application(1, seed=3).iloc[0].to_dict()
    row.pop("TARGET")
    return json.loads(json.dumps(row, default=lambda v: v.item() if hasattr(v, "item") else None))


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_score_thin_file_applicant(client):
    r = client.post("/score", json={"application": _application()})
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0 <= body["probability_of_default"] <= 1
    assert body["decision"] in {"approve", "review", "decline"}
    assert body["thin_file"] is True
    assert len(body["top_reasons"]) <= 3


def test_score_with_history(client):
    app = _application()
    sid = app["SK_ID_CURR"]
    req = {
        "application": app,
        "installments": [{"SK_ID_PREV": 1, "NUM_INSTALMENT_VERSION": 1, "NUM_INSTALMENT_NUMBER": i,
                          "DAYS_INSTALMENT": -30 * i, "DAYS_ENTRY_PAYMENT": -30 * i + 5,
                          "AMT_INSTALMENT": 100.0, "AMT_PAYMENT": 100.0} for i in range(1, 7)],
        "bureau": [{"SK_ID_BUREAU": 9, "CREDIT_ACTIVE": "Closed", "DAYS_CREDIT": -700,
                    "AMT_CREDIT_SUM": 5000.0}],
        "bureau_balance": [{"SK_ID_BUREAU": 9, "MONTHS_BALANCE": -2, "STATUS": "C"}],
    }
    r = client.post("/score?reasons=5", json=req)
    assert r.status_code == 200, r.text
    assert r.json()["thin_file"] is False and r.json()["SK_ID_CURR"] == sid


def test_rejects_invalid_input(client):
    app = _application()
    app["AMT_CREDIT"] = -5
    assert client.post("/score", json={"application": app}).status_code == 422
    req = {"application": _application(), "bureau_balance": [{"SK_ID_BUREAU": 1, "MONTHS_BALANCE": -1,
                                                                "STATUS": "0"}]}
    assert client.post("/score", json=req).status_code == 422  # orphan bureau_balance row


def test_batch(client):
    r = client.post("/score/batch", json=[{"application": _application()}] * 3)
    assert r.status_code == 200 and len(r.json()) == 3
