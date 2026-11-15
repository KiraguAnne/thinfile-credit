"""Credit-scoring API.

POST /score takes one applicant's raw records - the application form plus any history the
lender holds (all optional). Features are computed with the same SQL used in training, so a
thin-file applicant is scored exactly the way thin-file applicants were seen in training.

    uvicorn api.app:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, model_validator

from thinfile.predict import Scorer

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.environ.get("MODEL_PATH", ROOT / "artifacts" / "models" / "lgbm_C.joblib"))

HISTORY_TABLES = ("previous_application", "installments", "pos_cash", "credit_card", "bureau", "bureau_balance")


class Application(BaseModel):
    """The application form. Core fields are validated; all other Home Credit fields are accepted as-is."""
    model_config = ConfigDict(extra="allow")
    SK_ID_CURR: int = Field(0, description="Applicant id; copied onto every history record")
    AMT_INCOME_TOTAL: float = Field(..., gt=0)
    AMT_CREDIT: float = Field(..., gt=0)
    AMT_ANNUITY: float | None = Field(None, gt=0)
    AMT_GOODS_PRICE: float | None = Field(None, gt=0)
    DAYS_BIRTH: int = Field(..., lt=-6570, gt=-36500, description="Negative days since birth (age 18-100)")
    DAYS_EMPLOYED: int | None = Field(None, description="Negative days employed; 365243 = not employed")


class ScoreRequest(BaseModel):
    application: Application
    previous_application: list[dict[str, Any]] = []
    installments: list[dict[str, Any]] = []
    pos_cash: list[dict[str, Any]] = []
    credit_card: list[dict[str, Any]] = []
    bureau: list[dict[str, Any]] = []
    bureau_balance: list[dict[str, Any]] = Field([], description="Rows must carry SK_ID_BUREAU of a bureau row")

    @model_validator(mode="after")
    def _bureau_balance_links(self):
        ids = {b.get("SK_ID_BUREAU") for b in self.bureau}
        orphans = {r.get("SK_ID_BUREAU") for r in self.bureau_balance} - ids
        if orphans:
            raise ValueError(f"bureau_balance rows reference unknown SK_ID_BUREAU: {sorted(orphans)[:5]}")
        return self


class Reason(BaseModel):
    feature: str
    description: str
    impact: float = Field(..., description="Contribution to risk (log-odds for LightGBM, points lost for scorecard)")
    value: Any = None


class ScoreResponse(BaseModel):
    SK_ID_CURR: int | None
    probability_of_default: float
    credit_score: int
    decision: Literal["approve", "review", "decline"]
    thin_file: bool | None
    top_reasons: list[Reason]
    model_version: str
    latency_ms: float


app = FastAPI(title="Thin-file credit scoring", version="1.0.0",
              description="Alternative-data credit scoring for applicants with little or no bureau history.")
_scorer: Scorer | None = None


def scorer() -> Scorer:
    global _scorer
    if _scorer is None:
        if not MODEL_PATH.exists():
            raise HTTPException(503, f"Model artifact not found at {MODEL_PATH}. Run `make train` first.")
        _scorer = Scorer(MODEL_PATH)
    return _scorer


def _tables(req: ScoreRequest) -> dict[str, pd.DataFrame | None]:
    sid = req.application.SK_ID_CURR
    tables: dict[str, pd.DataFrame | None] = {"application": pd.DataFrame([req.application.model_dump()])}
    for name in HISTORY_TABLES:
        rows = getattr(req, name)
        if not rows:
            tables[name] = None
            continue
        df = pd.DataFrame(rows)
        if name != "bureau_balance":
            df["SK_ID_CURR"] = sid
        tables[name] = df
    return tables


@app.on_event("startup")
def _load():
    try:
        scorer()
    except HTTPException:
        pass  # /health will report it


@app.get("/health")
def health():
    ok = MODEL_PATH.exists()
    return {"status": "ok" if ok else "model_missing", "model_path": str(MODEL_PATH)}


@app.get("/model-info")
def model_info():
    return scorer().info


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest, reasons: int = Query(3, ge=0, le=10)):
    t0 = time.perf_counter()
    s = scorer()
    try:
        res = s.score_raw(_tables(req), k_reasons=reasons)[0]
    except Exception as e:  # malformed history records, wrong types, etc.
        raise HTTPException(422, f"Could not score request: {e}") from e
    return {**res, "model_version": f"{s.info['model_type']}_{s.info['feature_set']}-{s.info['version']}",
            "latency_ms": round(1000 * (time.perf_counter() - t0), 1)}


@app.post("/score/batch", response_model=list[ScoreResponse])
def score_batch(reqs: list[ScoreRequest], reasons: int = Query(3, ge=0, le=10)):
    if len(reqs) > 500:
        raise HTTPException(413, "At most 500 applicants per batch")
    return [score(r, reasons) for r in reqs]
