"""Kestrel warranty-claim fraud scoring API.

    uvicorn service.api:app --port 8000

POST /predict  one claim as JSON -> fraud score, risk band, action, reasons
GET  /health   model status
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kestrel_fraud.scoring import ArtifactError, ClaimError, Scorer, load_artifact  # noqa: E402

log = logging.getLogger("kestrel.api")
MAX_BODY_BYTES = 8 * 1024

app = FastAPI(title="Kestrel warranty-claim fraud score", version="1.0.0",
              docs_url="/docs", redoc_url=None)

_scorer: Optional[Scorer] = None
_load_error: Optional[str] = None


def get_scorer() -> Optional[Scorer]:
    global _scorer, _load_error
    if _scorer is None and _load_error is None:
        try:
            _scorer = Scorer(load_artifact())
        except ArtifactError as e:
            _load_error = str(e)
        except Exception:  # never leak internals
            log.exception("model load failed")
            _load_error = "Model could not be loaded. Re-run `python scripts/train_final.py`."
    return _scorer


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    claim_id: Optional[str] = Field(None, max_length=32)
    submitted_at: datetime = Field(..., description="IST, e.g. 2026-09-14 10:30")
    partner_id: str = Field(..., pattern=r"^SP\d{3,6}$")
    sku: str = Field(..., pattern=r"^KH-[A-Z]{2}-\d{2}$")
    product_serial: str = Field("", max_length=40)
    days_since_purchase: int = Field(..., ge=0, le=3650)
    claim_amount_inr: float = Field(..., gt=0, le=1_000_000)
    photo_attached: Literal["Y", "N"]
    partner_inspected: Literal["Y", "N"]
    claim_description: str = Field("", max_length=500)
    inspector_note: Optional[str] = Field(None, max_length=500)
    customer_prior_claims: int = Field(0, ge=0, le=100)
    source: Literal["crm", "legacy_zoho"] = "crm"

    @field_validator("photo_attached", "partner_inspected", mode="before")
    @classmethod
    def upper_yn(cls, v):
        return v.upper() if isinstance(v, str) else v


@app.middleware("http")
async def limit_body(request: Request, call_next):
    cl = request.headers.get("content-length")
    if cl is not None and cl.isdigit() and int(cl) > MAX_BODY_BYTES:
        return JSONResponse(status_code=413, content={"error": "payload_too_large",
                                                      "message": f"Send one claim (max {MAX_BODY_BYTES} bytes)."})
    return await call_next(request)


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    problems = [{"field": ".".join(str(p) for p in e["loc"][1:]) or "body", "problem": e["msg"]}
                for e in exc.errors()]
    return JSONResponse(status_code=422, content={"error": "invalid_claim",
                                                  "message": "The claim could not be scored. Fix the fields below.",
                                                  "details": problems})


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    log.exception("unhandled error")
    return JSONResponse(status_code=500, content={"error": "internal_error",
                                                  "message": "Something went wrong while scoring. Nothing was saved."})


@app.get("/health")
def health():
    s = get_scorer()
    if s is None:
        return JSONResponse(status_code=503, content={"status": "model_unavailable", "message": _load_error})
    return {"status": "ok", "model_version": s.meta["version"], "outcomes_up_to": s.meta["label_cutoff"],
            "band_thresholds": s.bands}


@app.post("/predict")
def predict(claim: Claim):
    s = get_scorer()
    if s is None:
        return JSONResponse(status_code=503, content={"error": "model_unavailable", "message": _load_error})
    record = claim.model_dump()
    record["submitted_at"] = record["submitted_at"].strftime("%Y-%m-%d %H:%M")
    record["claim_id"] = record["claim_id"] or "ad-hoc"
    try:
        return s.score(record)
    except ClaimError as e:
        return JSONResponse(status_code=422, content={"error": "invalid_claim", "message": str(e)})
