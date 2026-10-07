import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from conftest import ROOT, needs_data
from kestrel_fraud.data import DATA_DIR


@needs_data
def test_predictions_match_sample_submission():
    pred_path = ROOT / "predictions.csv"
    if not pred_path.exists():
        pytest.skip("predictions.csv missing; run scripts/train_final.py")
    pred = pd.read_csv(pred_path)
    sample = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")
    test = pd.read_csv(f"{DATA_DIR}/test_unlabelled.csv")
    assert list(pred.columns) == list(sample.columns)
    assert len(pred) == len(sample) == len(test)
    assert (pred["claim_id"].values == sample["claim_id"].values).all()
    assert pred["claim_id"].is_unique and set(pred.claim_id) == set(test.claim_id)
    assert pred["score"].notna().all()
    assert pred["score"].between(0, 1).all()
    assert pred["score"].nunique() > 100  # a real ranking, not a constant


@pytest.fixture(scope="module")
def client(artifact):
    from service.api import app
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_predict_valid_claim(client, good_claim):
    r = client.post("/predict", json=good_claim)
    assert r.status_code == 200
    b = r.json()
    assert 0 <= b["fraud_score"] <= 1
    assert b["risk_band"] in {"LOW", "MEDIUM", "HIGH"}
    assert b["suggested_action"]
    assert 1 <= len(b["reasons"]) <= 4
    for rsn in b["reasons"]:
        assert rsn["direction"] in {"raises risk", "lowers risk"} and isinstance(rsn["reason"], str)
        assert (rsn["weight"] > 0) == (rsn["direction"] == "raises risk")


def test_predict_is_deterministic(client, good_claim):
    a = client.post("/predict", json=good_claim).json()
    b = client.post("/predict", json=good_claim).json()
    assert a == b


def test_risk_moves_with_the_policy_gap(client, good_claim):
    """Same claim: inspected large claim vs uninspected just-under-limit claim at a new outlet
    that has confirmed fraud. Not a hard-coded ID check: it uses the documented mechanism."""
    import joblib  # noqa: F401
    from kestrel_fraud.scoring import load_artifact
    art = load_artifact()
    hist = art["history"]
    # pick any outlet onboarded < 12 months before the claim date whose visible record has fraud
    partners = art["partners"].assign(onb=lambda d: pd.to_datetime(d.onboarded_date))
    cand = [p for p, (ts, cf) in hist.outcomes.items() if cf[-1] > 0
            and (pd.Timestamp("2026-09-15") - partners.set_index("partner_id").loc[p, "onb"]).days < 365]
    if not cand:
        pytest.skip("no such outlet in history")
    risky = {**good_claim, "partner_id": cand[0], "claim_amount_inr": 1950.0, "partner_inspected": "N"}
    safe = {**good_claim, "claim_amount_inr": 3500.0, "partner_inspected": "Y",
            "inspector_note": "Unit inspected, fault confirmed"}
    assert client.post("/predict", json=risky).json()["fraud_score"] > client.post("/predict", json=safe).json()["fraud_score"]


@pytest.mark.parametrize("patch", [
    {"claim_amount_inr": -5}, {"days_since_purchase": -1}, {"photo_attached": "maybe"},
    {"submitted_at": "yesterday"}, {"partner_id": "DROP TABLE"}, {"sku": "KH-ZZ-99"},
    {"unexpected_field": 1},
])
def test_malformed_input_is_rejected_politely(client, good_claim, patch):
    r = client.post("/predict", json={**good_claim, **patch})
    assert r.status_code == 422
    body = r.json()
    assert body["error"] == "invalid_claim" and "Traceback" not in json.dumps(body)


def test_missing_fields_and_bad_json(client):
    assert client.post("/predict", json={}).status_code == 422
    r = client.post("/predict", content="{not json", headers={"content-type": "application/json"})
    assert r.status_code == 422


def test_oversized_payload_rejected(client, good_claim):
    r = client.post("/predict", content=json.dumps({**good_claim, "claim_description": "x" * 20000}),
                    headers={"content-type": "application/json"})
    assert r.status_code == 413


def test_unknown_partner_scores_with_warning(client, good_claim):
    r = client.post("/predict", json={**good_claim, "partner_id": "SP99999"})
    assert r.status_code == 200 and r.json()["warnings"]


def test_missing_model_fails_politely(tmp_path, monkeypatch, good_claim):
    import service.api as api
    from kestrel_fraud import scoring
    monkeypatch.setattr(api, "_scorer", None)
    monkeypatch.setattr(api, "_load_error", None)
    monkeypatch.setattr(api, "load_artifact",
                        lambda: scoring.load_artifact(tmp_path / "nope.joblib", tmp_path / "nope.json"))
    c = TestClient(api.app)
    r = c.post("/predict", json=good_claim)
    assert r.status_code == 503 and "train_final" in r.json()["message"]
    assert c.get("/health").status_code == 503


def test_tampered_artifact_is_not_loaded(tmp_path):
    from kestrel_fraud.scoring import ArtifactError, load_artifact
    m, man = tmp_path / "m.joblib", tmp_path / "m.json"
    m.write_bytes(b"not the real model")
    man.write_text(json.dumps({"sha256": "0" * 64}))
    with pytest.raises(ArtifactError):
        load_artifact(m, man)


def test_explanations_are_exact_decomposition(artifact, clean):
    """Group contributions + baseline logit reproduce the model's logit exactly."""
    from kestrel_fraud.features import build_features
    labelled, test, *_ = clean
    X = build_features(test.head(50), artifact["history"])
    pipe, ex = artifact["pipeline"], artifact["explainer"]
    logit = np.log(pipe.predict_proba(X)[:, 1] / (1 - pipe.predict_proba(X)[:, 1]))
    base = pipe.named_steps["clf"].intercept_[0] + float(ex.mean @ ex.coef)
    recon = base + ex.group_contributions(X).sum(axis=1).values
    assert np.allclose(logit, recon, atol=1e-6)
