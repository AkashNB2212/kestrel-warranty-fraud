"""Single-claim scoring used by the API (and by tests)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import MANIFEST_PATH, MODEL_PATH
from .data import attach_reference, clean_claims
from .features import build_features


class ArtifactError(RuntimeError):
    pass


class ClaimError(ValueError):
    pass


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_artifact(model_path: Path = MODEL_PATH, manifest_path: Path = MANIFEST_PATH) -> dict:
    """joblib uses pickle, so only a file whose SHA-256 matches the manifest written
    by scripts/train_final.py is ever deserialised."""
    model_path, manifest_path = Path(model_path), Path(manifest_path)
    if not model_path.exists() or not manifest_path.exists():
        raise ArtifactError("Model artifact not found. Run `python scripts/train_final.py` first.")
    manifest = json.loads(manifest_path.read_text())
    if sha256(model_path) != manifest.get("sha256"):
        raise ArtifactError("Model artifact checksum does not match manifest; refusing to load it.")
    return joblib.load(model_path)


def risk_band(score: float, bands: dict) -> tuple[str, str]:
    if score >= bands["high"]:
        return "HIGH", "Hold payout and send to the investigation desk (within the 40/month capacity)."
    if score >= bands["medium"]:
        return "MEDIUM", "Pay only after a photo and serial check by the service desk."
    return "LOW", "Approve through the normal process."


class Scorer:
    def __init__(self, artifact: dict):
        self.a = artifact
        self.model = artifact["pipeline"]
        self.history = artifact["history"]
        self.partners = artifact["partners"]
        self.products = artifact["products"]
        self.explainer = artifact["explainer"]
        self.bands = artifact["bands"]
        self.meta = artifact["meta"]

    def featurise(self, claim: dict):
        df = pd.DataFrame([claim])
        if df.loc[0, "sku"] not in set(self.products["sku"]):
            raise ClaimError(f"Unknown sku '{df.loc[0, 'sku']}'. Valid SKUs: {sorted(self.products['sku'])}")
        warnings = []
        if df.loc[0, "partner_id"] not in set(self.partners["partner_id"]):
            warnings.append("partner_id is not in the partner register; tenure and outlet type are unknown.")
        try:
            clean = attach_reference(clean_claims(df), self.partners, self.products)
        except ValueError as e:
            raise ClaimError(str(e)) from None
        X = build_features(clean, self.history)
        return X, warnings

    def score(self, claim: dict) -> dict:
        X, warnings = self.featurise(claim)
        p = float(self.model.predict_proba(X)[0, 1])
        band, action = risk_band(p, self.bands)
        return {
            "claim_id": claim.get("claim_id"),
            "fraud_score": round(p, 4),
            "risk_band": band,
            "suggested_action": action,
            "reasons": self.explainer.reasons(X, claim),
            "warnings": warnings,
            "model": {"version": self.meta["version"], "trained_on_outcomes_up_to": self.meta["label_cutoff"],
                      "band_thresholds": self.bands},
        }

    def score_frame(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]
