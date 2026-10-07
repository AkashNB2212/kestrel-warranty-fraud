"""Train the final model on all decided claims up to 30 Jun 2026, score
test_unlabelled.csv and write predictions.csv + the service artifact.

    python scripts/train_final.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kestrel_fraud.config import (ARTIFACT_DIR, FINAL_C, FINAL_LABEL_CUTOFF, MANIFEST_PATH,  # noqa: E402
                                  MODEL_PATH, MODEL_VERSION, POST_CHANGE_WEIGHT, ROOT)
from kestrel_fraud.data import AUTO_APPROVAL_START, DATA_DIR, TARGET, load_clean  # noqa: E402
from kestrel_fraud.explain import LinearExplainer  # noqa: E402
from kestrel_fraud.features import FEATURES, PartnerHistory, build_features  # noqa: E402
from kestrel_fraud.metrics import REVIEWS_PER_MONTH  # noqa: E402
from kestrel_fraud.model import make_logreg  # noqa: E402
from kestrel_fraud.scoring import sha256  # noqa: E402


def capacity_bands(scores: np.ndarray, months: np.ndarray) -> dict:
    """HIGH = would make the desk's monthly top-40; MEDIUM = monthly top 10%."""
    s = pd.DataFrame({"s": scores, "m": months})
    high = s.groupby("m")["s"].apply(lambda x: np.sort(x.values)[::-1][min(REVIEWS_PER_MONTH, len(x)) - 1]).mean()
    medium = s.groupby("m")["s"].quantile(0.90).mean()
    return {"high": round(float(high), 4), "medium": round(float(min(medium, high)), 4)}


def main(out_predictions: Path = ROOT / "predictions.csv") -> dict:
    labelled, test, all_claims, partners, products = load_clean()
    assert labelled["ts"].max() < pd.Timestamp(FINAL_LABEL_CUTOFF)
    history = PartnerHistory.build(all_claims, labelled, label_cutoff=FINAL_LABEL_CUTOFF)

    X = build_features(labelled, history)
    y = labelled[TARGET].values
    w = np.where(labelled["ts"].values >= np.datetime64(AUTO_APPROVAL_START), POST_CHANGE_WEIGHT, 1.0)
    model = make_logreg(C=FINAL_C)
    model.fit(X, y, clf__sample_weight=w)

    Xte = build_features(test, history)
    scores = model.predict_proba(Xte)[:, 1]
    pred = pd.DataFrame({"claim_id": test["claim_id"].values, "score": np.round(scores, 6)})

    # Exact shape / order of sample_submission.csv
    sample = pd.read_csv(Path(DATA_DIR) / "sample_submission.csv")
    pred = sample[["claim_id"]].merge(pred, on="claim_id", how="left")
    assert list(pred.columns) == list(sample.columns)
    assert len(pred) == len(sample) and pred["score"].notna().all() and pred["claim_id"].is_unique
    pred.to_csv(out_predictions, index=False)

    bands = capacity_bands(scores, test["ts"].dt.to_period("M").astype(str).values)
    explainer = LinearExplainer(model, Xte)   # reasons are relative to the current claim mix
    meta = {
        "version": MODEL_VERSION, "label_cutoff": FINAL_LABEL_CUTOFF, "features": FEATURES,
        "C": FINAL_C, "post_change_weight": POST_CHANGE_WEIGHT,
        "n_train": int(len(y)), "n_train_fraud": int(y.sum()),
        "sklearn": sklearn.__version__, "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    artifact = {"pipeline": model, "history": history, "partners": partners, "products": products,
                "explainer": explainer, "bands": bands, "meta": meta}
    ARTIFACT_DIR.mkdir(exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)
    MANIFEST_PATH.write_text(json.dumps({**meta, "bands": bands, "sha256": sha256(MODEL_PATH)}, indent=2))

    coefs = pd.Series(model.named_steps["clf"].coef_.ravel(),
                      index=model.named_steps["pre"].get_feature_names_out()).sort_values()
    coefs.to_csv(ROOT / "reports" / "final_model_coefficients.csv", header=["coef"])
    print(f"trained on {len(y)} decided claims ({int(y.sum())} fraud); scored {len(pred)} test claims")
    print(f"score range {pred.score.min():.4f}-{pred.score.max():.4f}; bands {bands}")
    return {"pred": pred, "bands": bands, "meta": meta}


if __name__ == "__main__":
    main()
