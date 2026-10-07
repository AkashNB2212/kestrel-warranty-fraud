import numpy as np
import pandas as pd

from conftest import needs_data
from kestrel_fraud.features import FEATURES, PartnerHistory, build_features
from kestrel_fraud.metrics import capacity_flags, full_report, rupee_block
from kestrel_fraud.model import make_hgb, make_logreg


@needs_data
def test_features_complete_and_handle_missing_note(clean):
    labelled, test, all_claims, *_ = clean
    h = PartnerHistory.build(all_claims, labelled, "2026-07-01")
    X = build_features(test.head(200), h)
    assert list(X.columns) == FEATURES
    assert X.drop(columns=["tenure_days"]).notna().all().all()
    assert test["inspector_note"].isna().any()  # missing notes are handled, not dropped


def _toy(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({c: rng.normal(size=n) for c in FEATURES})
    for c in ["family", "partner_type", "fault", "note", "serial_fmt"]:
        X[c] = rng.choice(["a", "b", "c"], size=n)
    X.loc[::17, "tenure_days"] = np.nan
    y = (X["amount_to_list"] + rng.normal(scale=0.5, size=n) > 1.2).astype(int)
    return X, y


def test_pipelines_fit_predict_deterministic_and_handle_unseen_categories():
    X, y = _toy()
    for make in (make_logreg, make_hgb):
        m1, m2 = make(), make()
        m1.fit(X, y); m2.fit(X, y)
        Xn = X.head(5).copy()
        Xn["family"] = "never-seen"
        p1, p2 = m1.predict_proba(Xn)[:, 1], m2.predict_proba(Xn)[:, 1]
        assert np.all((p1 >= 0) & (p1 <= 1)) and np.allclose(p1, p2)


def test_capacity_flags_top_k_per_month():
    scores = np.arange(100, dtype=float)
    months = np.array(["2026-07"] * 50 + ["2026-08"] * 50)
    f = capacity_flags(scores, months, k=10)
    assert f[:50].sum() == 10 and f[50:].sum() == 10
    assert f[40:50].all() and f[90:].all()


def test_rupee_block_arithmetic():
    y = np.array([1, 1, 0, 0])
    flags = np.array([1, 0, 1, 0])
    amt = np.array([1000.0, 500.0, 300.0, 200.0])
    r = rupee_block(y, flags, amt, n_months=1)
    assert r["fraud_stopped_inr"] == 1000 and r["fraud_missed_inr"] == 500
    assert r["goodwill_cost_inr"] == 380 and r["net_benefit_inr"] == 620
    assert r["review_contact_cost_inr"] == 2 * 260


def test_majority_baseline_reported_and_correct():
    y = np.array([0] * 97 + [1] * 3)
    rep = full_report(y, np.zeros(100), np.ones(100), np.array(["m"] * 100))
    assert abs(rep["majority_baseline_accuracy"] - 0.97) < 1e-9
