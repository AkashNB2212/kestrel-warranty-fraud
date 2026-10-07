"""Leakage guards: features may only depend on information available at submission."""
import numpy as np
import pandas as pd
import pytest

from conftest import needs_data
from kestrel_fraud import features as feat
from kestrel_fraud.data import TARGET
from kestrel_fraud.evaluation import DEV_FOLDS, HOLDOUT, split
from kestrel_fraud.features import FEATURES, PartnerHistory


def _claims(times, partner="SP1", amounts=None, inspected=None, labels=None):
    n = len(times)
    return pd.DataFrame({
        "claim_id": [f"C{i}" for i in range(n)], "partner_id": partner, "ts": pd.to_datetime(times),
        "claim_amount_inr": amounts or [1950.0] * n, "inspected": inspected or [0] * n,
        TARGET: labels if labels is not None else [0] * n,
    })


def test_feature_list_excludes_target_ids_and_free_text():
    banned = {TARGET, "claim_id", "product_serial", "claim_description", "inspector_note",
              "submitted_at", "partner_id", "source", "city"}
    assert not banned & set(FEATURES)


def test_activity_features_use_only_strictly_earlier_claims():
    hist_claims = _claims(["2026-05-01", "2026-05-10", "2026-05-20"])
    h = PartnerHistory.build(hist_claims, hist_claims, label_cutoff="2026-07-01")
    f = h.features(["SP1"], [np.datetime64("2026-05-10")])
    assert f.loc[0, "p_claims_30d"] == 1          # only the 1 May claim, not itself or 20 May
    future = pd.concat([hist_claims, _claims(["2026-05-15", "2026-06-01"])], ignore_index=True)
    h2 = PartnerHistory.build(future, future, label_cutoff="2026-07-01")
    assert h2.features(["SP1"], [np.datetime64("2026-05-10")]).loc[0, "p_claims_30d"] == 1


def test_outcomes_respect_lag_and_cutoff():
    lab = _claims(["2026-01-01", "2026-03-01", "2026-05-25", "2026-06-20"], labels=[1, 1, 1, 1])
    h = PartnerHistory.build(lab, lab, label_cutoff="2026-06-01")
    t = np.datetime64("2026-06-15")
    f = h.features(["SP1"], [t])
    # 25 May is inside the 30-day lag, 20 Jun is after the cutoff -> only 2 visible
    assert f.loc[0, "p_prior_frauds"] == 2 and f.loc[0, "p_prior_decided"] == 2
    # flipping labels that are not visible must not change the features
    lab2 = lab.copy()
    lab2.loc[[2, 3], TARGET] = 0
    f2 = PartnerHistory.build(lab2, lab2, label_cutoff="2026-06-01").features(["SP1"], [t])
    pd.testing.assert_frame_equal(f, f2)
    # flipping a visible label must change them
    lab3 = lab.copy()
    lab3.loc[0, TARGET] = 0
    f3 = PartnerHistory.build(lab3, lab3, label_cutoff="2026-06-01").features(["SP1"], [t])
    assert f3.loc[0, "p_prior_frauds"] == 1


def test_label_lag_constant_is_positive():
    assert feat.LABEL_LAG_DAYS >= 1


@needs_data
def test_temporal_splits_are_chronological(clean):
    labelled = clean[0]
    for f in DEV_FOLDS + [HOLDOUT]:
        tr, va = split(labelled, f)
        assert len(tr) and len(va)
        assert tr["ts"].max() < va["ts"].min()
        assert va["ts"].min() >= pd.Timestamp(f.train_end) and va["ts"].max() < pd.Timestamp(f.valid_end)
    # holdout is the latest labelled block and comes after every dev validation window
    assert HOLDOUT.valid_end > max(f.valid_end for f in DEV_FOLDS)
    assert labelled["ts"].max() < pd.Timestamp(HOLDOUT.valid_end)


@needs_data
def test_test_period_is_after_all_labels(clean):
    labelled, test, *_ = clean
    assert test["ts"].min() > labelled["ts"].max()


@needs_data
def test_final_history_uses_no_outcome_after_cutoff(clean):
    labelled, test, all_claims, *_ = clean
    h = PartnerHistory.build(all_claims, labelled, label_cutoff="2026-07-01")
    for ts, _ in h.outcomes.values():
        assert ts.max() < np.datetime64("2026-07-01")
