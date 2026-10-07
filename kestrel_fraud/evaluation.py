"""Chronological evaluation. Every fold re-builds partner history with
label_cutoff = fold start, so validation claims never see their own period's outcomes."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import TARGET
from .features import PartnerHistory, build_features
from .metrics import full_report


@dataclass(frozen=True)
class Fold:
    name: str
    train_end: str   # exclusive; also the label cutoff
    valid_end: str   # exclusive


# Development folds (used for model selection) and the final out-of-time holdout
# (reported once). Boundaries are calendar months, IST timestamps.
DEV_FOLDS = [
    Fold("dev1_JanFeb26", "2026-01-01", "2026-03-01"),
    Fold("dev2_MarApr26", "2026-03-01", "2026-05-01"),
    Fold("dev3_May26_regime_change", "2026-05-01", "2026-06-01"),
    # First fold with *some* post-change rows in training (1-15 May).
    Fold("dev4_lateMay26", "2026-05-16", "2026-06-01"),
]
HOLDOUT = Fold("holdout_Jun26", "2026-06-01", "2026-07-01")


def split(labelled: pd.DataFrame, fold: Fold):
    start, end = pd.Timestamp(fold.train_end), pd.Timestamp(fold.valid_end)
    tr = labelled[labelled["ts"] < start]
    va = labelled[(labelled["ts"] >= start) & (labelled["ts"] < end)]
    assert tr["ts"].max() < va["ts"].min(), "temporal split violated"
    return tr, va


def fold_matrices(labelled, all_claims, fold: Fold, train_filter=None):
    tr, va = split(labelled, fold)
    if train_filter is not None:
        tr = train_filter(tr)
    hist = PartnerHistory.build(all_claims, labelled, label_cutoff=fold.train_end)
    return tr, va, build_features(tr, hist), build_features(va, hist)


def evaluate_scores(va: pd.DataFrame, scores) -> dict:
    return full_report(va[TARGET].values, scores, va["claim_amount_inr"].values,
                       va["ts"].dt.to_period("M").astype(str).values)


def run_fold(make_model, labelled, all_claims, fold: Fold, train_filter=None, sample_weight_fn=None):
    tr, va, Xtr, Xva = fold_matrices(labelled, all_claims, fold, train_filter)
    model = make_model()
    fit_kw = {}
    if sample_weight_fn is not None:
        fit_kw["clf__sample_weight"] = sample_weight_fn(tr)
    model.fit(Xtr, tr[TARGET].values, **fit_kw)
    scores = model.predict_proba(Xva)[:, 1]
    return model, va, scores, evaluate_scores(va, scores)
