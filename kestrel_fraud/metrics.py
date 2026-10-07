"""Evaluation metrics: the board's accuracy KPI, the class-aware metrics that show
whether fraud is actually caught, and the capacity-constrained rupee view
(ops-policy §4/§5: 40 reviews a month, Rs 380 goodwill per genuine claim held,
full claim amount lost per fraud paid, Rs 260 blended cost per service contact)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)

REVIEWS_PER_MONTH = 40
GOODWILL_PER_GENUINE_HELD = 380.0
SERVICE_CONTACT_COST = 260.0


def capacity_flags(scores, months, k: int = REVIEWS_PER_MONTH) -> np.ndarray:
    """Flag the top-k scored claims within each calendar month (the desk's capacity)."""
    s = pd.Series(np.asarray(scores, dtype=float))
    m = pd.Series(np.asarray(months))
    rank = s.groupby(m.values).rank(method="first", ascending=False)
    return (rank <= k).values.astype(int)


def classification_block(y, flags) -> dict:
    y = np.asarray(y).astype(int)
    flags = np.asarray(flags).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, flags, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y, flags),
        "balanced_accuracy": balanced_accuracy_score(y, flags),
        "precision": precision_score(y, flags, zero_division=0),
        "recall": recall_score(y, flags, zero_division=0),
        "f1": f1_score(y, flags, zero_division=0),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def rupee_block(y, flags, amounts, n_months: float) -> dict:
    y = np.asarray(y).astype(int)
    flags = np.asarray(flags).astype(int)
    amounts = np.asarray(amounts, dtype=float)
    caught = float(amounts[(y == 1) & (flags == 1)].sum())
    missed = float(amounts[(y == 1) & (flags == 0)].sum())
    goodwill = GOODWILL_PER_GENUINE_HELD * int(((y == 0) & (flags == 1)).sum())
    contact = SERVICE_CONTACT_COST * int(flags.sum())
    reviews = max(int(flags.sum()), 1)
    return {
        "fraud_exposure_inr": caught + missed,
        "fraud_stopped_inr": caught,
        "fraud_missed_inr": missed,
        "goodwill_cost_inr": goodwill,
        "review_contact_cost_inr": contact,
        "net_benefit_inr": caught - goodwill,
        "net_benefit_after_contact_inr": caught - goodwill - contact,
        "net_benefit_per_month_inr": (caught - goodwill) / n_months,
        "stopped_per_review_inr": caught / reviews,
    }


def full_report(y, scores, amounts, months) -> dict:
    y = np.asarray(y).astype(int)
    scores = np.asarray(scores, dtype=float)
    n_months = len(pd.unique(np.asarray(months)))
    flags = capacity_flags(scores, months)
    rep = {
        "n": int(len(y)), "n_fraud": int(y.sum()), "prevalence": float(y.mean()),
        "majority_baseline_accuracy": float(1 - y.mean()),
        "pr_auc": float(average_precision_score(y, scores)) if y.sum() else float("nan"),
        "roc_auc": float(roc_auc_score(y, scores)) if 0 < y.sum() < len(y) else float("nan"),
        "brier": float(brier_score_loss(y, np.clip(scores, 0, 1))),
        "mean_score": float(scores.mean()),
        "at_capacity": classification_block(y, flags),
        "at_0.5": classification_block(y, (scores >= 0.5).astype(int)),
        "rupees_at_capacity": rupee_block(y, flags, amounts, n_months),
    }
    # The ceiling: a perfect ranker with the same 40/month capacity.
    rep["rupees_perfect_ranker"] = rupee_block(
        y, capacity_flags(y * 1e6 + np.asarray(amounts) * y, months), amounts, n_months)
    return rep
