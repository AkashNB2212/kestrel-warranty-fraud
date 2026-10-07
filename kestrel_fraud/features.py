"""Point-in-time feature construction.

Leakage rules (enforced here, tested in tests/test_leakage.py):
  * Claim-level features use only fields present on the claim at submission.
  * Partner *activity* features use other claims submitted strictly before this one
    (no labels involved, so test-period claims may contribute to later test claims).
  * Partner *fraud-history* features use investigation outcomes only for claims
    submitted before min(claim_time - LABEL_LAG, label_cutoff). label_cutoff is the
    moment the training export was taken (1 Jul 2026 for the final model, the fold
    start for validation folds), so no outcome from the scored period is ever used.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import AUTO_APPROVAL_LIMIT_INR, AUTO_APPROVAL_START, TARGET

LABEL_LAG_DAYS = 30          # assumed time for the desk to close an investigation
NEAR_THRESHOLD_LOW = 1800.0  # "just under" the Rs 2,000 auto-approval line
NEW_PARTNER_DAYS = 365
SMOOTHING = 20.0             # pseudo-claims for the partner fraud-rate shrinkage
DEFAULT_PRIOR = 0.01         # used until MIN_PRIOR_N outcomes are visible network-wide
MIN_PRIOR_N = 200

NUMERIC_FEATURES = [
    "log_amount", "amount_to_list", "log_list_price", "days_since_purchase",
    "warranty_used_frac", "customer_prior_claims", "photo", "inspected",
    "tenure_days", "new_partner", "post_change", "near_threshold", "auto_approved",
    "auto_x_new_partner", "auto_x_near_threshold",
    "p_claims_30d", "p_claims_90d", "p_small_share_90d", "p_near_thr_share_90d",
    "p_uninspected_share_90d", "p_prior_frauds", "p_prior_decided", "p_fraud_rate_sm",
]
CATEGORICAL_FEATURES = ["family", "partner_type", "fault", "note", "serial_fmt"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

_DAY = np.timedelta64(1, "D")


def claim_level_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    amt = df["claim_amount_inr"].astype(float).clip(lower=0)
    out["log_amount"] = np.log1p(amt)
    out["amount_to_list"] = amt / df["list_price_inr"].astype(float)
    out["log_list_price"] = np.log1p(df["list_price_inr"].astype(float))
    out["days_since_purchase"] = df["days_since_purchase"].astype(float)
    out["warranty_used_frac"] = df["days_since_purchase"] / (df["warranty_months"] * 30.44)
    out["customer_prior_claims"] = df["customer_prior_claims"].astype(float)
    out["photo"] = df["photo"].astype(float)
    out["inspected"] = df["inspected"].astype(float)
    out["tenure_days"] = (df["ts"] - df["onboarded"]).dt.days.astype(float)
    out["near_threshold"] = ((amt >= NEAR_THRESHOLD_LOW) & (amt < AUTO_APPROVAL_LIMIT_INR)).astype(float)
    out["new_partner"] = (out["tenure_days"] < NEW_PARTNER_DAYS).astype(float)
    # Policy §5 regime: after 1 May 2026 small claims skip inspection.
    out["post_change"] = (df["ts"] >= AUTO_APPROVAL_START).astype(float)
    out["auto_approved"] = (out["post_change"].astype(bool) & (amt < AUTO_APPROVAL_LIMIT_INR)).astype(float)
    out["auto_x_new_partner"] = out["auto_approved"] * out["new_partner"]
    out["auto_x_near_threshold"] = out["auto_approved"] * out["near_threshold"]
    for c in CATEGORICAL_FEATURES:
        out[c] = df[c].astype(str)
    return out


@dataclass
class PartnerHistory:
    """Per-partner event arrays used to answer point-in-time queries.

    activity: claim timestamps + flags (no labels, no customer data).
    outcomes: claim timestamps + fraud outcome for decided claims.
    """
    activity: dict = field(default_factory=dict)
    outcomes: dict = field(default_factory=dict)
    label_cutoff: np.datetime64 | None = None
    global_outcomes: tuple | None = None   # all partners: (ts, cumulative frauds)

    @classmethod
    def build(cls, all_claims: pd.DataFrame, labelled: pd.DataFrame, label_cutoff) -> "PartnerHistory":
        label_cutoff = np.datetime64(pd.Timestamp(label_cutoff))
        act = {}
        a = all_claims.sort_values("ts")
        amt = a["claim_amount_inr"].astype(float)
        a = a.assign(
            _small=(amt < AUTO_APPROVAL_LIMIT_INR).astype(int),
            _near=((amt >= NEAR_THRESHOLD_LOW) & (amt < AUTO_APPROVAL_LIMIT_INR)).astype(int),
            _unins=(1 - a["inspected"]).astype(int),
        )
        for pid, g in a.groupby("partner_id", sort=False):
            act[pid] = (
                g["ts"].values.astype("datetime64[ns]"),
                np.concatenate([[0], np.cumsum(g["_small"].values)]),
                np.concatenate([[0], np.cumsum(g["_near"].values)]),
                np.concatenate([[0], np.cumsum(g["_unins"].values)]),
            )
        lab = labelled[labelled["ts"].values < label_cutoff].sort_values("ts")
        outc = {}
        for pid, g in lab.groupby("partner_id", sort=False):
            outc[pid] = (
                g["ts"].values.astype("datetime64[ns]"),
                np.concatenate([[0], np.cumsum(g[TARGET].values.astype(int))]),
            )
        glob = (lab["ts"].values.astype("datetime64[ns]"),
                np.concatenate([[0], np.cumsum(lab[TARGET].values.astype(int))]))
        return cls(activity=act, outcomes=outc, label_cutoff=label_cutoff, global_outcomes=glob)

    def features(self, partner_ids, times) -> pd.DataFrame:
        times = np.asarray(times, dtype="datetime64[ns]")
        n = len(times)
        cols = {k: np.zeros(n) for k in [
            "p_claims_30d", "p_claims_90d", "p_small_share_90d", "p_near_thr_share_90d",
            "p_uninspected_share_90d", "p_prior_frauds", "p_prior_decided"]}
        prior = np.full(n, DEFAULT_PRIOR)
        lag = np.timedelta64(LABEL_LAG_DAYS, "D")
        gts, gcf = self.global_outcomes
        for i, (pid, t) in enumerate(zip(partner_ids, times)):
            if pid in self.activity:
                ts, cs, cn, cu = self.activity[pid]
                hi = np.searchsorted(ts, t, side="left")          # strictly before t
                lo30 = np.searchsorted(ts, t - 30 * _DAY, side="left")
                lo90 = np.searchsorted(ts, t - 90 * _DAY, side="left")
                n90 = hi - lo90
                cols["p_claims_30d"][i] = hi - lo30
                cols["p_claims_90d"][i] = n90
                if n90 > 0:
                    cols["p_small_share_90d"][i] = (cs[hi] - cs[lo90]) / n90
                    cols["p_near_thr_share_90d"][i] = (cn[hi] - cn[lo90]) / n90
                    cols["p_uninspected_share_90d"][i] = (cu[hi] - cu[lo90]) / n90
            visible = min(t - lag, self.label_cutoff)
            kg = np.searchsorted(gts, visible, side="left")
            if kg >= MIN_PRIOR_N:   # network-wide fraud rate known at that time
                prior[i] = gcf[kg] / kg
            if pid in self.outcomes:
                ts, cf = self.outcomes[pid]
                k = np.searchsorted(ts, visible, side="left")
                cols["p_prior_frauds"][i] = cf[k]
                cols["p_prior_decided"][i] = k
        out = pd.DataFrame(cols)
        out["p_fraud_rate_sm"] = (out["p_prior_frauds"] + SMOOTHING * prior) / (
            out["p_prior_decided"] + SMOOTHING)
        return out


def build_features(df: pd.DataFrame, history: PartnerHistory) -> pd.DataFrame:
    base = claim_level_features(df).reset_index(drop=True)
    part = history.features(df["partner_id"].values, df["ts"].values)
    X = pd.concat([base, part], axis=1)
    return X[FEATURES]
