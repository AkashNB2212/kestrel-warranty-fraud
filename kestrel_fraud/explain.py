"""Deterministic reason codes from the logistic-regression model.

For a linear model the log-odds decompose exactly into per-feature terms:
    logit = intercept + sum_j coef_j * z_j
Each feature's contribution is coef_j * (z_j - mean_j), i.e. relative to the
average current claim. Correlated features are summed into a small number of
plain-language groups (one sentence each). A group is shown as "raises risk"
only when its net contribution is positive AND the claim's value for it is a
recognised risk state (and vice-versa for "lowers risk"); this suppresses
collinearity artefacts such as "a clean record raises risk". No language model
is involved, so a reason can never be invented.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import CATEGORICAL_FEATURES, NUMERIC_FEATURES

GROUPS = {
    "route": ["auto_approved", "auto_x_new_partner", "post_change", "new_partner", "tenure_days"],
    "outlet_record": ["p_prior_frauds", "p_prior_decided", "p_fraud_rate_sm"],
    "outlet_activity": ["p_claims_30d", "p_claims_90d", "p_small_share_90d", "p_near_thr_share_90d",
                        "p_uninspected_share_90d"],
    "amount": ["log_amount", "amount_to_list", "log_list_price", "family", "near_threshold",
               "auto_x_near_threshold"],
    "customer": ["customer_prior_claims"],
    "evidence": ["inspected", "note", "photo"],
    "timing": ["days_since_purchase", "warranty_used_frac"],
    "fault": ["fault"],
    "outlet_type": ["partner_type"],
    "serial": ["serial_fmt"],
}
FEATURE_GROUP = {f: g for g, fs in GROUPS.items() for f in fs}


def _inr(x: float) -> str:
    return f"Rs {x:,.0f}"


def describe(group: str, x: pd.Series, raw: dict):
    """Return (state, sentence). state: +1 risk state, -1 reassuring state, 0 neutral."""
    amt = float(raw.get("claim_amount_inr", 0))
    if group == "route":
        if x["auto_approved"] and x["new_partner"]:
            return +1, (f"Under Rs 2,000, so it is paid without inspection, and the outlet is new "
                        f"(onboarded {int(x['tenure_days'])} days ago).")
        if x["auto_approved"]:
            return 0, "Under Rs 2,000, so it is paid without inspection; the outlet is established."
        if np.isnan(x["tenure_days"]):
            return 0, "Outlet is not in the partner register."
        return -1, "Claim goes through the normal inspection route."
    if group == "outlet_record":
        fr, dec = int(x["p_prior_frauds"]), int(x["p_prior_decided"])
        if fr > 0:
            return +1, f"This outlet already has {fr} confirmed fraudulent claim(s) out of {dec} with an outcome."
        if dec >= 5:
            return -1, f"This outlet has a clean record: {dec} earlier claims with an outcome, none fraudulent."
        return 0, "Little investigation history for this outlet yet."
    if group == "outlet_activity":
        n30, near = int(x["p_claims_30d"]), x["p_near_thr_share_90d"]
        if n30 >= 4 or near >= 0.3:
            return +1, (f"Busy outlet: {n30} claims in the previous 30 days; "
                        f"{near * 100:.0f}% of its recent claims sit just under Rs 2,000.")
        return -1 if n30 <= 1 else 0, f"Outlet filed {n30} claim(s) in the previous 30 days."
    if group == "amount":
        r = x["amount_to_list"]
        if x["near_threshold"] and x["auto_approved"]:
            return +1, f"Claimed {_inr(amt)}: just under the Rs 2,000 no-inspection limit."
        if r >= 0.5:
            return +1, f"Claim is {r * 100:.0f}% of the {x['family']}'s list price ({_inr(amt)})."
        return -1, f"Claim is a modest {r * 100:.0f}% of the {x['family']}'s list price ({_inr(amt)})."
    if group == "customer":
        n = int(x["customer_prior_claims"])
        return (+1, f"Customer already has {n} earlier warranty claim(s).") if n else (-1, "Customer's first warranty claim.")
    if group == "evidence":
        if not x["inspected"] and not x["photo"]:
            return +1, "No inspection sign-off and no photo."
        if not x["inspected"]:
            return +1, "No inspection sign-off (photo attached)."
        return -1 if x["photo"] else 0, f"Inspected and signed off{' with photo' if x['photo'] else ''}."
    if group == "timing":
        return 0, f"Claim made {int(x['days_since_purchase'])} days after purchase."
    if group == "fault":
        return 0, f"Fault reported: '{x['fault']}'."
    if group == "outlet_type":
        return 0, f"Outlet type: {str(x['partner_type']).replace('_', ' ')}."
    if group == "serial":
        return (+1 if x["serial_fmt"] in ("hyphen", "lowercase", "other") else 0,
                f"Serial number typed in a non-standard format ({x['serial_fmt']}).")
    return 0, group


class LinearExplainer:
    def __init__(self, pipeline, X_background: pd.DataFrame):
        self.pre = pipeline.named_steps["pre"]
        self.coef = pipeline.named_steps["clf"].coef_.ravel()
        Z = self.pre.transform(X_background)
        Z = Z.toarray() if hasattr(Z, "toarray") else Z
        self.mean = np.asarray(Z).mean(axis=0)
        self.names = list(self.pre.get_feature_names_out())
        self.source = []
        for n in self.names:
            kind, rest = n.split("__", 1)
            self.source.append(rest if kind == "num" else
                               next(c for c in CATEGORICAL_FEATURES if rest.startswith(c + "_")))

    def contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Exact per-feature log-odds contributions (rows = claims)."""
        Z = self.pre.transform(X)
        Z = Z.toarray() if hasattr(Z, "toarray") else np.asarray(Z)
        df = pd.DataFrame((Z - self.mean) * self.coef, columns=self.names)
        return df.T.groupby(self.source).sum().T[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

    def group_contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        c = self.contributions(X)
        return c.T.groupby(lambda f: FEATURE_GROUP[f]).sum().T

    def reasons(self, X_row: pd.DataFrame, raw: dict, k: int = 4, min_abs: float = 0.1):
        g = self.group_contributions(X_row).iloc[0].sort_values(key=np.abs, ascending=False)
        x = X_row.iloc[0]
        ups, downs = [], []
        for group, v in g.items():
            if abs(v) < min_abs:
                continue
            state, text = describe(group, x, raw)
            if v > 0 and state > 0:
                ups.append({"direction": "raises risk", "weight": round(float(v), 2), "reason": text})
            elif v < 0 and state < 0:
                downs.append({"direction": "lowers risk", "weight": round(float(v), 2), "reason": text})
        # up to k reasons, risk drivers first, always at least one counter-point when there is one
        n_up = min(len(ups), k - 1 if downs else k)
        return ups[:n_up] + downs[:max(k - n_up, 0)][: (k - n_up)]
