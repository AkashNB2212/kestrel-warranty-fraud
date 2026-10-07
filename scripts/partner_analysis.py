"""Tests Ritu's hypothesis: "the newer partners are the problem".

Tenure is measured at claim time (submitted_at - onboarded_date); "new" = under
12 months. Writes reports/partner_hypothesis.json.

    python scripts/partner_analysis.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")
from kestrel_fraud.config import REPORT_DIR  # noqa: E402
from kestrel_fraud.data import AUTO_APPROVAL_LIMIT_INR, AUTO_APPROVAL_START, TARGET, load_clean  # noqa: E402


def rate_table(df, by):
    g = df.groupby(by, observed=True).agg(claims=(TARGET, "size"), fraud=(TARGET, "sum"),
                                          partners=("partner_id", "nunique"),
                                          fraud_inr=("fraud_inr", "sum"))
    g["fraud_rate"] = (g["fraud"] / g["claims"]).round(4)
    return g.reset_index().astype({c: str for c in by} if isinstance(by, list) else {by: str}).to_dict("records")


def main():
    L, *_ = load_clean()
    L = L.copy()
    L["tenure_m"] = (L["ts"] - L["onboarded"]).dt.days / 30.44
    L["new_partner"] = (L["tenure_m"] < 12).astype(int)
    L["period"] = np.where(L["ts"] >= AUTO_APPROVAL_START, "post_1May26", "pre_1May26")
    L["auto"] = ((L["ts"] >= AUTO_APPROVAL_START) & (L["claim_amount_inr"] < AUTO_APPROVAL_LIMIT_INR)).astype(int)
    L["near"] = ((L["claim_amount_inr"] >= 1800) & (L["claim_amount_inr"] < 2000)).astype(int)
    L["log_amt"] = np.log(L["claim_amount_inr"])
    L["fraud_inr"] = L["claim_amount_inr"] * L[TARGET]
    L["tenure_bucket"] = pd.cut(L["tenure_m"], [-0.01, 3, 6, 12, 24, 120],
                                labels=["0-3m", "3-6m", "6-12m", "12-24m", "24m+"])
    out = {}
    out["by_period_and_newness"] = rate_table(L, ["period", "new_partner"])
    out["by_period_and_tenure_bucket"] = rate_table(L, ["period", "tenure_bucket"])
    out["by_period_route_newness"] = rate_table(L[L.period == "post_1May26"], ["auto", "new_partner"])

    # Concentration among new partners after 1 May
    post_new = L[(L.period == "post_1May26") & (L.new_partner == 1)]
    pp = post_new.groupby("partner_id")[TARGET].agg(["size", "sum"]).sort_values("sum", ascending=False)
    top5 = pp.head(5)
    rest = post_new[~post_new.partner_id.isin(top5.index)]
    est_post = L[(L.period == "post_1May26") & (L.new_partner == 0)]
    out["concentration_post"] = {
        "new_partners_with_claims": int(len(pp)), "new_partners_with_any_fraud": int((pp["sum"] > 0).sum()),
        "fraud_from_new_partners": int(pp["sum"].sum()), "fraud_from_top5_new_partners": int(top5["sum"].sum()),
        "top5_claims": int(top5["size"].sum()),
        "top5_fraud_rate": round(float(top5["sum"].sum() / top5["size"].sum()), 3),
        "other_new_partners_fraud_rate": round(float(rest[TARGET].mean()), 4),
        "other_new_partners_claims": int(len(rest)), "other_new_partners_fraud": int(rest[TARGET].sum()),
        "established_fraud_rate": round(float(est_post[TARGET].mean()), 4),
        "established_claims": int(len(est_post)), "established_fraud": int(est_post[TARGET].sum()),
    }
    # Concentration among established partners before 1 May
    pre = L[L.period == "pre_1May26"]
    q = pre.groupby("partner_id")[TARGET].agg(["size", "sum"]).sort_values("sum", ascending=False)
    out["concentration_pre"] = {"partners_with_any_fraud": int((q["sum"] > 0).sum()),
                                "total_fraud": int(q["sum"].sum()), "top7_partner_fraud": int(q.head(7)["sum"].sum()),
                                "top7_fraud_after_1May": int(L[(L.period == "post_1May26") & L.partner_id.isin(q.head(7).index)][TARGET].sum()),
                                "top7_claims_after_1May": int(L[(L.period == "post_1May26") & L.partner_id.isin(q.head(7).index)].shape[0])}

    # Controlled comparison (logistic regression, odds ratios with 95% CI)
    def fit(df, formula):
        m = smf.logit(formula, data=df).fit(disp=0, maxiter=200, method="bfgs")
        ci = np.exp(m.conf_int())
        return {k: {"odds_ratio": round(float(np.exp(m.params[k])), 2),
                    "ci95": [round(float(ci.loc[k, 0]), 2), round(float(ci.loc[k, 1]), 2)],
                    "p": float(f"{m.pvalues[k]:.2g}")} for k in m.params.index if k != "Intercept"}
    post = L[L.period == "post_1May26"]
    out["controlled_post"] = {
        "new_only": fit(post, f"{TARGET} ~ new_partner"),
        "new_plus_route": fit(post, f"{TARGET} ~ new_partner + auto + near + log_amt + C(partner_type)"),
        "new_plus_route_excl_top5": fit(post[~post.partner_id.isin(top5.index)],
                                        f"{TARGET} ~ new_partner + auto + log_amt + C(partner_type)"),
    }
    out["controlled_pre"] = {"new_plus_controls": fit(pre, f"{TARGET} ~ new_partner + log_amt + C(partner_type)")}

    (REPORT_DIR / "partner_hypothesis.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
