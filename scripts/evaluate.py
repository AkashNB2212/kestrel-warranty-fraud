"""Chronological evaluation of the final configuration, challengers and baselines.

    python scripts/evaluate.py

Writes reports/evaluation.json (all numbers quoted in VALIDATION.md / the memo)
and reports/holdout_errors.csv (error analysis on June 2026).
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")
from kestrel_fraud import features as feat  # noqa: E402
from kestrel_fraud.config import FINAL_C, POST_CHANGE_WEIGHT, REPORT_DIR  # noqa: E402
from kestrel_fraud.data import AUTO_APPROVAL_START, TARGET, load_clean  # noqa: E402
from kestrel_fraud.evaluation import DEV_FOLDS, HOLDOUT, evaluate_scores, fold_matrices, run_fold  # noqa: E402
from kestrel_fraud.metrics import capacity_flags  # noqa: E402
from kestrel_fraud.model import make_hgb, make_logreg  # noqa: E402

recency = lambda tr: np.where(tr["ts"] >= AUTO_APPROVAL_START, POST_CHANGE_WEIGHT, 1.0)
CANDIDATES = {
    "FINAL_logreg_recency": dict(make=lambda: make_logreg(C=FINAL_C), sw=recency),
    "logreg_unweighted": dict(make=lambda: make_logreg(C=FINAL_C), sw=None),
    "hgb_recency": dict(make=lambda: make_hgb(), sw=recency),
    "hgb_unweighted": dict(make=lambda: make_hgb(), sw=None),
}


def rules(Xva: pd.DataFrame, va: pd.DataFrame) -> dict:
    return {
        "baseline_all_genuine": np.zeros(len(va)),
        "rule_largest_claims_first": va["claim_amount_inr"].values.astype(float),
        "rule_newest_partners_first": -Xva["tenure_days"].fillna(1e5).values,
        "rule_partner_fraud_history": Xva["p_fraud_rate_sm"].values,
    }


def rule_rows(Xva, va) -> dict:
    rows = {}
    for n, s in rules(Xva, va).items():
        r = evaluate_scores(va, s)
        if n == "baseline_all_genuine":   # flags nothing: the board's "accuracy" baseline
            from kestrel_fraud.metrics import classification_block, rupee_block
            zero = np.zeros(len(va), dtype=int)
            r["at_capacity"] = classification_block(va[TARGET].values, zero)
            r["rupees_at_capacity"] = rupee_block(va[TARGET].values, zero, va["claim_amount_inr"].values, 1)
        rows[n] = compact(r)
    return rows


def compact(r: dict) -> dict:
    c, b = r["at_capacity"], r["rupees_at_capacity"]
    return {"n": r["n"], "fraud": r["n_fraud"], "prevalence": round(r["prevalence"], 4),
            "baseline_acc": round(r["majority_baseline_accuracy"], 4), "pr_auc": round(r["pr_auc"], 3),
            "roc_auc": round(r["roc_auc"], 3), "acc@cap": round(c["accuracy"], 4),
            "bal_acc@cap": round(c["balanced_accuracy"], 3), "prec@cap": round(c["precision"], 3),
            "rec@cap": round(c["recall"], 3), "f1@cap": round(c["f1"], 3),
            "stopped_inr": round(b["fraud_stopped_inr"]), "exposure_inr": round(b["fraud_exposure_inr"]),
            "net_inr": round(b["net_benefit_inr"])}


def calibration_table(y, s, bins=(0, .01, .03, .1, .3, 1.0001)):
    df = pd.DataFrame({"y": y, "s": s, "b": pd.cut(s, bins, right=False)})
    g = df.groupby("b", observed=True).agg(n=("y", "size"), mean_score=("s", "mean"), fraud_rate=("y", "mean"))
    return [{"bin": str(i), **{k: (round(float(v), 4) if k != "n" else int(v)) for k, v in row.items()}}
            for i, row in g.iterrows()]


def tenure_breakdown(va, Xva, scores):
    flags = capacity_flags(scores, va["ts"].dt.to_period("M").astype(str).values)
    df = pd.DataFrame({"new": Xva["new_partner"].values.astype(bool), "y": va[TARGET].values, "f": flags})
    out = {}
    for k, g in df.groupby("new"):
        tp = int(((g.y == 1) & (g.f == 1)).sum())
        out["new_partner" if k else "established_partner"] = {
            "claims": int(len(g)), "fraud": int(g.y.sum()), "fraud_rate": round(float(g.y.mean()), 4),
            "flagged": int(g.f.sum()), "fraud_caught": tp,
            "recall": round(tp / max(int(g.y.sum()), 1), 3),
            "precision": round(tp / max(int(g.f.sum()), 1), 3)}
    return out


def main():
    labelled, test, all_claims, partners, products = load_clean()
    report = {"dev": {}, "holdout": {}, "sensitivity": {}}

    # ---- development folds (model selection happened here) -------------------
    for f in DEV_FOLDS:
        tr, va, Xtr, Xva = fold_matrices(labelled, all_claims, f)
        rows = rule_rows(Xva, va)
        for n, c in CANDIDATES.items():
            _, _, s, r = run_fold(c["make"], labelled, all_claims, f, sample_weight_fn=c["sw"])
            rows[n] = compact(r)
        report["dev"][f.name] = {"train_end": f.train_end, "valid_end": f.valid_end, "results": rows}

    # ---- final out-of-time holdout: June 2026 (scored once) --------------------
    tr, va, Xtr, Xva = fold_matrices(labelled, all_claims, HOLDOUT)
    rows = rule_rows(Xva, va)
    final_scores, full = None, None
    for n, c in CANDIDATES.items():
        _, _, s, r = run_fold(c["make"], labelled, all_claims, HOLDOUT, sample_weight_fn=c["sw"])
        rows[n] = compact(r)
        if n.startswith("FINAL"):
            final_scores, full = s, r
    # Same model, but the desk queue ordered by expected rupee loss instead of probability.
    ev = final_scores * va["claim_amount_inr"].values
    flags_ev = capacity_flags(ev, va["ts"].dt.to_period("M").astype(str).values)
    from kestrel_fraud.metrics import classification_block, rupee_block
    rows["FINAL_queue_by_expected_loss"] = {**compact(evaluate_scores(va, ev)),
                                            "net_inr": round(rupee_block(va[TARGET].values, flags_ev, va["claim_amount_inr"].values, 1)["net_benefit_inr"])}
    report["holdout"] = {
        "train_end": HOLDOUT.train_end, "valid_end": HOLDOUT.valid_end,
        "n_train": int(len(tr)), "n_train_fraud": int(tr[TARGET].sum()),
        "results": rows, "final_full": full,
        "calibration": calibration_table(va[TARGET].values, final_scores),
        "by_partner_tenure": tenure_breakdown(va, Xva, final_scores),
    }
    # thresholds other than capacity, for the accuracy discussion
    y = va[TARGET].values
    report["holdout"]["threshold_sweep"] = []
    for t in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7]:
        b = classification_block(y, (final_scores >= t).astype(int))
        report["holdout"]["threshold_sweep"].append({"threshold": t, **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in b.items()}})

    # ---- error analysis on the holdout -----------------------------------------
    err = va[["claim_id", "submitted_at", "partner_id", "partner_type", "family", "claim_amount_inr",
              "inspected", "photo", "customer_prior_claims", TARGET]].copy()
    err["score"] = final_scores
    err["flagged"] = capacity_flags(final_scores, va["ts"].dt.to_period("M").astype(str).values)
    for c in ["tenure_days", "new_partner", "auto_approved", "near_threshold", "p_prior_frauds",
              "p_prior_decided", "amount_to_list"]:
        err[c] = Xva[c].values
    err["outcome"] = np.select(
        [(err[TARGET] == 1) & (err.flagged == 1), (err[TARGET] == 1) & (err.flagged == 0),
         (err[TARGET] == 0) & (err.flagged == 1)], ["TP", "FN", "FP"], "TN")
    err.sort_values("score", ascending=False).to_csv(REPORT_DIR / "holdout_errors.csv", index=False)
    report["holdout"]["error_profile"] = (
        err.groupby("outcome").agg(n=("claim_id", "size"), median_amount=("claim_amount_inr", "median"),
                                   share_new_partner=("new_partner", "mean"), share_auto=("auto_approved", "mean"),
                                   share_inspected=("inspected", "mean"), mean_partner_frauds=("p_prior_frauds", "mean"),
                                   median_score=("score", "median"))
        .round(3).reset_index().to_dict(orient="records"))

    # ---- sensitivity: label lag assumption, legacy-label noise -----------------
    for lag in [0, 30, 60]:
        old = feat.LABEL_LAG_DAYS
        feat.LABEL_LAG_DAYS = lag
        _, _, _, r = run_fold(CANDIDATES["FINAL_logreg_recency"]["make"], labelled, all_claims, HOLDOUT,
                              sample_weight_fn=recency)
        feat.LABEL_LAG_DAYS = old
        report["sensitivity"][f"label_lag_{lag}d"] = compact(r)
    _, _, _, r = run_fold(CANDIDATES["FINAL_logreg_recency"]["make"], labelled, all_claims, HOLDOUT,
                          train_filter=lambda d: d[d["source"] != "legacy_zoho"], sample_weight_fn=recency)
    report["sensitivity"]["drop_legacy_zoho_rows"] = compact(r)

    # bootstrap CI for holdout PR-AUC / recall@capacity (claim-level resampling)
    from sklearn.metrics import average_precision_score
    rng = np.random.default_rng(0)
    months = va["ts"].dt.to_period("M").astype(str).values
    aps, recs = [], []
    for _ in range(1000):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum() == 0:
            continue
        aps.append(average_precision_score(y[i], final_scores[i]))
        fl = capacity_flags(final_scores[i], months[i])
        recs.append(((y[i] == 1) & (fl == 1)).sum() / y[i].sum())
    report["holdout"]["bootstrap_90ci"] = {
        "pr_auc": [round(float(np.percentile(aps, 5)), 3), round(float(np.percentile(aps, 95)), 3)],
        "recall_at_capacity": [round(float(np.percentile(recs, 5)), 3), round(float(np.percentile(recs, 95)), 3)]}

    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / "evaluation.json").write_text(json.dumps(report, indent=2, default=str))
    for name, block in [*report["dev"].items(), ("HOLDOUT " + HOLDOUT.name, report["holdout"])]:
        print(f"\n== {name}")
        print(pd.DataFrame(block["results"]).T.to_string())
    print("\nsensitivity"); print(pd.DataFrame(report["sensitivity"]).T.to_string())
    print("\nbootstrap", report["holdout"]["bootstrap_90ci"])
    print("\ntenure", json.dumps(report["holdout"]["by_partner_tenure"], indent=1))
    print("\nerrors", pd.DataFrame(report["holdout"]["error_profile"]).to_string())
    print("\ncalibration", pd.DataFrame(report["holdout"]["calibration"]).to_string())
    print("\nthresholds", pd.DataFrame(report["holdout"]["threshold_sweep"]).to_string())


if __name__ == "__main__":
    main()
