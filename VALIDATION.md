# VALIDATION — does it work, and how often does it not?

All numbers come from `python scripts/evaluate.py` (→ `reports/evaluation.json`, `reports/holdout_errors.csv`) and `python scripts/partner_analysis.py` (→ `reports/partner_hypothesis.json`). A clean-environment rebuild (fresh venv from `requirements.txt`, code + task pack only) reproduced `predictions.csv` and every metric **exactly** (max score difference 0.0).

## 1. Strategy

* **Chronological only.** Random splits would let the model see future outlet behaviour and the post-May process change. Every fold rebuilds outlet history with outcomes cut off at the fold start (plus a 30-day closure lag), exactly as the final model is cut off at the 1 Jul 2026 export.
* **Separation:**
  - Model development: dev folds 1–4.
  - Final validation: the June 2026 holdout, scored after the configuration was locked (DECISIONS D-12, D-13).
  - Final scoring: `test_unlabelled.csv` (Jul–Sep 2026), never used for anything except the band thresholds, which use its unlabelled scores.

| Fold | Train (decided claims) | Validate | Claims / frauds |
|---|---|---|---|
| dev1 | < 1 Jan 2026 | Jan–Feb 2026 | 1,424 / 13 |
| dev2 | < 1 Mar 2026 | Mar–Apr 2026 | 1,447 / 18 |
| dev3 | < 1 May 2026 | May 2026 (process change; nothing like it in training) | 709 / 14 |
| dev4 | < 16 May 2026 | 16–31 May 2026 | 382 / 9 |
| **Holdout** | **< 1 Jun 2026 (10,433 claims, 119 fraud)** | **June 2026** | **713 / 22** |

* **Operating point = the desk's capacity.** The desk can investigate 40 claims a month, so "flag" means **in that month's top 40 scores**. Threshold metrics at 0.5 are also reported.

## 2. The 97% accuracy KPI

| Period | Fraud prevalence | "Nothing is fraud" accuracy |
|---|---|---|
| All 15 months (11,146 decided claims) | 1.27% | **98.7%** |
| Since 1 May 2026 | 2.53% | 97.5% |
| June 2026 holdout | 3.09% | **96.9%** |

**A model that never flags anything already beats 97% over the history, so 97% accuracy alone does not show that any fraud is detected.** On the June holdout:

* At a 0.5 threshold the model reaches **97.5% accuracy**, above the target and the baseline, but it catches only **6 of 22** frauds (recall 27%).
* At desk capacity it catches **14 of 22** (recall 64%) with accuracy **95.2%**. Accuracy drops because 40 reviews a month with about 22 frauds must include at least 18 genuine claims.

Accuracy rewards doing nothing. I recommend the board track "share of fraud caught" and "₹ stopped per check" alongside it.

## 3. Model comparison (June 2026 holdout, scored once)

Columns: fraud caught = within 40 reviews; net ₹ = fraud stopped − ₹380 × genuine held.

| Approach | PR-AUC | ROC-AUC | Fraud caught | Precision | Recall | F1 | Acc. | Bal. acc. | Fraud ₹ stopped | Net ₹ |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline: flag nothing | 0.031 | 0.50 | 0/22 | – | 0.00 | 0.00 | 96.9% | 0.50 | 0 | 0 |
| Rule: largest claims first | 0.050 | 0.46 | 1/22 | 0.03 | 0.05 | 0.03 | 91.6% | 0.50 | 17,866 | 3,046 |
| Rule: newest outlets first (Ritu's hypothesis as a rule) | 0.200 | 0.85 | 4/22 | 0.10 | 0.18 | 0.13 | 92.4% | 0.57 | 5,242 | −8,438 |
| Rule: outlet's past fraud rate | 0.101 | 0.77 | 5/22 | 0.13 | 0.23 | 0.16 | 92.7% | 0.59 | 8,210 | −5,090 |
| Logistic regression, unweighted | 0.349 | 0.92 | 13/22 | 0.33 | 0.59 | 0.42 | 95.0% | 0.78 | 18,938 | 8,678 |
| Gradient boosting, recency-weighted | 0.428 | 0.90 | 13/22 | 0.33 | 0.59 | 0.42 | 95.0% | 0.78 | 18,916 | 8,656 |
| Gradient boosting, unweighted | 0.316 | 0.91 | 10/22 | 0.25 | 0.45 | 0.32 | 94.1% | 0.71 | 14,480 | 3,080 |
| **FINAL: logistic regression, recency-weighted** | **0.529** | **0.94** | **14/22** | **0.35** | **0.64** | **0.45** | **95.2%** | **0.80** | **20,143** | **10,263** |
| Same model, queue by probability × amount | 0.500 | 0.94 | 13/22 | 0.33 | 0.59 | 0.42 | 95.0% | 0.78 | 19,304 | 9,044 |

Final model on June: confusion matrix at capacity **TP 14 · FP 26 · FN 8 · TN 665**; at 0.5 **TP 6 · FP 2 · FN 16 · TN 689**. Bootstrap 90% interval for PR-AUC is **0.37–0.70**, and for recall within capacity 0.50–0.83. With 22 frauds, every number here is noisy.

**Threshold sweep (June):**

| Threshold | Accuracy | Bal. acc. | Precision | Recall | F1 |
|---|---|---|---|---|---|
| 0.05 | 94.3% | 0.86 | 0.32 | 0.77 | 0.45 |
| 0.10 | 95.4% | 0.80 | 0.36 | 0.64 | 0.46 |
| 0.20 | 96.8% | 0.72 | 0.48 | 0.45 | 0.47 |
| 0.50 | 97.5% | 0.63 | 0.75 | 0.27 | 0.40 |

**Calibration (June):**

| Score band | Claims | Mean score | Observed fraud rate |
|---|---|---|---|
| < 0.01 | 639 | 0.0002 | 0.5% |
| 0.01–0.03 | 15 | 0.017 | 0% |
| 0.03–0.10 | 20 | 0.059 | 25% |
| 0.10–0.30 | 20 | 0.154 | 25% |
| ≥ 0.30 | 19 | 0.555 | 47% |

Brier score 0.021. The ranking is good. Mid-range scores *under*-state risk, so treat any claim above about 3% as worth a look, which is why the MEDIUM band starts at 0.034.

## 4. Stability across time (final configuration, every fold)

| Fold | PR-AUC | ROC-AUC | Recall within capacity | Net ₹ |
|---|---|---|---|---|
| dev1 Jan–Feb | 0.751 | 0.93 | 0.77 | 56,386 |
| dev2 Mar–Apr | 0.446 | 0.80 | 0.56 | 42,067 |
| **dev3 May (no post-change training data)** | **0.011** | **0.11** | **0.00** | **−15,200** |
| dev4 late May | 0.446 | 0.98 | 1.00 | 2,594 |
| **June holdout** | **0.529** | **0.94** | **0.64** | **10,263** |

**The most important result is dev3.** When the process changed on 1 May, a model trained only on the old pattern ranked fraud *worse than random*. Before May, fraud was large inspected claims at about 7 established franchises. After May it was small uninspected claims at a few new outlets. The model recovers within two weeks once a few post-change outcomes exist (dev4). **Every process change needs a monitored retrain.**

## 5. Error analysis (June holdout; full list in `reports/holdout_errors.csv`)

| Outcome | n | Median ₹ | New outlet | Auto-approved | Median score |
|---|---|---|---|---|---|
| Caught (TP) | 14 | 1,387 | 100% | 100% | 0.43 |
| Missed (FN) | 8 | 1,095 | 75% | 88% | 0.035 |
| False alarm (FP) | 26 | 1,630 | 100% | 100% | 0.18 |

Representative cases:
* **Obvious fraud, caught.** WC710957: ₹1,978 room-heater claim, no inspection, outlet 190 days old with 2 confirmed frauds already. Score 0.98.
* **Fraud missed, cold start.** WC711116 and WC711279 (₹1,406 / ₹1,646): outlet onboarded in June, 18–25 days old, with **no outcomes yet**. Scores 0.004–0.04. 4 of the 8 misses are outlets too new to have a record.
* **Fraud missed, low value.** WC710867 (₹777) and WC711200 (₹959): at known problem outlets, but far below ₹2,000, so the "just under the limit" signal is absent.
* **Fraud missed, old pattern.** WC711262: ₹17,866 inspected robot-vacuum claim at a 4-year-old freelance outlet. Score ≈ 0, because the model now down-weights the pre-May pattern, which produced only 2 frauds since May.
* **Genuine claim flagged.** WC710873: ₹1,995 air-fryer claim, no inspection, no photo, at a new outlet with 1 confirmed fraud. Score 0.77, but genuine. Roughly half of the problem outlets' claims are genuine, so outlet-level suspicion produces false alarms.
* **Ambiguous.** Claims at exactly ₹1,995 from 3–6-month-old outlets with no record (e.g. WC710930, score 0.18, genuine). The model cannot separate these without a photo or inspection.

**Failure modes, in order of importance:**
1. Brand-new outlets with no investigated claims (cold start). Eleven outlets were onboarded after the training data ends and appear only in the test period.
2. Fraud far below the ₹2,000 line.
3. A return of the old large-claim pattern.
4. Genuine customers at flagged outlets.

## 6. Robustness checks (June holdout)

| Variant | PR-AUC | ROC-AUC | Recall within capacity | Net ₹ |
|---|---|---|---|---|
| Final (30-day outcome lag) | 0.529 | 0.94 | 0.64 | 10,263 |
| Outcome lag 0 days | 0.568 | 0.94 | 0.68 | 10,993 |
| Outcome lag 60 days | 0.515 | 0.94 | 0.64 | 9,614 |
| Drop legacy Zoho rows | 0.564 | 0.95 | 0.68 | 11,602 |

## 7. Partner tenure ("the newer partners are the problem")

Tenure is measured at claim time; "new" means under 12 months.

| Period | Outlet | Claims | Frauds | Fraud rate |
|---|---|---|---|---|
| Before 1 May 2026 | established | 8,326 | 104 | 1.25% |
| Before 1 May 2026 | new | 1,398 | 1 | 0.07% |
| Since 1 May 2026 | established | 1,211 | 2 | 0.17% |
| Since 1 May 2026 | new | 211 | 34 | 16.1% |

* **Controlled.** Logistic regression since May with route (auto-approved), near-limit, amount and outlet type: new-outlet odds ratio **127 (95% CI 30–542)**. Excluding the five worst outlets it is still **24 (4–126)**. **The effect survives controls.**
* **Before May the effect was reversed.** Odds ratio 0.05 (0.01–0.38): new outlets were safer, and fraud came from established franchises (67 of 105 frauds at 7 outlets, none of which has had a fraud since May).
* **Concentration.** 5 of 45 new outlets produced **29 of 34** new-outlet frauds; 54% of their claims were fraudulent. The other 40 new outlets ran at **3.2%** (5/157).
* **Where the model's flags fall (June).** All 40 flags went to new-outlet claims: recall 70% (14/20) and precision 35% for new outlets. The 2 frauds at established outlets were missed.

Conclusion: the data supports "**a handful of** new outlets, **since auto-approval removed inspection** of small claims". It does not support "new partners" as a group. This is association, not cause.

## 8. Expected score on the hidden labels

I do not know the hidden labels. This is an estimate.

| Metric (if this is how it is scored) | Point estimate | Range |
|---|---|---|
| PR-AUC / average precision | **0.50** | 0.35–0.65 |
| ROC-AUC | **0.92** | 0.88–0.95 |
| Recall within 40 reviews/month (120 over Jul–Sep) | 60% | 45–75% |
| Accuracy if thresholded at 0.5 | 97–98% | (the "flag nothing" baseline is ≈97% at ~3% prevalence) |

**How I estimated it.** The June holdout (PR-AUC 0.53, bootstrap 0.37–0.70; ROC-AUC 0.94) is the closest analogue: same process regime, trained just before the period. Dev4 (0.45 / 0.98) and the pre-May folds (0.45–0.75 / 0.80–0.93) bracket it.

The test period differs from June in both directions:
* **Better:** the final model has two months of post-change claims, and the May–June outcomes for the problem outlets are visible. Those outlets have 63 claims in the test period.
* **Worse:**
  - Outlet records go stale (no outcomes after 30 Jun, so September uses 2–3-month-old records).
  - Eleven outlets onboarded after June have no record (failure mode 1).
  - Fraudsters can adapt, for example by moving below ₹1,500 or to new outlets.

The point estimate sits slightly below June to reflect those risks.

**Assumptions:**
* Test prevalence is similar to June (~3%; the model's own expected count is ~68 frauds across 2,252 claims, about 3.0%).
* The auto-approval rule was still in force in Jul–Sep (test inspection rate 21%, as in May–June).
* Hidden labels come from the same investigation desk.
