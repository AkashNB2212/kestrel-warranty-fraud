# DECISIONS — Kestrel warranty-claim fraud (Variant C)

Where the pack was ambiguous, I made a decision and recorded it here. The evidence for each is reproducible: `python scripts/audit.py` writes `reports/data_audit.md`.

## What the pack establishes (read before any modelling)

| Question | Answer from the pack |
|---|---|
| Target | `is_fraud` (1 fraud / 0 not fraud / blank = undecided at export). README.txt |
| Submission | `claim_id,score`, 2,252 rows, same order as `sample_submission.csv`; higher = more likely fraud |
| Claim time | `submitted_at`, IST. Train: 1 Apr 2025 – 30 Jun 2026. Test: 1 Jul – 30 Sep 2026 |
| Review capacity | **40 claims/month** (policy §5, Farhan's email) |
| Costs | Fraud paid = full claim amount. Genuine claim held = ₹380 goodwill. Service contact = ₹260 (policy §4) |
| Process change | From **1 May 2026** claims under ₹2,000 are auto-approved **without inspection** (policy §5) |
| Systems | Zoho until 30 Sep 2025, then CRM; `source` column. Zoho stored "undecided" as 0 (Tanmay) |
| Re-submissions | The same `claim_id` can appear twice (Tanmay) |
| New partners | About 60 onboarded in the past year (policy §6); 60 have `onboarded_date` ≥ Nov 2025 |
| Data handling | Must not be published or pushed to public repos (policy §10) |

## Decisions

**D-1 Undecided claims.** The 215 blank `is_fraud` rows (all from the CRM) are excluded from training and evaluation. They are neither fraud nor genuine yet.

**D-2 Legacy Zoho zeros.** Zoho could not store blanks, so some of its 4,875 "0" rows may really be undecided. The CRM blank rate (~3%) suggests roughly 150 such rows. They cannot be identified, so I keep them. Sensitivity check: dropping all Zoho rows raises June holdout PR-AUC from 0.529 to 0.564 but lowers dev fold 1 from 0.751 to 0.69, so the net effect is within noise. Kept (VALIDATION §6).

**D-3 Re-submissions.** 681 claims appear twice, 0–5 days apart, and are identical except for the timestamp, including the label. I keep the **first** submission, because that is when the payout decision is made. Test has no duplicates.

**D-4 Timestamps.** All `submitted_at` values are treated as IST. The UTC caveat in policy §9 applies only to *resolution events*, which are not in this export. The hour-of-day distribution is identical for the Zoho and CRM sources, so no shift was applied.

**D-5 Free text is untrusted.** `claim_description` is reduced to its leading fault phrase (13 known phrases) and `inspector_note` to its 7 fixed values plus `none`/`other`. Five descriptions have extra text appended after the fault phrase. This partner-typed text is ignored by design, and no free text reaches the model or an LLM.

**D-6 "New partner".** Tenure is measured **at claim time** (`submitted_at − onboarded_date`), and "new" means under 12 months, matching "the past year" in policy §6. No claim predates its outlet's onboarding.

**D-7 Point-in-time partner history.** An outlet's fraud record uses only outcomes for claims submitted before `min(claim time − 30 days, export cutoff)`. The 30-day lag is an assumption, because the export has no investigation-closure dates. Sensitivity at 0/30/60 days: holdout PR-AUC 0.568/0.529/0.515, so the conclusion does not depend on it. The network-wide prior used for shrinkage is point-in-time too. A leakage test caught an earlier version that used one global rate (D-13). Activity counts (claims in the last 30/90 days) use only claims submitted strictly before, which is fine because they are label-free.

**D-8 Policy regime as features.** `post_change`, `auto_approved` (on or after 1 May 2026 and under ₹2,000), `near_threshold` (₹1,800–1,999) and `auto_approved × new_partner` come from the written policy, not from looking at test data. Since May, fraud is 34 of 36 auto-approved claims.

**D-9 Validation design.** Chronological only (no random splits).
- Dev folds for selection: Jan–Feb 2026, Mar–Apr 2026, May 2026 (first month of the new process) and 16–31 May 2026 (the first fold whose training data contains post-change claims).
- **Final holdout: June 2026** (713 decided claims, 22 frauds), trained on everything before 1 Jun 2026.
- The final model is trained on all decided claims before 1 Jul 2026, with outcomes visible up to the 1 Jul export.

**D-10 Which metric decides.** The client KPI (accuracy) is reported everywhere. It does **not** select the model, because predicting "nothing is fraud" already scores 98.7% over 15 months and 96.9% in June. Selection used PR-AUC, recall and precision within the desk's 40 reviews per month, and rupees, in line with Farhan's "fraud stopped per claim we check, in rupees".

**D-11 Rupee accounting.** Net benefit = fraud amounts stopped − ₹380 × genuine claims held. The ₹260 "blended cost of a service contact" is ambiguous: it could be the cost of every review or only of customer contacts. I show results **both with and without** it. Policy gives no cost for inspections, so I do not invent one.

**D-12 Model choice, locked before the holdout was scored.** Logistic regression (C=0.3) on 28 fields (23 numeric, 5 categorical), with claims after 1 May 2026 weighted ×10.
- Why: on the only post-change dev fold it beat gradient boosting (PR-AUC 0.45 vs 0.24, all 9 frauds inside capacity). It tied on earlier folds, and its reasons are an exact decomposition.
- Grid: C ∈ {0.1, 0.3, 1}, weight ∈ {5, 10, 20}. Results were flat, so I took the middle values.
- Rejected:
  - A reduced 12-feature set (mean dev AP −0.10; I tried it to make reasons cleaner).
  - Gradient boosting as the final model.
  - Ranking the queue by expected loss (fewer rupees on holdout).
  - Dropping Zoho rows.

**D-13 Disclosed change after first holdout run.** Two non-tuning fixes were made after the holdout had been scored once:
- the point-in-time shrinkage prior (a leakage fix);
- a tighter solver tolerance, so results are identical across scipy builds.

Holdout PR-AUC moved 0.532 → 0.533 → 0.529; recall within capacity was unchanged at 14/22. No hyperparameter was changed.

**D-14 Submitted score = probability.** The submission is the model probability, as the brief asks. The desk queue also uses the probability; ranking by probability × amount was tested and did worse on June (net ₹9,044 vs ₹10,263).

**D-15 Not used as features:**
- `city`, to avoid encoding geography (a fairness risk to small-city outlets; Meenal's concern). Outlet record and tenure carry the signal.
- `source`, which is the same as date.
- Raw serial and raw `partner_id`, to avoid memorising IDs. Outlets enter only through their point-in-time record.

**D-16 Service behaviour.** Partner history in the service is the snapshot at export: claims through 30 Sep 2026 and outcomes through 30 Jun 2026.
- Unknown `partner_id`: scored, with a warning.
- Unknown SKU: rejected (422).
- Request bodies are capped at 8 KB.
- The artifact is loaded only if its SHA-256 matches the manifest, because joblib uses pickle.

**D-17 Risk bands.**
- HIGH: score ≥ 0.135, the average monthly "top-40" line on Jul–Sep claims, i.e. what the desk can review.
- MEDIUM: score ≥ 0.034, the monthly top 10%.
- These thresholds use only unlabelled test scores.

**D-18 No LLM in the product.** Scoring and reasons are deterministic, and no API key is needed. An LLM would add no measurable value on 13 fixed fault phrases. It would also be exposed to partner-typed free text, which is untrusted input.

**D-19 `submission-form.md` was not in the pack.** The brief asks for it, but no template was supplied. I created it using the brief's own deliverable list (1–6) as its structure, and it should be pasted into the official form if one exists.

**D-20 Confidentiality.** Nothing is pushed anywhere. `.gitignore` excludes every task-pack file and the model artifact (which embeds outlet history), per policy §10.
