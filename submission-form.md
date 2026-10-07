# Submission form — Kestrel Home Warranty Claim Review (Variant C)

> **Note:** the brief asks for `submission-form.md`, but it was not in the task pack I received. This form uses the brief's own deliverable list (1–6) as its structure. If an official template exists, these answers can be copied into it field by field.

**Candidate:** _(name as on application)_ — akashnileemborgohain412@gmail.com
**Date:** 7 Oct 2026

---

## 1. Predictions

* **File:** `predictions.csv`. Columns `claim_id,score`, 2,252 rows (one per `test_unlabelled.csv` claim), same order as `sample_submission.csv`.
* **Scores:** no missing scores; range 0.0000–0.9797; higher = more likely fraud.
* **What the score is:** the logistic-regression probability that the claim is fraud.

**What I expect it to score, and why:**

| If scored by | Expect | Range |
|---|---|---|
| PR-AUC / average precision | **0.50** | 0.35–0.65 |
| ROC-AUC | **0.92** | 0.88–0.95 |
| Recall of fraud in each month's top 40 | ~60% | 45–75% |
| Accuracy at a 0.5 cut-off | 97–98% | the "flag nothing" baseline is ≈97% |

**Why:**
* **Basis.** The June 2026 out-of-time holdout is the same post-1-May regime and was trained just before the period: PR-AUC 0.53 (bootstrap 90% CI 0.37–0.70), ROC-AUC 0.94, 14 of 22 frauds inside 40 reviews. Earlier folds bracket it (late May 0.45/0.98; Jan–Apr 0.45–0.75/0.80–0.93).
* **Why slightly lower than June:**
  - outlet records go stale (no outcomes after 30 Jun);
  - 11 outlets onboarded after June have no record;
  - fraudsters can adapt.
* **Assumptions:** test fraud rate ≈ June (~3%), auto-approval still in force, and labels from the same desk.

## 2. Working service

* **API:** `uvicorn service.api:app --port 8000`, then `POST /predict` with one claim as JSON. It returns `fraud_score`, `risk_band` (HIGH/MEDIUM/LOW), `suggested_action`, 1–4 `reasons`, `warnings` and model metadata. There is also `GET /health`.
* **Screen:** `streamlit run service/app.py`. It is a single form that calls the API and shows score, level, action and reasons.
* **Clean machine:** `python3 -m venv .venv && pip install -r requirements.txt && python scripts/train_final.py`, then the two commands above (README.md). No API key is used anywhere. Errors return polite JSON:
  - 422 for invalid fields or unknown SKU;
  - 413 for oversized bodies;
  - 503 "run train_final.py" if the model is missing or tampered with;
  - no stack traces.

## 3. Evidence that it works — and how often it does not

* **`VALIDATION.md`:**
  - chronological folds plus a June holdout scored once;
  - baselines and challengers;
  - confusion matrices, threshold sweep and calibration;
  - error analysis with named cases;
  - robustness checks and the expected-score reasoning.
* **How often it does not work (June):**
  - it misses 8 of 22 frauds;
  - 26 of its 40 flags are genuine claims;
  - it fails completely on a policy change it has not seen: on May 2026 its ROC-AUC was 0.11.
* **`tests/`:** 37 automated tests, all passing, covering schema, leakage guards, temporal splits, the submission, the API including malformed input, and an exact check of the explanations.
* **Supporting reports:**
  - `reports/data_audit.md`, `evaluation.json`, `holdout_errors.csv`, `partner_hypothesis.json`;
  - `DECISIONS.md` (20 recorded decisions).
* **Reproducibility:** a fresh virtualenv reproduced `predictions.csv` and every metric byte-for-byte.

## 4. Memo to Ritu

`MEMO_Ritu_Deshpande.md`, one page. The decision: use the flag to fill the desk's 40 monthly checks, and change the board metric away from accuracy. The number: 14 of 22 June frauds caught, 1 in 3 checks is fraud. The rupees: ₹20,143 stopped, ₹10,263/month net. Next week:
- switch off auto-approval at the five outlets behind 29 of 34 new-outlet frauds;
- start the queue;
- get the inspection cost from Finance;
- close the 215 open cases;
- retrain monthly.

## 5. Screen recording

The plan is `SCREEN_RECORDING_SCRIPT.md` (2:45): what I tried (LR and gradient boosting), what broke (May regime flip), what I changed (policy features and recency weighting, locked before the holdout), what I threw away, and a live demo with tests. *The video itself has to be recorded by the candidate.*

## 6. AI tools — what I used, what it cost, what I discarded

* **Used:** Claude Code (Claude Opus 5.5) in the Claude desktop app, for exploration, code, tests, documentation and browser verification of the screen.
* **Cost:** no API keys or metered spend; covered by the candidate's Claude plan.
* **In the product:** no AI or LLM. Scoring and reasons are deterministic and run locally.
* **Discarded:**
  - gradient boosting as the final model;
  - a reduced feature set (−0.10 PR-AUC);
  - an expected-loss queue order;
  - dropping legacy rows;
  - `city` as a feature;
  - any LLM for reasons or free text (no measurable value, and it would read untrusted partner text).
* **What AI got wrong and was caught:**
  - contradictory first-draft reasons;
  - a subtle prior leak (caught by a test);
  - a scipy reproducibility issue;
  - an overstated memo claim.

  All are logged in `PROMPT_EVOLUTION.md`.

## Decisions I had to make without anyone to ask

See `DECISIONS.md`. The main ones:
1. Accuracy is reported but does not select the model.
2. Undecided claims are excluded; Zoho zeros are kept, with a sensitivity check.
3. Re-submissions are reduced to the first one.
4. Outlet fraud history assumes a 30-day investigation lag.
5. The ₹260 service contact is shown with and without.
6. No LLM in the product.
