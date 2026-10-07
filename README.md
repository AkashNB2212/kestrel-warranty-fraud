# Kestrel Home — warranty-claim fraud flag (Variant C)

This project scores each warranty claim for fraud risk **before payout**, so the investigation desk spends its 40 checks a month on the riskiest claims. Each score comes with reasons a Kestrel employee can read. There is no paid API and no LLM: a local scikit-learn model, a FastAPI endpoint and a one-screen Streamlit app.

| Deliverable | File |
|---|---|
| Predictions for `test_unlabelled.csv` | `predictions.csv` (`claim_id,score`, 2,252 rows, sample order) |
| Service | `service/api.py` (POST `/predict`), `service/app.py` (screen) |
| Evidence | `VALIDATION.md`, `reports/`, `tests/` (37 tests) |
| Memo | `MEMO_Ritu_Deshpande.md` |
| Decisions | `DECISIONS.md` |

## Architecture

```
task-pack CSVs ──► kestrel_fraud/data.py      clean, de-duplicate re-submissions, normalise categories
                   kestrel_fraud/features.py  claim fields + point-in-time outlet history (leak-free)
                   kestrel_fraud/model.py     logistic regression pipeline (impute/scale/one-hot → LR)
scripts/train_final.py ─► artifacts/model.joblib + manifest.json (SHA-256) + predictions.csv
service/api.py  (FastAPI)  POST /predict ─► kestrel_fraud/scoring.py ─► score, band, action, reasons
service/app.py  (Streamlit) ─HTTP─► /predict
```

## Data

The task-pack files are **not** included in this repository (ops-policy §10). Put them in the project root, next to this README:
`train.csv test_unlabelled.csv partners.csv products.csv sample_submission.csv`
(or set `KESTREL_DATA_DIR=/path/to/pack`).

## Clean-machine setup (macOS / Linux, Python 3.10–3.12)

```bash
python3 -m venv .venv
```
```bash
source .venv/bin/activate
```
```bash
pip install -r requirements.txt
```

## Reproduce everything

```bash
python scripts/audit.py
```
```bash
python scripts/evaluate.py
```
```bash
python scripts/partner_analysis.py
```
```bash
python scripts/train_final.py
```

* `audit.py` → `reports/data_audit.md`: data quality and leakage audit.
* `evaluate.py` → `reports/evaluation.json`, `reports/holdout_errors.csv`: chronological folds, the June holdout, baselines, challengers, calibration, error analysis and sensitivity checks.
* `partner_analysis.py` → `reports/partner_hypothesis.json`: the new-partner analysis.
* `train_final.py` → `predictions.csv`, `artifacts/model.joblib` and `artifacts/manifest.json`. It must run before the service starts.

Training takes under a minute on a laptop. Results are deterministic: a fresh venv reproduced `predictions.csv` byte-for-byte.

## Tests

```bash
python -m pytest -q tests
```

The 37 tests cover:
- schema and target checks, re-submission de-duplication, normalisation;
- leakage guards (future claims or outcomes cannot change features, chronological folds);
- pipelines handling unseen categories and missing values;
- capacity and rupee metrics;
- `predictions.csv` against the sample;
- API happy path, malformed input, oversized payload, missing model, tampered artifact;
- an exact-decomposition check of the reasons.

## Run the service

Terminal 1 (API):
```bash
uvicorn service.api:app --port 8000
```
Terminal 2 (screen, opens http://localhost:8501):
```bash
streamlit run service/app.py
```
Set `KESTREL_API_URL` if the API is not on `http://127.0.0.1:8000`. Interactive API docs are at http://127.0.0.1:8000/docs.

Example request:
```bash
curl -s -X POST http://127.0.0.1:8000/predict -H 'content-type: application/json' -d '{"submitted_at":"2026-09-15 11:00","partner_id":"SP3160","sku":"KH-AF-02","product_serial":"KH123456789","days_since_purchase":120,"claim_amount_inr":1940,"photo_attached":"Y","partner_inspected":"N","claim_description":"motor not running","customer_prior_claims":0}'
```
The response contains `fraud_score` (0–1), `risk_band` (HIGH/MEDIUM/LOW), `suggested_action`, 1–4 `reasons` (each "raises risk" or "lowers risk", computed from the model and the claim), `warnings`, and model metadata.

The service fails politely in each of these cases:
- invalid fields → 422 with a field list;
- body over 8 KB → 413;
- unknown SKU → 422;
- unknown outlet → scored, with a warning;
- model missing or checksum mismatch → 503 "run `python scripts/train_final.py`";
- any other error → 500 without a stack trace.

## The model

* **Logistic regression** (C = 0.3) with claims after 1 May 2026 weighted ×10. Training data: 11,146 decided, de-duplicated claims up to 30 Jun 2026 (141 fraud).
* **Inputs:**
  - claim: amount, amount ÷ list price, product line, days since purchase, customer's prior claims, photo, inspection sign-off, inspector-note category, fault category, serial format;
  - policy route: on or after 1 May and under ₹2,000 (auto-approved), just under ₹2,000;
  - outlet tenure at claim time;
  - outlet activity in the last 30/90 days;
  - outlet's confirmed-fraud record, using only outcomes closed ≥30 days before the claim and before the export.
* **June 2026 out-of-time holdout:**
  - PR-AUC 0.53, ROC-AUC 0.94;
  - with 40 checks/month: 14 of 22 frauds caught, 35% precision, ₹10,263/month net;
  - accuracy 95.2% at capacity, 97.5% at a 0.5 threshold; the "flag nothing" baseline is 96.9%.

  See `VALIDATION.md`.
* **Reasons:** an exact log-odds decomposition of the linear model, grouped into plain-language sentences. A group is shown only when its direction matches the claim's actual value. No text generation is involved.

## AI and build tools used

Claude Code (Claude Opus 5.5) in the Claude desktop app was used for exploration, code, tests, docs and verification. There are no other tools and no API spend, and nothing in the product calls a model API. Discarded along the way: gradient boosting as the final model, a reduced feature set, ordering the queue by rupee size, dropping legacy Zoho rows, `city` as a feature, and any LLM (see `DECISIONS.md`, D-12, D-15 and D-18).

## Known limitations

* **Cold start.** Outlets with no investigated claims are scored mostly on route, amount and tenure; 4 of 8 June misses were such outlets.
* **Stale outlet records.** No outcomes after 30 Jun 2026 were available, so the service's outlet record is frozen at that date. Retrain monthly.
* **Regime changes.** A model trained before 1 May missed every May fraud. Retrain immediately after any policy change.
* **Small samples.** 22 frauds in the holdout means PR-AUC has a 90% interval of 0.37–0.70.
* **Label noise.** Legacy Zoho "0" labels may hide undecided cases. The 30-day closure lag is an assumption, because the export has no closure dates.
* **Rupee economics.** Claim-by-claim checking of small claims roughly breaks even if each check costs the ₹260 service contact. The outlet-level action carries most of the value.
* **Security.** The artifact is a pickle (joblib). It is only loaded when its SHA-256 matches the manifest the training script writes, so never load artifacts from elsewhere.
