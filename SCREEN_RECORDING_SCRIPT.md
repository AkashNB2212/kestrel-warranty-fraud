# Screen recording script (target 2:45, hard limit 3:00, no slides)

Before recording, have the API (`uvicorn service.api:app --port 8000`) and the screen (`streamlit run service/app.py`) running, a terminal in the project root, and these files open in tabs: `reports/data_audit.md`, `VALIDATION.md`, `predictions.csv`.

| Time | Show | Say (roughly) |
|---|---|---|
| 0:00–0:20 | `email-thread.txt`, then policy §5 | "The ask was 97% accuracy. Two lines in the pack change the job: the desk can check only 40 claims a month, and from 1 May claims under ₹2,000 are paid without inspection." |
| 0:20–0:45 | `reports/data_audit.md`: monthly table | "Only 1.3% of claims are fraud, so flagging nothing is already 98.7% accurate. The audit also found 681 re-submitted claims, which I de-duplicated, 215 undecided claims, which I excluded, and Zoho rows where undecided was stored as 0. Look at May: inspection falls from 93% to 22% and fraud triples." |
| 0:45–1:15 | `VALIDATION.md` §4 (stability table) | "What I tried first: logistic regression and gradient boosting on chronological folds. Good before May, but on May itself **worse than random**, because the fraud pattern flipped from large inspected claims at old franchises to small uninspected claims at new outlets. What I changed: policy-derived features and extra weight on post-change claims. I locked that choice on a late-May fold *before* scoring the June holdout." |
| 1:15–1:40 | `VALIDATION.md` §3 table | "June holdout: 14 of 22 frauds caught within the 40 checks, against at most 5 for the simple rules, and ₹10,263 a month net. 97.5% accuracy is reachable at a 0.5 threshold, but it catches only 6 frauds, which is why I'm recommending a different board metric." |
| 1:40–2:00 | `VALIDATION.md` §7 | "Ritu's new-partner hunch: true only since May, and it is five outlets out of 45. Before May, new outlets were the cleanest." |
| 2:00–2:30 | Streamlit: load the "new outlet" example → Score; then the established example → Score | "The service: one endpoint, one screen. A ₹1,940 uninspected claim at an outlet with 10 confirmed frauds scores about 67%, HIGH, hold for the desk, with the reasons. A large inspected claim at a clean outlet scores LOW." |
| 2:30–2:45 | Terminal: `python -m pytest -q tests` (37 passed) | "What I threw away: gradient boosting, a slimmer feature set that hurt accuracy, ranking by rupee size, and any LLM. A leakage test I wrote caught a real leak, which I fixed and disclosed. A clean environment reproduces the predictions byte-for-byte." |
