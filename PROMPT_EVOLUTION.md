# PROMPT_EVOLUTION — how AI was used, what changed, what was thrown away

**Tool:** Claude Code (model: Claude Opus 5.5) in the Claude desktop app, in one working session. It had local file, shell and in-app-browser tools. No other assistant, model API or agent framework was used. **Inside the product: no AI or LLM.**

**Cost:** no API keys and no metered spend. The session ran on the candidate's Claude plan, and token usage was not separately metered.

## The instruction
One long brief from the candidate. It set the role (lead ML engineer), made the task pack the source of truth, and laid out phases: inspect → audit and leakage → temporal evaluation → metrics → simple models → partner hypothesis → rupees → predictions → explanations → service → tests → docs. It also set hard limits: no deep learning, no LLM classification, no random split as primary, no hard-coding, no pushing to GitHub.

## What happened, in order (including mistakes)

| # | Step | What changed as a result |
|---|---|---|
| 1 | Read README, email thread, policy PDF (text extracted with pdfplumber), every column | Found the 40/month review cap, ₹380 / claim-amount / ₹260 costs, **1 May 2026 auto-approval below ₹2,000**, Zoho blanks stored as 0, re-submissions. The metric plan became "capacity-constrained" instead of plain accuracy. |
| 2 | Data audit | 681 duplicate claim IDs (identical except timestamp) were de-duplicated to the first submission. 215 undecided rows excluded. Inspection rate fell from 93% to 22% in May while fraud tripled. Five descriptions had text appended after the fault phrase, so free text was restricted to the leading fault phrase. |
| 3 | First models (LR, gradient boosting) on dev folds | Strong before May (PR-AUC 0.45–0.75). **Worse than random on May 2026 (ROC-AUC 0.1–0.3).** This was the key discovery: the fraud pattern flipped from large inspected claims at old franchises to small uninspected claims at a few new outlets. |
| 4 | Added policy-derived regime features + recency weighting; added a late-May dev fold | Recovered on late May (ROC-AUC 0.98, 9/9 frauds within capacity). LR beat gradient boosting on the post-change fold, so **LR was chosen and locked before the June holdout was scored.** |
| 5 | Small grid (C × weight, 9 combos) | Flat, so I took the middle values (C=0.3, ×10) rather than chasing a 9-fraud fold. |
| 6 | Holdout scored once | PR-AUC 0.53, 14/22 caught within 40 reviews. Reported all challengers too. |
| 7 | API smoke test showed **contradictory reasons** ("outlet is new" both raising and lowering risk; "clean record raises risk") | Tried a reduced 12-feature model to remove collinearity, but it lost 0.10 dev PR-AUC and was **discarded**. Instead, reasons are grouped into plain-language stories and shown only when the direction matches the claim's value. A test asserts that the decomposition is exact. |
| 8 | Writing the leakage test **caught a real leak**: the shrinkage prior was one global fraud rate computed up to the export date | Made the prior point-in-time. Holdout moved by 0.001 (disclosed in DECISIONS D-13). |
| 9 | Clean-venv rebuild: predictions differed slightly; statsmodels crashed | Cause was scipy 1.17. Pinned scipy and tightened the LR tolerance. Rebuild then matched **byte-for-byte**. |
| 10 | Memo fact-check | The first draft said fraud is "often at exactly ₹1,995"; the check found 8 of 36. Rewrote it to the true and more useful finding: ₹1,995 went from 0 claims before May to 87 since. |
| 11 | Browser check of the Streamlit screen | The in-app preview launcher could not open the Documents folder, so the servers were started from the shell and the screen was driven in the browser pane (score 67.3%, HIGH, 4 reasons; re-checked after the final retrain). |

## Discarded (and why)
* **Gradient boosting as the final model.** Better on two pre-May folds, worse on both post-change folds, and less explainable. Kept as a reported challenger.
* **Reduced feature set.** Cleaner coefficients, but −0.10 dev PR-AUC.
* **Ranking the desk queue by probability × amount.** Less rupee benefit on June (₹9,044 vs ₹10,263).
* **Dropping legacy Zoho rows.** Slightly better on June, worse on Jan–Feb; within noise.
* **`city` as a feature.** Geographic proxy, and a fairness risk to small-city outlets.
* **An LLM for reasons or for reading the free text.** It would add no accuracy, require a key, and expose the product to partner-typed free text.

## What the AI did *not* decide
The business framing was set by the candidate's brief and the pack: capacity-based evaluation, refusing to sell 97% accuracy, and the policy-driven features. AI drafted the code and documents. Every number in the documents was re-read from the generated `reports/` files after the final run.
