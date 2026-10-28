# Weight-only failure review

1940 flagged experiences out of 5280 recorded selections.
Records preserve the original feedback, weights before/after, case text and feature contributions.
Review makes no policy updates or retrieval calls. It does not replace the original experiment.

| Finding | Count | Meaning |
| --- | --- | --- |
| observed_outcome_failure | 1437 | Observed functional outcome was zero; this alone does not establish a bad choice. |
| monitoring_suboptimal_choice | 16 | Recorded monitoring regret was positive under the stipulated simulator. |
| monitoring_regression | 4 | Updated policy had lower recorded expected success than its matched frozen control. |
| process_proxy_zero | 1024 | The recorded procedural proxy was zero; this is not a general ethics verdict. |

Kinds overlap; their counts must not be added to obtain a failure total.
A zero observed outcome can follow a good expected choice. A lucky outcome can hide a poor expected choice.
Training records are not assigned counterfactual regret. Monitoring findings use only the original saved evaluator measurements.
Non-greedy training selections are explicitly marked; not every such failure is a ranking error.

[Browse and filter every finding](report.html) · [Index CSV](failure_index.csv) · [Full records](failures.jsonl) · [Counts and source hashes](summary.json)
