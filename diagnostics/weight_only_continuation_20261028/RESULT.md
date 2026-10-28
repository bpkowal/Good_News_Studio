# Continued weight learning versus identical frozen weights

Both arms resumed exactly the same trained checkpoints. Only one received additional weight updates.
Retrieval is disabled. No Parliament agent, admitted world or operative selection changed.

**These are reused monitoring cases, not a new held-out generalization claim.**

| Monitoring split / arm | Expected functional score | Toy-optimal choices | Regret | Observed success |
| --- | --- | --- | --- | --- |
| validation / frozen_weights | 0.7485 | 70.0% | 0.0765 | 0.675 |
| validation / updated_weights | 0.7935 | 90.0% | 0.0315 | 0.725 |
| test / frozen_weights | 0.9500 | 100.0% | 0.0000 | 1.000 |
| test / updated_weights | 0.9500 | 100.0% | 0.0000 | 1.000 |

Each pair had the same initial weights, case schedule, random outcome draws and training budget.
The policies may choose different actions, so their observed experience differs after they diverge.
Only the selected action’s sampled outcome updates weights; no oracle alternative enters learning.
Frozen controls do not update. Neither arm updates during monitoring. No best checkpoint is chosen from monitoring scores.

Improvement is not guaranteed: constant-rate updates from noisy outcomes can also cause regressions.
Scores describe the specified simulator. They do not establish better Parliament reasoning or moral correctness.

[Browser report](report.html) · [paired results](results.json) · [weight changes](weight_changes.csv) · [weight explanations](weight_explanations.json) · [checkpoints](policy_checkpoints.json)

Continue again from the updated arm, using a fresh output directory:

```bash
.venv/bin/python -B run_weight_update_comparison.py \
  --from-run diagnostics/weight_only_continuation_20261028 \
  --output diagnostics/weight_learning_next
```
