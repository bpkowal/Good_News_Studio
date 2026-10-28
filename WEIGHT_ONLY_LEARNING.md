# Weight-only continuation

This increment keeps the contextual action-value learner and leaves case
retrieval off. It restores the previous experiment's learned checkpoints;
learning no longer needs to start from initial weights on each run.

```bash
.venv/bin/python -B run_weight_update_comparison.py \
  --output diagnostics/weight_only_continuation_20261028
```

The default parent is the frozen October 27 simulation. The command validates
its manifest, loads all five seeds from the `learned_values` arm, and gives each
seed two identical starting copies. One remains frozen. The other receives 512
additional normalized weight updates, using the same learning rate, feature
schema, action IDs and outcome signal as before. Both arms get matched case
schedules and random outcome draws. They may choose different actions as their
policies diverge; this is not an experiment with identical observed rewards.

The experiment is offline and makes no API calls. Only sampled outcomes from
training cases change weights. Decision and process scores remain separate.
Neither arm updates during evaluation. No case retrieval is performed or stored,
and no monitoring result selects a best checkpoint or tunes a parameter.

The parent policy files, original experiment, Parliament agents, worlds and
operative decisions remain unchanged. New policy versions are saved in the new
directory. The report labels the old validation/test scenarios as **reused
monitoring cases**: they cannot support a claim of fresh generalization after
their earlier results have been inspected.

## Inspect and continue

Open the output's `report.html`, or read `RESULT.md`. It compares the two arms
directly and includes failures or regressions. Further updates are not guaranteed
to help: noisy feedback and a constant learning rate may move estimates away
from a good ranking as well as toward one.

`weight_changes.csv` contains each action/feature weight before and after
training. `weight_explanations.json` shows each cue's weighted contribution to
the raw value prediction, along with the clipped value. These explain the
software calculation; they are not philosophical warrants or calibrated moral
probabilities. `evaluation.csv` lists every monitored choice.

Each new comparison also writes `failure_review/report.html`, with filters for
finding, arm, phase, case and action. It preserves one self-contained JSONL
record per flagged experience: exact case text, original evidence/feedback,
selection probabilities, weights before and after, value contributions, and
links to matched frozen-control measurements where applicable. The CSV index
and grouped counts provide search and denominators without reading the full
training journal.

The review keeps four findings separate: zero observed outcomes, positive
simulator regret in monitoring, regression against the frozen copy, and a zero
procedural-proxy score. Findings can overlap. A good expected choice followed
by a bad outcome is not automatically labeled suboptimal; a lucky poor choice
can still be flagged. Training choices get no invented counterfactual regret.
Non-greedy sampled training responses are explicitly identified. These records
support later human investigation; they do not automatically alter feedback,
retrain the policy, or add failed cases to retrieval.

To inspect an existing comparison without running more training:

```bash
.venv/bin/python -B run_weight_update_comparison.py \
  --review-only \
  --from-run diagnostics/weight_only_continuation_20261028 \
  --output diagnostics/weight_only_failure_review_20261028
```

Use a fresh output directory. The source comparison is opened read-only and
its manifest verifies before review. Source hashes and original execution ID
remain with the archive. The review's own manifest records
`SAVED_REVIEW_NO_API` / `behavioral_failure_review`, zero model calls and zero
policy updates. The existing comparison's report and checkpoints are preserved.

`experience.sqlite3` journals the selected actions, observed feedback, exact
weight updates and initial/trained/evaluated checkpoints. `controls.json`
records matched initial hashes, schedules and update counts. `design.json`
freezes the parent run ID, parent file hashes and comparison settings before
the first new trial. Existing run manifests and resource snapshots preserve
the parent files and source version used by the continuation.

To continue from the updated checkpoint of this comparison, choose another
fresh directory:

```bash
.venv/bin/python -B run_weight_update_comparison.py \
  --from-run diagnostics/weight_only_continuation_20261028 \
  --output diagnostics/weight_only_next
```

The loader continues the predeclared `updated_weights` arm, never whichever
seed or arm looked best on evaluation. Changed action/feature identities are
rejected rather than silently remapped. This is still the disclosure simulator
with supplied contextual annotations and specified rewards. A real Parliament
connection needs supported cue/action mappings and independent feedback before
learned weights can influence an operative selection.
