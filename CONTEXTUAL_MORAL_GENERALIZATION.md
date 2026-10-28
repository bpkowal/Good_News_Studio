# Contextual Moral Generalization v0.1

The first increment is an offline terminal-action learning experiment, separate
from the Parliament's operative policy. Run it from the repository root:

```bash
.venv/bin/python -B run_contextual_moral_generalization.py \
  --output diagnostics/contextual_moral_generalization_20261027 \
  --checkpoint-tag contextual-learning-v01
```

Use a new output directory for each experiment. Open its `report.html` directly
in a browser, or read `RESULT.md`. This command uses only Python's standard
library, makes no API calls, and preserves the existing agents, frozen worlds,
validators, viewer and native ledgers. The standalone report is not a new live
Parliament run. No branch, commit or push is performed by the runner.

The next increment focuses on the weight-only mechanism: see
[Weight-only continuation](WEIGHT_ONLY_LEARNING.md) for restored checkpoints,
an identical frozen-model control, and explicit weight-change records. The
original three-arm experiment and its retrieval results remain preserved.

## Frozen experiment

There are 48 synthetic disclosure scenarios, arranged in 12 feature-combination
groups with four surface variants per group. Three responses are available in
every case: disclose, protect, or seek information. The latter is a terminal
response for this one-step experiment, not a modeled sequence of investigations.

The shared cues describe established protective/harmful purpose or unresolved
purpose, coercion, and urgency. They are supplied experimental annotations; the
parser and LLM do not extract them. They are not objective moral measurements.
Case IDs and text never enter policy features. Feature and action identities are
versioned, and each action has one shared feature-weight vector rather than a
separate value for each case. Pairwise cue interactions are included in advance.

Complete cue combinations, including all their wording variants, belong to one
split only: 32 training cases, eight validation cases and eight test cases. The
test split deliberately contains coercive protective urgency and noncoercive
harmful urgency, challenging the shortcut that coercion alone settles disclosure.
The fixed control implements precisely that shortcut. It is a deliberately
limited heuristic; it is not a frozen Parliament, an LLM or a competitive moral
reasoning baseline. Beating it would establish only a small mechanism result.

Three arms receive identical training schedules and common simulator random
draws for seeds 11, 23, 37, 41 and 53:

| Arm | Selection | Learning |
| --- | --- | --- |
| `fixed_heuristic` | Protect under coercion; otherwise disclose | None |
| `learned_values` | Shared contextual values; softmax plus exploration during training | Observed functional outcome only |
| `learned_values_memory` | Same contextual model, blended with nearby observed precedents | Same weight update plus training-only case retrieval |

The normalized linear update is
`w_a += alpha * (observed_outcome - dot(w_a,x)) * x / dot(x,x)`.
The intercept starts at 0.5, other weights at zero. Learning rate is 0.30,
softmax temperature 0.15 and training exploration 0.15. There are 16 shuffled
epochs, or 512 trials per arm and seed. Evaluation chooses greedily from frozen
values. Softmax probabilities describe selection; they are not calibrated
probabilities of moral correctness.

Memory retrieves up to 12 feature-nearest observed training examples per action
from the same arm and seed. Proximity and recency select neighbors, not success.
A distance-dependent blend caps memory influence at 0.25. Both successful and
unsuccessful examples are identified where present, without artificially
balancing them into a 50:50 outcome estimate. No test or validation observation
can enter retrieval, and no counterfactual reward for an unchosen action is
available to it. These choices are fixed before the first experiment; neither
validation nor test results tune this run.

## Feedback and limits

`contextual_moral_simulation.py` specifies the toy contingencies independently
of `behavioral_policy.py`. Its expected functional scores are probabilities of
simulated functional success, not judgments of moral correctness or a complete
model of death, rescue, persecution or disclosure consequences. Known protective
purpose favors disclosure; known harmful purpose favors protection; unresolved
purpose generally favors investigation, with delay costs under urgency.
Coercion alone does not determine the payoff. This encodes experimental values;
learning cannot independently justify those values.

Feedback retains three separate quantities:

- `outcome_score`: the sampled observation, with simulator version, action and
  random draw as evidence. Only this updates weights.
- `decision_score`: expected functional success under the stipulated model,
  used for evaluation. This is not a certificate of normative defensibility.
- `process_score`: whether unresolved purpose was addressed by investigation.
  This is a predefined procedural proxy, not general reasoning quality.

A favorable expected decision can have an unlucky observed outcome; both remain
in the record. Evaluator confidence is 1 only in the narrow sense that the
specified simulation was calculated, not that moral truth was measured. Outcome,
decision and process scores are not collapsed into a combined reward. Feedback
reassessment is not yet implemented: this first experiment preserves the original
feedback and its provenance rather than silently revising past rewards.

Report expected functional score, observed success, regret, toy-optimal action
rate and process separately, including each held-out context's chosen actions.
Seed variation and learning curves are descriptive. Four surface variants do
not create four independent ethical contexts; the test has only two independent
cue configurations in one domain. Do not infer broad transfer, normative
improvement, improved agent arguments or superiority over Parliament from this
experiment. New actions, missing-cue discovery, deeper-deliberation recruitment
and live policy integration remain later increments.

## Records and integration boundary

The SQLite `experiences` journal records selections, available actions, cue
vectors, selection probabilities, observed outcomes, evaluator provenance,
retrieval references and the weights before and after each trial. The policy
update and receipt commit in one transaction; failure restores the in-memory
weights. Checkpoints persist the initial, trained, validation and test models.
Held-out observations remain inspectable but cannot update weights or be used as
precedent. Each arm and seed has isolated weights and case memory.

The runner freezes `design.json`, `cases.freeze.json` and implementation hashes
before training. Existing source archives, run manifests and checkpoint receipts
track the new `behavioral_simulation` path as `SIMULATION_NO_API`. Existing
Parliament execution paths retain their identities. A fresh output directory is
required so that reruns cannot overwrite prior evidence.

Outputs include `results.json`, `evaluation.csv`, `evaluations.json`,
`learning_curves.json`, `policy_checkpoints.json`, `controls.json`,
`experience.sqlite3`, `RESULT.md`, `report.html` and the execution manifest.

Before using this after a real Parliament terminal decision, we would still need
an explicit mapping to admitted action IDs, supported cue evidence, externally
sourced feedback, permission to make a selection operative, and a matched fixed
Parliament baseline. This experiment does not infer that a provisional agent
preference or a topography probability is an authorized final decision.
Behavioral “topography” here means the selected response; it is distinct from
the existing Care/welfare/obligation 3D presentation.

## Credit

The proposal's memory-based learning precedent is Shinn et al.,
[Reflexion: Language Agents with Verbal Reinforcement Learning (2023)](https://arxiv.org/abs/2303.11366).
That work uses verbal reflection and episodic memory to influence later choices
without updating LLM weights. This pilot uses a conventional contextual value
update and observed-case retrieval; it does not implement Reflexion's verbal
reflection loop, reproduce its results, or establish its applicability to ethics.
No new third-party library or model training infrastructure is required.
