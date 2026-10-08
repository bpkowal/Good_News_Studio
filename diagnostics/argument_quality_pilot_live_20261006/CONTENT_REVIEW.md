# Argument-quality findings

The checklist did not demonstrate an overall argument-quality improvement in
this bounded pilot. It produced a few useful local changes, alongside a failed
Care ledger submission. This is one sample per cell, assessed by one reviewer;
it is not a reliable comparison of prompt effect sizes.

All 18 native submissions were schema-valid. The three inputs passed fresh
native world admission. The [controls](controls.json) confirm identical shared
inputs, framework commitments, model, token ceiling and one adapter call per
cell within each comparison. No response was repaired or normalized.

| Source variant | Standard U / D / Care | Checklist U / D / Care |
| --- | --- | --- |
| Original probabilities | Ben / Ben / Ben | Ben / Ben / Ben |
| Probabilities swapped | Cara / Cara / Cara | Cara / Cara / Cara |
| Original probabilities plus promise to Cara | Ben / Cara / Ben | Ben / Cara / Ben |

These are specialist recommendations, including provisional ones, not a
collective Parliament verdict. Responding to counterfactuals already works in
the standard arm. Agreement in the original case is not evidence of fixation.

## Content scoring

Scores follow the [pilot rubric](../../ARGUMENT_QUALITY_PILOT.md). They assess
visible arguments, not moral truth or agreement with the reviewer's verdict.

| Dimension | Standard | Checklist | Evidence |
| --- | --- | --- | --- |
| Restricted arithmetic, utilitarian | 1 | 1 | Correct action-by-action 0.95/0.05 calculation and swap, but neither explains the limits of survival count as a welfare endpoint. |
| Duty reasoning, deontology | 1 | 1 | Duties and priorities are named. The inference from imperfect beneficence to requiring the best-probability rescue remains insufficiently defended. Promise fidelity is classified negative under violation and positive under fulfilment without explaining the classification change. |
| Objection handling, utilitarian | 0 | 0 | Promise case retains survivor maximization; no substantive treatment of promise/trust consequences or their unknown welfare significance. |
| Objection handling, deontology | 1 | 1 | Competing rescue and promise duties are named; priority is mostly asserted as perfect over imperfect. |
| Objection handling, Care | 1 | 1 | Both recognize unmet competing needs and the promise. Neither fully defends the effectiveness/trust tradeoff. |
| Counterfactual sensitivity, all three frameworks | 2 | 2 | Probability changes reverse recommendations; the promise changes deontology's governing duty and recommendation and changes Care's relational reasons while retaining its preference. |

The checklist's useful local changes should not be lost:

- In the probability-swap case, deontology calls the less effective rescue
  permissible rather than prohibited. It still declares the more effective
  rescue required without fully defending that stronger conclusion.
- In the promise case, Care's raw ledger says treatment of Ben strains trust
  and treatment of Cara preserves it. The standard ledger paradoxically labels
  Cara's promise-honouring action `BETRAYS`. This is a concrete argument-record
  improvement, though insufficient to establish an overall gain.
- Promise-case deontology changes `DOING_HARM` to omission/withheld-benefit
  descriptions. It still uses a perfect negative-duty classification for the
  omission without establishing the separate basis native review requires.

## Separate technical and substantive failures

**Arithmetic:** all six utilitarian responses calculate the restricted survivor
endpoint correctly. Native validation nevertheless reports that the numeric
EV cannot be recomputed from E2/E4. This is a count/evidence accounting gap,
not evidence that multiplication failed. It must be resolved as a restricted
patient-count projection; no welfare magnitude or duration should be invented.

**Duty reasoning:** baseline standard deontology is provisional; checklist
baseline deontology commits. That admission improvement does not defend its
moral inference. Promise-case deontology remains provisional in both arms.
Source-only promise recognition and the harm/duty classifications are separate
problems. The promise is explicit in the source but is not projected into the
medical world graph by this experiment.

**Care consistency:** standard promise-case Care commits with uncertainty.
Checklist Care's raw reasons improve trust direction, but it reports `SELECTED`
while both relational assessments remain `CONTESTED`. Native review rejects the
ledger with `unresolved Care ranking must remain provisional`. The underlying
argument is saved; successful submission must not require discarding it.

## Next bounded changes supported by this experiment

1. Provide a reproducible expected-survivor calculation from explicit patient
   identity/count and probability, separately from unknown total-welfare EV.
2. Route the explicit promise into supported non-outcome evidence so native
   reasoning can cite it without inventing a health effect.
3. Give specialists an inspectable argument work record before compact ledger
   submission: premise, competing claim, priority bridge, objection response,
   and reversal condition. Preserve useful arguments when the ledger is
   inconsistent, and compile the submission from that same record.

Do not add blueprint families, force framework disagreement, or weaken native
review to manufacture progress. The next comparison should measure these
specific improvements and repeat the cells before claiming a dependable gain.

Raw responses and native records are under each case/arm/framework directory.
The [results](results.json) retain recommendation, validation errors and argument
fields; the [run table](RESULT.md) reports admission separately.
