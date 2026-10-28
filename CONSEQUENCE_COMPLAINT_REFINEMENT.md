# Consequences before complaint comparison

Status: implemented and tested with four live replies. The
[content review](diagnostics/consequence_complaint_live_20261026/CONTENT_REVIEW.md)
finds a local factual gain in the blanket case, with unresolved normative bridges.

The subsequent [critical-question increment](CONTRACTUALIST_CRITICAL_QUESTIONS.md)
adds structured open questions to fresh follow-ups. The four-call results below
retain their earlier instructions and schema; they have not been rewritten or rerun.

This increment supplies the existing person-by-option positions directly to an
argument follow-up and asks it to correct factual mistakes before defending its
comparative moral premise. It extends the saved-complaint experiment in
[OPPOSING_COMPLAINT_PROBE.md](OPPOSING_COMPLAINT_PROBE.md), not Parliament's world
schema, roster or admission rules.

## Reuse, rather than another reconstruction

`complaint_consequences.py` projects the saved argument's `burdens` and
`burden_positions` across every source option. It reuses
`complaint_comparison.derive_position_changes`, preserving quantitative descriptions
when a risk is PRESENT in both options. It includes exact cited source spans,
original JSON pointers, claimed factual grounds and binding findings. All rows
retain their identified person and option. Missing/duplicate cells stay UNKNOWN;
unassigned positions remain visible. No new severity, probability, entitlement or
moral ranking is computed.

The matrix is still a **fallible model annotation**, not verified source truth.
Its burden labels may already contain judgments such as “betrayal.” Those stay
explicitly proposed labels. A normative ground does not become a factual premise
because it cites text. Exact quotations establish traceability, not entailment.
The source text takes priority if a row is wrong. This limitation is part of the
model input as well as the report.

The reply adds two small fields to the previous response format:

- `factual_review`: exact target in the prior argument, assigned complaint or
  consequence record; proposed status (supported, contradicted, not established,
  normative rather than factual, or conditional); correction and source references.
- `normative_warrant`: the priority principle, its justification and its limits.

The existing answer, countercomplaint, changed inferences, conclusion and remaining
objection are preserved. A mistaken factual premise does not erase the underlying
person's complaint. The actual assurance is distinct from a future policy that
would qualify promises. Consent can change the warrant without erasing a burden.
An identified recipient's probability is not a generic person's probability.

Binding checks annotate these new fields; they do not decide whether the correction
is true, reject an otherwise readable reply, authorize a world update or retire an
objection. There is no automatic morally correct winner.

## Bounded comparison, specified before calls

Reuse the exact original arguments and frozen complaints from
`diagnostics/opposing_complaint_live_20261025`. Do not regenerate the selector or
feed it the earlier content review. Neither reply receives the previous arms'
answers, peer arguments, an intended answer or an assistant-authored correction.

Default live subset: cooling with Sal's informed consent, and blanket with the
prior assurance. Two fresh calls per case, **four total**, with no retries:

1. `focused_baseline`: original focused response instruction and source packet.
2. `consequences_first`: same source packet plus the consequence projection and
   instruction to correct factual premises before comparing complaints.

Both arms use the same extended response schema, original framework instruction,
model (`o3`), low reasoning, and 8192-token ceiling. The schema itself solicits
factual review even in the baseline; this is not the old experiment's exact
baseline. The treatment combines structured input and instruction, so this test
does not isolate their individual effects. One sample per condition/case is
exploratory. Calls, refusals, truncations and failures remain saved.

Use a fresh output directory:

```bash
.runtime/venv/bin/python run_consequence_complaint_probe.py \
  --output diagnostics/consequence_complaint_live_20261026 --execute
```

Without `--execute`, this freezes inputs offline and makes no API calls. `--case`
can select a subset from the original frozen experiment; the mechanism contains
no case-specific moral priority or authored factual correction.

## Assessment

Compare actual objection addressed, factual/source fidelity, defended priority
warrant, and remaining uncertainty/reversal conditions separately. For the two
cases, inspect specifically whether the agent preserves the uncooled individual,
consented economic loss and unknown severity; and whether it preserves each
person's conditional survival probabilities and the actual assurance. These are
review questions, not model-visible expected answers.

A populated correction field, a changed recommendation, a structural pass or an
`ANSWERED` label is not a quality score. Report any new unsupported premise and
any disagreement the correction fails to settle. Keep raw inputs/outputs frozen
and write content assessment separately. No operative ledger or source changes.
