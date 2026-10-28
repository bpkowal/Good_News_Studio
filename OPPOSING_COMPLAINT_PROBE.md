# Addressing the actual opposing complaint

Status: bounded saved-reply experiment completed with twelve live calls. See the
[content review](diagnostics/opposing_complaint_live_20261025/CONTENT_REVIEW.md):
opponent recognition improved; justification did not reliably improve. This adds no operative specialist, world
fact, native acceptance rule or automatic resolution of an objection.

## Question

Can a specialist improve its reasoning by replying to the actual complaint of a
person disadvantaged by its recommendation, instead of answering a weak or
misattributed objection and repeating its preferred principle?

This builds on the matched saved-reply design in
`run_objection_response_pilot.py` and the precise-inference follow-up in
`run_specific_inference_probe.py`. Unlike the earlier case-specific assigned
challenges, the new selector is scenario-independent and receives no assistant
content review, desired victim, correct answer, or other framework's response.

## Procedure frozen before calls

Use the four saved Contractualist arguments from
`diagnostics/contractualist_comparison_v03_live_20261025`. Preserve the exact source,
options, framework instruction and original answer. Validate the original output
hash receipts before preparing a new directory.

1. One independent o3 pass per case proposes a strong grounded complaint against
   the original recommendation. It identifies the disadvantaged person, feasible
   alternative, source quotations, exact quoted prior inference and why the old
   answer fails to settle that complaint. It may disagree with the original
   `strongest_objection` field. The assignment remains a fallible model proposal.
2. Freeze all four assignments before any response arm. Both arms receive exactly
   the same assigned complaint and original argument in fresh, stateless requests.
3. The ordinary arm asks for a revision in light of that complaint. The focused
   arm asks the agent to concede its force, identify the priority premise, explain
   why the complainant should accept that priority, compare the best counterclaim
   to the alternative, and revise the conclusion if it cannot defend the inference.
4. Preserve both replies and exact inferred changes as new attributed records.
   No reply is dispatched to native acceptance and no old statement is rewritten.

Maximum: four assignment calls and eight reply calls, twelve in total. Keep the
opening model (o3) and low reasoning setting. Both response arms have the same
compact schema and 8192-token ceiling; selectors have 4096. No SDK retries, hidden
repairs, further argument rounds, peer input or collective verdict. A missing or
malformed assignment is recorded, not replaced by an authored objection. Invalid
quote bindings remain visible and do not become source evidence. No grounded
target means both response arms are explicitly unrun for that case.

The common response schema itself asks for relevant reasoning; the ordinary arm
is therefore not a pristine unconstrained baseline. Comparison with the original
opening includes an extra selector and revision opportunity and is not matched
on total cost. Arm-to-arm matching holds input, schema, budget and assignment
constant; their instruction differs. One sample per arm/case is exploratory.

## Content assessment

First assess the assignment: is it the complaint of someone disadvantaged by the
actual recommendation, with a feasible alternative and source-supported grounds?
It need not be uniquely the strongest. An invalid assignment makes the reply
comparison harder to interpret, even if both arms share it.

Then use the existing 0–2 objection-response rubric, without a total score:

- Actual complaint addressed: does the answer respond to that person's burden
  and proposed alternative rather than swap the complainant or action?
- Bridge defended: does it explain the disputed priority instead of repeating
  the selected norm, comparing unrelated alternatives or importing aggregation?
- Uncertainty/source fidelity: are missing facts left unknown and normative
  assumptions explicit? Does the final conclusion respect the unresolved premise?
- Reversal condition: does it identify a relevant factual or normative change,
  rather than an invented numerical cutoff or generic promise to reconsider?

Record an exact passage for each material improvement or regression. Changed
recommendations, longer prose, admitted faults, hedging and self-reported ANSWERED
are not quality measures by themselves. A defended conditional conclusion can be
useful without retiring the objection. An assistant-only review is labeled as
non-blind and fallible; it does not certify either normative winner.

## Run

The next increment supplies the existing consequence matrix and an explicit
fact-first review. See [the refinement design](CONSEQUENCE_COMPLAINT_REFINEMENT.md)
and its [four-call content comparison](diagnostics/consequence_complaint_live_20261026/CONTENT_REVIEW.md).

Prepare offline (zero API calls), or add `--execute` for the bounded experiment:

```bash
.runtime/venv/bin/python run_opposing_complaint_probe.py \
  --output diagnostics/opposing_complaint_live_20261025 --execute
```

Use a fresh output directory. `--case` can restrict the prior cases; `--framework`
can select another specialist present in the same saved run. No names, scenarios,
complaint weights or morally correct choices are hard-coded in the mechanism.

The output preserves original packets, requests, implementation bytes, provider
responses, assignments, both readable replies and hash receipts. The original
16-call study and its partially connected argument graphs remain unchanged.
