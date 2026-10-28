# Contractualist contribution: clarify the decision without forcing a winner

Status: implemented. One live check produced two critical questions and no forced
recipient recommendation. The argument graph remains partial; see the
[result and limitations](diagnostics/contractualist_questions_20261026/RESULT.md).

The Contractualist pilot can contribute a structured account of the critical
unanswered questions, even when it cannot defend a ranking. It may also recommend
an action when the comparison is supported. Neither a recommendation nor
abstention is a mandatory outcome.

## Current behavior

Fresh generation and the consequence-first follow-up ask the specialist to:

1. Identify the actual competing principles and individuals' complaints.
2. Separate missing facts, contested interpretations and normative priority.
3. State the specific unanswered question and why it matters to the decision.
4. Explain what evidence or argument could help and what follows conditionally
   from different answers.
5. Leave a ranking unresolved, or state a conditional preference with its
   disputed premise visible. Do not substitute an arbitrary numerical exchange
   rate, an imagined person's consent, a headcount or an unjustified tie-breaker.

It must not claim that contractualism is incapable of answering a question merely
because the current argument cannot answer it. Rejecting an absolute rule does
not answer a qualified alternative. Hypothetical reasonable acceptance is an
argument to examine, not actual consent or a prediction of average approval.

`critical_questions` records have:

- `id`, `kind` (`FACTUAL`, `INTERPRETIVE`, `NORMATIVE`) and `question`.
- `role`: `DECISION_DEPENDENCY` or `RESIDUAL_CONCERN`.
- Affected `person_ids`, `action_ids`, `principle_ids` and `target_handles`.
- `why_it_matters`, `needed_to_resolve` and `conditional_implications`, each of
  which separates an assumption from what would follow if it were defended.

All are model proposals. Dependencies describe a justification's needs; they are
not new automatic admission blockers. A question does not settle its own answer
or prevent a recommendation supported by independent reasons.

## Reports and compatibility

Questions appear before the argument graph in `candidate_arguments.md` and get a
separate `critical_questions.md`. The run summary displays decision status and
question count separately from technical construction status. An `UNRESOLVED`
decision can coexist with a useful, successfully recorded contribution.

The shared neutral schema makes this field available to other frameworks without
requiring a contractualist method. The Contractualist-specific instructions carry
the stronger question-focused task. The consequence-first follow-up uses current
task guidance while preserving the exact saved source and old framework
instruction in its packet.

Older responses remain readable and retain their raw fields. If they never
recorded structured questions, the report says `NOT_RECORDED_LEGACY`; it does not
pretend the questions were absent or resolved. Reference and uniqueness/dependency
findings are annotations, not moral validation. No world atom, operative roster,
native acceptance rule or source fact is changed.

Implementation: `contractualist_questions.py`, `argument_frameworks.py`,
`contractualist_pilot.py`, `complaint_consequences.py` and their runners. Versioned
snapshots now capture the question helper as well. Existing frozen outputs keep
their original implementation bytes and identities; newly prepared requests use
the new versions.

## Verification

126 related tests pass, including question-only output, conditional preferences,
recommendations with residual questions, preserving old responses, unknown
references without deletion, and keeping conditional answers out of world facts.
All sixteen original four-framework pilot responses remain readable unchanged.
The authored offline example in
`diagnostics/contractualist_questions_20261026/authored_question_only_example.md`
is explicitly a structural test, not a model result.

The predeclared check used one fresh live Contractualist generation for the unchanged blanket
fixture, without other frameworks or an intended winner. It received the current
task and common schema, not this document, this conversation, previous replies,
review notes or an authored answer. One o3 call, low reasoning, 8192-token ceiling,
zero retries. Assessment asks whether it identifies the actual open priority
question, distinguishes it from factual unknowns and limits its conclusion
accordingly. Producing a question or a null recommendation alone is not success.
No matched improvement claim can follow from one call.

```bash
.runtime/venv/bin/python run_contractualist_pilot.py \
  --case blanket_prior_assurance --frameworks contractualist \
  --output diagnostics/contractualist_questions_20261026/live_blanket --execute
```
