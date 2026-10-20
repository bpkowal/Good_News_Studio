# Deontology rule-attention diagnosis and proposed bounded experiment

Review date: 2026-10-19. Offline examination of the saved laboratory run; no API calls, no world or production-code changes.

## Verified observations

1. The frozen world preserves N0/N1, their force, bearer/authority party references, governed proposition references PR1/PR2, and source clauses. It does not encode an action-specific compliance judgment or settle the moral legitimacy of the institution.
2. Native epistemic-ledger seeding projects normative records to claim="obligation" and claim="permission", support_ids=[norm_id]. The governed content becomes a separate NOT_ESTABLISHED discourse proposition. The saved deontology prompt contains no governed_proposition_id, bearer_party_id or authority_party_id keys. Original prose remains available, so this is loss of explicit structured binding, not complete loss of the rule.
3. The deontology instruction asks for a primary and strongest competing norm per action and extensively specifies maxim, instrumentalization, harm classification and duty priority. It does not explicitly request examination of each source rule's applicability, institutional authority, or emergency conflict. The existing auth field concerns coercive authorization, not a general assessment of institutional-rule authority.
4. Both saved model replies are parseable JSON and contain dp records for A0 and A1. Both use the generic non-harm duty as primary and rescue as competing; auth=NOT_APPLICABLE. Neither gives an independent institutional-rule assessment. Therefore absence of an operative ledger should not be described as absence of model-authored duty records.
5. The first reply cites E2/E5 rather than either source norm. Its A1 row says dt=IMPERFECT but pb=PERFECT_DUTY, triggering the consistency repair. The repaired reply retains that combination and cites N0 for the different claim "Perfect duties override imperfect duties". Native processing reclassifies that strengthened premise as framework-local. This protects provenance but does not recover the missing source-rule assessment.
6. run_logic_puzzles.project_conflicts records NO_COMMITTED_LEDGER and continues before collecting framework warnings. The saved initial conflict graph therefore includes only utilitarian, Care and virtue conflicts. Deontology and Rawlsianism are diagnostics, not reviewable claims. candidate_maxim_review also obtains its options only from the committed ledger, so simply scheduling a no-ledger agent would not expose its authored maxims there.
7. The first deontology prompt is 58,541 characters (7,091 whitespace-delimited words) with a 3,072-token requested response ceiling. Instruction/format burden is a plausible contributor, not a demonstrated cause. Both saved replies are valid JSON; this diagnosis does not claim provider truncation.

## Proposed changes, in order

### 1. Restore structured rule context without changing the world schema

Render a compact source-rule view from existing normative_propositions, propositions, parties and clauses: source norm ID, force, exact governed content, bearer, authority, source reference. Preserve missing references and unresolved identity. Label the existence/content of a stated rule separately from whether it is morally binding. Do not automatically merge "The rule" with another authority mention or assert the experiment's current state from a conditional rule alone.

Ask the deontology agent to assess source-rule applicability to both active alternatives, moral authority and emergency conflict before the final duty ranking. A rule may be discussed and rejected as controlling; rule coverage is not obedience. Do not equate the existing coercion-auth field with institutional legitimacy. Keep this assessment compact and attributed; avoid enlarging every framework's mandatory response schema.

### 2. Make failed reasoning submissions reviewable

Project a clearly nonoperative proposed-argument node for a framework without a committed ledger. Attach its native warnings and retained authored duty/maxim records, with provenance to the exact response. Reuse bounded native review with an explicit no-ledger recovery path; it must not assume a prior committed record exists. Merely putting diagnostics in the queue is insufficient because review_jobs/reconcile currently operate around ledger claims.

Preserve normal ledger admission and world immutability. A failed argument remains visible and challengeable; it does not become an operative vote or scenario fact.

### 3. Challenge the actual inference

For this saved reply, ask why possible staff exposure constitutes violation of a perfect duty, why emergency rescue does or does not qualify rule compliance, and why foreseen affected staff are described as employed as means. Ask the agent to reconcile its classifications rather than change them to satisfy a validator. Distinguish stop-and-wait as an active alternative from simply doing nothing. Allow a defensible conditional answer or an unresolved duty conflict.

Add an informational source-rule coverage record: addressed, explicitly judged irrelevant, or unassessed, with the agent's explanation. This records attention; it cannot certify a good normative argument. It must not be another world-admission gate.

## Small evaluation before wider adoption

First perform offline checks for exact source binding, no inferred authority/identity, unchanged world digest, and visibility/review routing of the saved rejected submission. Retain existing committed-ledger behavior.

Then run a matched deontology-only comparison on this exact frozen world: current prompt versus compact rule-bound context and source-rule assessment instruction, same model and output/call budgets. Assess applicability, authority, emergency conflict, risk/means distinctions, completeness of duty records and native admission separately. A changed preferred action is not a success criterion. Keep uncommitted proposals and both arguments visible.

If the change helps this case, use two transfer cases before generalizing: an explicit emergency exception and a scenario without an institutional rule. Neither should be forced into a perfect-duty hierarchy or fabricated authority analysis. A single matched pair would establish feasibility only, not reliable improvement.

## Discussion point

Recommended first implementation: restore rule bindings plus compact rule-attention instruction, and fix review visibility for rejected submissions. Do not begin with a larger duty ontology, a longer generic prompt, or a requirement that the institutional rule must win.
