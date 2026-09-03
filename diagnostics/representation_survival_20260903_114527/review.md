# T0–T3 Proposition-Survival Review

Source trace: `workspace_workspace_20260903_114225_252500_20260903_114527.json`

This review distinguishes the complete in-memory T2 state from the T2 text actually
delivered through `WorkspaceBroadcast.compact()`. That distinction is decisive in
this run. Symbols: ✓ substantially preserved; △ partially preserved or only
available through a parallel/private channel; ✗ absent; — not asserted at T0.

| Feature | T0 testimony | T1 structured ledger | T2 in-memory state | T2 delivered shared text | T3 structured result |
|---|---:|---:|---:|---:|---:|
| Main claim | ✓ | ✓ | ✓ | △ | ✓ |
| Supporting premises | ✓ | ✓ | △ | ✗ | ✓ |
| Qualifiers | ✓ | ✓ | △ | ✗ | △ |
| Conditional dependencies | ✓ | ✓ | ✓ | △ | ✓ |
| Defeaters | ✓ | ✓ | ✓ | ✗ | ✓ |
| Unresolved classifications | ✓ | ✓ | ✓ | △ | ✓ |
| Counterargument structure | ✓ | ✓ | △ | ✗ | ✓ |
| Decision boundaries | ✓ | ✓ | ✓ | △ | ✓ |
| Reason for current lean | ✓ | ✓ | ✓ | ✗ | △ |

## Finding

The suspected general ledger-loss defect is not supported by this trace. The
deontological, virtue, Care, and Rawlsian framework ledgers and committed framework
states are exactly equal at T1 and T3. The utilitarian records are regenerated, but
their central comparison and threshold survive: expected sanitation deaths versus
eight infant deaths.

The clearest representation cliff is T1/T2-in-memory → T2-delivered-shared-text.
`WorkspaceBroadcast.compact()` serializes the complete `problem_state` and then keeps
only its first 4,000 characters. In this trace that prefix contains conflicts,
constraints, and the beginning of Care's position. It does not reach the detailed
`workspace_contributions` for any framework. Consequently, the shared channel loses
most agent-attributed grounds, qualifiers, defeaters, counterarguments, and reasons
for leaning even though those objects still exist in memory.

This is specifically a loss of *cross-agent visibility*. Each recurrent specialist
also receives its own prior committed framework state and private contribution, so
it can copy its own ledger forward. Each assigned challenge is also supplied through
a separate untruncated agent-specific prompt field. Those private channels explain
why T3 ledgers survive despite the impoverished shared broadcast; they do not restore
the global workspace's visibility into other agents' reasoning.

## Framework notes

- **Utilitarian:** The conditional comparison, missing empirical quantity, factual
  question, and reversal threshold survive. The ledger is not byte-identical because
  effect valuations are regenerated, but the decision boundary is stable.
- **Deontological:** The duty conflict and reason for uncertainty survive in the
  ledger. The cycle-2 challenge response nevertheless marks the issue `RESOLVED`
  while moving from the scenario's “near-certain” outcome to a “certain” premise.
  That is an answer-quality/validation failure, not a representation-loss failure.
- **Virtue:** The public-steward role, prudence/compassion conflict, vice risks, and
  unresolved practical-wisdom ranking survive in the structured ledger. The short
  rationale does not display them.
- **Care:** The transactional ledger preserves entrusted responsibility, dependency,
  trust, responsiveness, and the competing community claim. Its top-level cycle-1
  `unresolved` field is `NONE` even though ledger assessments remain `CONTESTED`, so
  this framework has a local projection mismatch.
- **Rawlsian:** Basic-interest security, worst-off subject, maximin basis, and the
  non-applicability of a basic-liberty conflict survive exactly. The short rationale
  “protects worst-off lives” hides that structure but does not replace it.

## Narrow next change

Do not expand the ethical schemas first. Replace raw prefix slicing of the shared
problem state with a deterministic, per-framework balanced projection. Every active
framework should receive a bounded capsule containing: current claim/lean, grounds,
qualifiers, unresolved classification, defeaters, counterclaim, and decision
boundary. Allocate the same cap per framework, preserve provenance, and keep this
projection informational only so visibility and coercion controls remain unchanged.

After that transport fix, rerun this exact diagnostic. Only then decide whether T3
generation needs a stronger reasoning contract. The next separate issue exposed by
this trace is validation of challenge answers—especially preventing a response from
declaring an epistemic-strength challenge resolved by repeating the same strength
upgrade that triggered it.
