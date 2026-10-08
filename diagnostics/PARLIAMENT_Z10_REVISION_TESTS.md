# Z10 / Parliament revision tests — 2026-10-01

Tested Parliament commit `614af0c1601dd7169842f130f1c26a5dd0b30564`
with the local Z10 advisory hook, the fused `but not` clause rule, and the
exclusive-allocation compiler fix. The
remote `relent-framework` branch still points to that commit, so the hook and
both fixes are currently local changes represented by
`integrations/parliament_z10/grounding_hook.patch`.

## Targeted outcomes

- Replaying the saved rescue candidate now returns `COMMITTED` with no errors.
  The admitted parties are exactly Maria, the child, and the dog.
- `save the child` retains clauses C1 and C2. `save the dog` legitimately uses
  shared clause C1 because `dog` is in its `but not` remnant and the rival action
  is named on the left side.
- Guardrails reject a shared-only left-side action, an action absent from the
  sentence, a token found only after the remnant's comma boundary, and a remnant
  whose left side does not name the rival action.
- The medicine replay no longer crashes. `compile_exclusive_allocation_complements`
  now builds `party_by_id` from the admitted model before resolving recipient
  effects, matching the lookup maps used by nearby compiler functions.
- The saved medicine candidate now returns `COMMITTED`. A constrained allocation
  draft pass turns each uniquely parented nonreceipt row into an
  `EXCLUSIVE_ALLOCATION_COMPLEMENT`, carries the source-grounded `one` quantity,
  and materializes the selected-transfer-to-nonreceipt edge. It abstains when a
  unique same-action direct transfer cannot be identified.
- A direct `RESOURCE_TRANSFER` to the affected person may parent a stipulated
  welfare outcome without an invented state on the transferred resource. The
  existing intermediate checks still reject person-to-person or act-to-harm
  shortcuts when the source actually names a separate process bearer.
- The four R-layer controller failures are fixed structurally. `xcomp` remains a
  controller-bearing complement; a `ccomp` is admitted only when its child has
  the infinitival `to` auxiliary. Finite reporting complements remain excluded.

## Test results

| Suite | Result |
| --- | --- |
| Z10 bridge + focused Z10/Z9/coverage | 22 run, green |
| Saved rescue replay and adversarial clause checks | Passed |
| Saved medicine compiler replay | Passed; compiled and committed with six source-supported effects |
| R controller suite | 46 run, green; 2 documented expected failures |
| Dilemma dataset integrity and frozen hashes | Passed, 16 dev + 16 heldout |
| Parser full discovery | 405 run, green; 4 skipped, 13 expected failures |
| Parliament launcher + compiler sample | 71 run; 1 failure and 6 errors, identical known failing test identities |
| Parliament global-workspace suite | 514 run; 11 import errors from missing optional `langchain_huggingface` in the minimal smoke environment |
| Parliament world-state suite | 229 run; 9 failures and 24 errors from broader fixture/validator drift |

The 71-test Parliament failures remain the pre-existing hospital-malware fixture
drift and checkpoint-label mismatch; their identities and counts did not change.

The current 71-test Parliament log is `parliament_medicine_controller_regression.log`.
The larger exploratory logs are `parliament_global_workspace_revision.log` and
`parliament_world_state_revision.log`. No new live model call was needed for this
revision test: it replays the two saved live candidates through current admission
code, which isolates the deterministic rule and compiler behavior.
