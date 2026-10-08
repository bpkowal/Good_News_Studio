# Bounded contract-repair result

Implemented and checked offline. No live provider call, commit, push or diagnostic
deletion occurred. Earlier diagnostic outputs are unchanged.

1. **Exact boundary.** The model authored `ss: SELECTED` alongside a provisional
   explanation. `run_objection_response_pilot.py`, `RevisionBackend.complete_json`,
   previously saved the explanation and forwarded `native_submission` as a native
   `choices` transport envelope without comparing them. No compiler actually
   generated `SELECTED` from prose. The new gate runs before that transport return;
   native `CompactLocalSpecialist.evaluate` remains unchanged.
2. **Invariant.** Authored submission strength cannot exceed declared structured
   repair strength. `DEFEND`, `CONDITIONALIZE`, `WEAKEN`, `RETRACT` are explicit;
   scope is pinned by the host. Contradictions are blocked rather than silently
   edited. Restricted count findings survive independently of dispatch. Native
   admission remains authoritative for consistent submissions.
3. **Files.** New `inference_repair.py`, `prepare_inference_repair_probe.py`,
   `test_inference_repair.py`, `INFERENCE_REPAIR_CONTRACT.md`; modified existing
   `run_objection_response_pilot.py`, `run_specific_inference_probe.py`,
   `SPECIFIC_INFERENCE_PROBE.md`. Diagnostics here are inspectable artifacts only.
4. **Tests added.** Fifteen new deterministic tests cover strength, count/unit
   preservation, numeric and qualitative reversals, supported DEFEND, WEAKEN,
   RETRACT, locality, scope, malformed records, self-report and native integration.
   The native test verifies blocked dispatch and provisional status preservation.
5. **Results.** [24 focused tests pass](focused_tests.log);
   [153 broader regressions pass](broad_tests.log). Native integration files match
   their previous hashes, and `git diff --check` passes.
6. **Scope of behavioral change.** The one-inference runner now requires the opt-in
   repair record and refuses to reuse existing diagnostics. Ordinary objection
   experiments, reviews, frozen-world schemas and native validators are unchanged.
   Bounded native-call failure handling is reused to stop a gate failure; targeted
   summary reports `CONTRACT_BLOCKED`, not an unexplained provider failure. No
   additional review/repair call or automatic objection retirement is introduced.
7. **Remaining limits.** This is structural consistency, not a proof that a model's
   `SUPPORTED` declaration is true. A falsely defended structured record can still
   be consistent with a selected submission even if its prose is qualified. General
   prose contradictions and novel conversion paraphrases need semantic assessment.
   Non-opt-in paths are outside this gate. Native ledger commitment does not certify
   argument repair. See the [contract and limits](../../INFERENCE_REPAIR_CONTRACT.md).
8. **Saved promise packet.** [Audit](audit.json): exact historical output is blocked
   for a missing repair record. With an explicitly labeled reviewer annotation it
   also fails `SELECTION_STRENGTH`, `UNLICENSED_WELFARE_UNIT`, `NONOPERATIVE_REVERSAL`
   and `UNLICENSED_NUMERIC_REVERSAL`. The `0.95`/`0.05 EXPECTED_COUNT PERSON` findings
   remain intact. A reviewer-created cautious example passes the local gate; it is
   not a model correction or admission result. The [prepared packet](prepared_replay_packet.json)
   has `execute: false` and `provider_calls: 0`.
