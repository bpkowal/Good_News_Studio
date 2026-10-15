# Negation and hypothetical reversal validation

Implemented as an additive native patch, pinned in pipeline.lock.json. The world schema and frozen tram world are unchanged.

## Corrected distinctions

The action-case check now preserves explicit negation for paired choices, including “not pulling”, “without pulling”, and “refraining/abstaining from pulling”. It no longer treats “does” as the meaningful anchor distinguishing omission from intervention. Normal action mapping, harm acknowledgment, false-resolution checks, and semantic verification remain in place.

Numbers in clearly hypothetical reversal prose are treated as counterfactual parameters, not assertions about the current scene. The saved “0 vs 1” example therefore no longer destroys the separately grounded boundary at 10. A typed boundary whose numeric parameter is unverified remains a non-operative proposal: its hypothetical prose survives, but no graph update commits. An unsupported current-world numeric assertion still gets rejected. Existing structural, precision, action-local effect and recomputed-value checks remain active.

This is a bounded language check, not a general entailment verifier. Ambiguous wording outside the supported operator patterns still needs semantic review. It does not establish that casualty minimization equals total welfare or that an agent's moral priority is justified.

## Exact saved-response reassessment

All five original provider responses were replayed through the native engine without changing their bytes or making new API calls. The shared world stayed identical and all authored preferences stayed the same.

- Utilitarian: both reported warnings disappear. The hypothetical prose and grounded 10-death boundary survive. Its report position changes from ATTENUATED to FULL under existing admission rules; that is a procedural change, not a new moral endorsement.
- Deontology and Care: the false omission-mapping warning disappears. Their substantive uncertainty remains, including the doing-harm prohibition bridge and Care's unestablished entrustment claim.
- Virtue and Rawlsian: no targeted validator difference. Rawlsian numerical-grounding objections remain.

The original live run and report are preserved. The [new report](report_final/DEMO_REPORT.md) is explicitly a saved-response reassessment, not another live reasoning experiment. The viewer's original saved run has not been silently replaced.

## Tests and controls

Four new tests pass, covering paraphrases and action-order permutations; actual opposite-action descriptions; swapped death counts and cross-action citations; the exact saved hypothetical example; unverified operative parameters; and invented current-world numeric assertions. Four existing native regression tests also pass, including harm mapping, false resolution, malformed multi-clause updates and committed-polarity inversion. The broader native module requires the existing full RelEnt profile; the compact profile's import attempt lacked NumPy, so no dependencies were added to it.

The eight locked patches reconstruct the current native source tree, and source readiness passes. Original provider replies, all five recorded preferences and the shared world remain unchanged.

- [Before/after validation fields](reassessed_final/validation_comparison.json)
- [Exact-response and manifest verification](verification.json)
- [Four new tests](tests_final.log)
- [Four existing regression tests](native_regression_final.log)
- [Runtime reconstruction](runtime_verification.json)
- [Reassessment controls](reassessed_final/replay_controls.json)
