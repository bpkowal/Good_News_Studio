# Project Status

Date: 2026-08-20

This branch is evolving the original ethical-agent pipeline into a recurrent
global-workspace architecture with graph-backed state, typed validation, and
non-voting guardrails.

## What is working

- The legacy ethical agents still run, and the workspace path now preserves
  their original testimony as frozen baseline evidence.
- Action identity is graph-backed, so option order and presentation labels are
  no longer the source of truth.
- Framework state for Rawlsian, utilitarian, deontological, and virtue
  reasoning is tracked in typed ledgers rather than free-form prose.
- Visibility, autonomy, reversal, and contingency checks are handled as
  middleware-style audits instead of extra ethical voters.
- Synthesis is now more conservative: it rejects unsupported third actions and
  short-circuits closed-world dilemmas before spending a model call.
- Trace health now distinguishes genuine framework loss from rejected updates
  and uncertain committed state.

## Current design direction

- Move toward one authoritative semantic state shared by the renderer,
  reversal audit, synthesis planner, and halting logic.
- Keep LLMs as the high-entropy reasoning engine, while Python and graph
  validation enforce consistency, identity, and causal direction.
- Prefer typed graph updates and transactional rejection over text-based
  reinterpretation.

## Remaining rough edges

- Some trace-health signals are still stricter than the visible prose suggests,
  especially for Care and Rawls, so the instrumentation remains under active
  refinement.
- The synthesis and contingency pipeline still needs more graph-native handling
  of fallback questions and branch viability.
- `local_specialists.py` remains the largest compatibility surface and still
  contains several policy-adjacent validations that would benefit from smaller,
  more focused modules.

## Recent verification

- The full test suite currently passes.
- The current synthesis short-circuit for explicitly closed action sets is
  covered by tests.

## Next likely steps

1. Push more synthesis and contingency metadata into typed graph objects.
2. Keep tightening trace-health labels so rejected updates do not look like
   genuine framework collapse.
3. Continue extracting smaller modules from `local_specialists.py` once the
   current behavior is stable.
