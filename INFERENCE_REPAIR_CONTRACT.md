# Bounded inference-repair contract 0.1

This is an opt-in experiment envelope at the objection-response/native transport
boundary. It does not replace a world, ledger or argument schema, adjudicate an
ethical framework, or establish semantic support.

The saved probe already contains `ss: SELECTED` in the model-authored native object.
The adapter did not create that status; it forwarded an object inconsistent with
the separately saved provisional explanation. Previously, `RevisionBackend` saved
`argument_response` and returned the native `choices` envelope without comparison.
The native specialist then parsed that envelope in `CompactLocalSpecialist.evaluate`
and applied its own calibration/admission. The new check occurs before that return.

## Record and conservation rule

The targeted one-inference runner now requires `inference_repair` alongside the
unchanged native submission and attributed argument. Four operations are legal:
`DEFEND`, `CONDITIONALIZE`, `WEAKEN`, `RETRACT`. The record contains the assigned
inference ID, conclusion scope/status/action, explicit assumption provenance,
conditional preference and assumption IDs, preserved restricted findings,
unresolved claims and reversal provenance. Extra fields are rejected.

The host pins the selection scope. A total-welfare repair cannot relabel its
selection as a survivor-only decision to evade uncertainty. Supported restricted
findings coexist with an unresolved broader conclusion and an optional conditional
preference; they do not license the broader conclusion automatically.

`submission strength <= declared repair strength`:

- `DEFEND` requires `SUPPORTED`, with no declared conditional assumptions, and may
  dispatch a matching `SELECTED` recommendation. Native rules still apply.
- `CONDITIONALIZE`/`WEAKEN` requires `CONDITIONAL` or `UNRESOLVED`. Native status must
  remain `PROVISIONAL` or `UNSELECTED`, with `cc=false`, `esa=false`, `ia=NONE` and
  neither action declared `REQUIRED`/`REJECTED`. A provisional preference must match
  the record and reference declared assumptions.
- `RETRACT` requires `WITHDRAWN`. The repair and count findings remain inspectable,
  but no native submission through the retracted inference is dispatched. This
  avoids pretending the native required `r: A0|A1` field represents no ranking.

The gate **rejects**, rather than edits, incompatible submissions. It never fabricates
a downgrade or silently rewrites model fields. The entire raw response is retained.
A blocked proposal is represented as `CONTRACT_BLOCKED` by the targeted runner;
existing native bounded-call handling may leave no candidate or a `MODEL_ERROR`
record, explicitly marked `local_inference_repair_gate`. This is a local consistency
stop, not a provider failure. No repair call follows it.

## Units, reversal and locality

The existing recognition helper verifies the admitted single-person probability
endpoint. Prior `EXPECTED_COUNT PERSON` rows and their native evidence/calculations
are pinned verbatim. `0.95` and `0.05` do not become welfare or duration measurements.

Within this bounded path, explicit numeric life-equivalent/life-year/utility claims
in operative native fields or referenced repair assumptions require a host license.
None exists for the promise case. Unsupported or illustrative reversals stay in the
repair sidecar; they cannot be operative `ft`. Qualitative framework-derived reversal
text is permitted. Numeric operative reversals require an exact, host-authorized
scope/text/evidence derivation. A model's `SOURCE_GROUNDED` label alone does not
authorize one. This is not a general numeric-threshold subsystem.

The host pins the source/frozen-world and framework hashes. Fields outside a narrow
native edit list (including `ev`, proposition evidence, effect classifications,
grounding and proposed world updates) must remain unchanged. IDs, JSON shape and
references are checked. `answered` and other self-reports are telemetry only: they
cannot satisfy the gate, commit a ledger, retire an objection or upgrade selection.

## Verification and prepared saved packet

Tests cover conditional selection mismatch, preserved count rows, explicit and
assumption-hidden conversions, unsupported/falsely grounded reversal thresholds,
positive DEFEND and WEAKEN paths, retraction, scope switching, source/framework
immutability, protected-field changes, malformed records, self-report isolation,
and native dispatch/provisional preservation. All tests are offline.

Prepare the saved October 7 probe without any provider-call path:

```bash
.venv/bin/python prepare_inference_repair_probe.py --output-dir diagnostics/inference_repair_contract_20261007_final
```

Use a fresh directory. The prepared packet contains the exact saved reply and
contract, plus an explicitly labeled reviewer test annotation and a cautious example.
Those are not model-authored repairs. The historical reply is blocked for a missing
repair record. With the reviewer annotation it is also blocked for selection
strength, unsupported unit conversion and reversal. The cautious example passes
local consistency, not semantic truth verification or native admission.

No live call, objection retirement, new peer cycle or native validator modification
is part of this implementation. Ordinary objection experiments and existing native
paths remain unchanged; the one-inference runner opts into the new gate.

## Limits

This checks structured declarations against submission fields, not arbitrary prose
or ethical correctness. A model can still falsely declare an inference defended
without naming its assumptions; semantic/native review must assess that support.
Novel paraphrases of numeric conversions may evade the bounded lexical check.
Paths that do not opt into this envelope remain outside its coverage, including the
historical twelve-cell experiment and raw saved-native replay. Native admission is
still authoritative after dispatch; ledger commitment never means an objection was
repaired. The native integration test confirms this conditional case stays
provisional, but it is not a proof covering every future native path.
