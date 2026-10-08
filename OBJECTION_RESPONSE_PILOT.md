# Bounded objection-response experiment

Freeze the October 6 standard openings for utilitarianism, deontology and Care.
Use the promise case and probability-swap case. Each owner receives its own saved
argument and one explicit competing-claim challenge. No peer responses are supplied.

Compare an ordinary revision instruction with a focused instruction to defend the
reasoning bridge, acknowledge the force of the objection, retain uncertainty and
state a reversal condition. Both arms receive the same packet and output envelope,
the same o3 model and 4096-token ceiling, and exactly one logical revision allowance
per cell. Twelve calls total; provider retry behavior remains unchanged.

The envelope contains an attributed `argument_response` and the unchanged native
submission schema. The wrapper saves the whole raw response and passes only the
native submission to fresh native assessment. It never changes native fields,
world facts, validation or existing ledgers. This is a revision experiment, not a
continuity-checked update to a prior operative ledger.

Model labels `answered`, `partly_answered` and `unanswered` are self-reports. They
cannot retire conflicts. Missing and rejected submissions remain inspectable.

Content review precedes interpretation of admission results. Score each dimension
0 (missing/wrong), 1 (named but incomplete), or 2 (specific and defended):

- Addresses the actual objection and its strongest relevant point.
- Explains the factual and normative bridge instead of repeating a priority slogan.
- Acknowledges residual uncertainty and does not invent factual support.
- Gives a concrete reversal condition related to that bridge.

Check separately whether authored reasoning and native selection status agree.
Preference change, consensus, admission and a model's claim to have answered are
not quality scores. One sample per cell does not establish a treatment effect.

```bash
/tmp/parliament-smoke-env/bin/python run_objection_response_pilot.py --execute --output-dir diagnostics/objection_response_pilot_live_20261007
```

Omit `--execute` to prepare frozen inputs without API calls. Existing completed
output directories are not rerun automatically.

If the provider responses are intact but the transport adapter needs correction,
`--replay-native` re-assesses their exact saved native fields with zero API calls.
Original assessment traces remain available alongside `native_replay`.
