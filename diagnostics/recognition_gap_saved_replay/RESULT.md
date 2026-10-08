# Recognition fixes: saved-input replay

Implemented and installed the bounded recognition bridge in the current native
Parliament checkout. The world-state schema and saved world are unchanged.

| Saved input | Before | After |
| --- | --- | --- |
| Ben's 95% survival qualifier | utilitarian probability UNKNOWN | 0.95; outcome remains PROBABILISTIC |
| Cara's 5% survival qualifier | utilitarian probability UNKNOWN | 0.05; outcome remains PROBABILISTIC |
| Non-recipient death rule, quoted by Utilitarian and Deontological | HYPOTHETICAL | STIPULATED source-rule record, backed by E5/E6 → E7/E8 |
| Single-dose stock statement | no separate resource ledger atom | separate quantity record with resource-party and effect proof |
| “but not both” | no separate constraint ledger atom | separate exact-source exclusivity record |
| Strengthened stock/benefit compounds | whole claim hypothetical | supported quantity fragments visible; whole strengthened claim still hypothetical |

Five recognition records are projected: quantity, exclusivity and three source
rules. Their original evidence and supporting world IDs are retained. Exact rule
matching does not license “Ben will die” or “all patients will die” from the
generic non-recipient rule. Source quantity alone never proves exclusivity.
Entrusted-care claims remain hypothetical.

The source implementation is [parliament_recognition.py](../../parliament_recognition.py).
The native [integration patch](../../integrations/parliament_z10/recognition_gap.patch)
adds its matching helper and updates graph-to-ledger projection, dependency
bookkeeping, utilitarian accounting and compact valuation expansion. It has
already been applied to `/tmp/parliament-smoke-614af0c`. The runner now records
the installed helper's hash. Patch application to the pre-increment files was
checked and reproduces all five tested native files exactly.

Probability parsing preserves explicit local percentages as decimal masses,
including 1% → 0.01. Bounds, negated/hedged values, conflicting qualifiers,
overall/local combinations and precision too large for the native field remain
unknown. This does not invent magnitude, duration, utilities or event occurrence.

Validation: **121 tests executed: 120 passed, one pre-existing failure, no skips**.
The trolley omission admission test still rejects AV3/AV6 for unestablished
derivation assumptions and POLARITY_INVERTED transformations, exactly as before
this increment. Medicine admission, conditional admission, the rescue expected
rejection, Logic_Puzzles acceptance and hypothesis containment passed. See the
[complete regression log](regression.log).

The [replay](replay.json) uses the saved final-cycle agent claims and existing
hypotheses, not newly generated model output. Two quoted death-rule claims
rebind to the new stipulated rule even when the old hypothesis records are
present. The [recognition graph](recognition_graph.md) shows evidence links.
**Zero API calls; no normative judgment recomputed.**

Remaining limits: resource projection currently uses the admission-backed
allocation-complement evidence path. Broader stock constructions and unrestricted
paraphrases are not implemented. A quantity fragment does not establish words
such as “only” or “life-saving”, nor “only one patient can benefit”. The missing
deontological duty ledger and unresolved welfare magnitudes/durations are separate
issues; this replay does not prove live normative convergence.

Reproduce the saved-input check:

```bash
/tmp/parliament-smoke-env/bin/python replay_parliament_recognition.py
.venv/bin/python -m unittest -v test_parliament_recognition
```
