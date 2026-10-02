# Z8: separate ellipsis proposition identities

Use `parsing_game_Z8.export_candidate_graph(text, package_id=None)`.
Z8 starts from Z5, separates stripping reconstructions, then applies Z6 condition
dependencies and Z7 condition-content analysis. Earlier exporters remain unchanged.

“Maria saved the child, but not the dog” now has two proposition identities:

- The spoken saving proposition, with Maria and the child as participants.
- A reconstructed saving proposition, with Maria and the dog as participants,
  negative polarity, and unresolved assessment.

Each reconstructed participant requires its own predication candidate. Consumers
grouping by proposition no longer combine the child and dog roles. Compatible
unchanged roles are copied: a recipient replacement retains the medicine object.
Competing replacement roles stay exclusive; ambiguous unchanged roles are not
silently copied. Participant bundles may therefore still be incomplete hypotheses.

Modal candidates and their interpretation sets/questions are cloned onto the new
proposition. Their dependencies and scope references are remapped locally, avoiding
a dependency on the positive antecedent or a self-modal dependency cycle. Outer
attribution and conditional/hypothetical scope is retained. This identity change
does not resolve negation-versus-modal operator ambiguity.

## Schema 0.4 provenance record

The envelope requires `reconstructions`, empty when none are exported. It retains
schema 0.3's required `condition_contents`. Each reconstruction record contains:

```json
{
  "id": "reconstruction_0",
  "proposition_id": "p_reconstructed_4",
  "antecedent_proposition_id": "p1",
  "antecedent_candidate_id": "c0",
  "predication_candidate_id": "c3",
  "participant_candidate_ids": ["c4", "c5"],
  "evidence_ids": ["e0", "e6", "e4"],
  "method": "stripping_not_nominal"
}
```

IDs are illustrative. The antecedent reference is provenance, not a selection
dependency, coreference assertion, or occurrence claim. The new proposition uses
the antecedent's predicate evidence plus exact negation/remnant evidence; no missing
words are inserted into document text or given fabricated source offsets.

The validator checks distinct identities, unique reconstruction ownership, typed
antecedent/target predication references, acyclic antecedent provenance, complete
participant-ID inventory for the reconstructed proposition, and each participant's
local predication dependency. It accepts legacy schemas 0.1–0.3 unchanged and rejects
the new field in those schemas. Package and selection versions must match.
Partial selections remain permitted; the participant inventory is not a requirement
to select mutually exclusive role alternatives together.

Condition-content analysis runs after identity separation. A reconstructed predicate
inside an antecedent remains an explicit content gap until supported condition
structure accounts for it. It cannot silently disappear merely because it received
a new identity.

## Validation

164 tests passed: 155 baseline tests and 9 new tests covering role isolation,
simultaneous spoken/reconstructed selections, local modal dependencies, retained
objects, ambiguous roles, attribution/condition scope, multiple reconstructions,
deterministic IDs, invalid/circular provenance, and unchanged non-stripping output.

```sh
.venv/bin/python -m unittest test_parsing_game_Z8 -v
```

This increment covers Z5's exported nominal stripping candidates. It does not add
new graph export for verb-phrase ellipsis, gapping, or sluicing, or verify the
semantic correctness of a copy. Original-text index alignment is now handled by
the shared proposer/Z5 path, as described below.
Old saved exports retain their old identities; re-export through Z8 for separation.

## Original-text alignment follow-up

`EllipsisProposer.propose(text, ..., doc=None)` now generates indices from the exact
input text, without corpus surface normalization. Z5 passes the same original spaCy
Doc to the proposer and to stripping export. An explicitly supplied Doc with
different text raises `ValueError`. Z6–Z8 inherit this fix; no schema change is needed.

Whitespace tokens are ignored when locating the `but/and not` cue and the nominal
remnant, but are retained in the source document and global token numbering.
`not only` remains excluded even when whitespace separates the words. The detector
may still parse text for scoring; those parses do not supply exported indices.
The Z4 base graph still independently parses the same original text with the same
spaCy pipeline. This is not a wholesale single-parse refactor of the exporter stack.

[Frozen invariance fixtures](fixtures/ellipsis_text_invariance.json) cover 21 forms:
extra spaces, tabs, newlines, sentence gaps, leading/trailing whitespace, cosmetic
punctuation spacing, straight/curly quotation marks, an exclamation mark, Unicode
names, and prepositional remnants. Tests check original proposal indices, semantic
role/scope projections, exact source offsets, Z5/Z8 reconstruction survival, and
selection validity. This does not assert invariance to punctuation that changes
sentence meaning or arbitrary spaCy attachment changes.

Alignment validation: 169 tests passed, including five new test methods exercising
the 21 frozen variants and the existing 164-test regression suite.
