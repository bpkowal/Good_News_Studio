# Parsing game O: observational argument context

O is a standalone copy of N with a third evidence channel. N is unchanged.
Structure and semantic policies still make the same decisions; raw context does
not enter CEM, alter assertion/occurrence status, or enable graph commitment.

```sh
python parsing_game_O.py
python parsing_game_O.py --sentence "Rain caused flooding by noon."
python -m unittest test_parsing_game_O -v
```

## Evidence contract

Claim evidence now includes `argument_context`, including records without a
recovered argument pair. `parse_world_state()` uses output schema 4 and adds
`argument_contexts` to each enriched proposition. The numeric feature vector,
trained-policy schema 2, and N's semantic policies are unchanged.

An `ArgumentContext` contains:

- `window`: exact local sentence text and token annotations;
- `arguments`: original supplied span descriptors, in input order;
- `regions`: `left`, `argument_1`, `between`, `argument_2`, `right`;
- `status`, `observational_only`, and its own schema version (1).

Each region contains exact `text`, document-global `start`/`end` character offsets,
and tokens with text, lemma, POS, dependency, token/head indices, character
offsets, and original trailing whitespace. Offsets are end-exclusive Python
string indices. Region text is authoritative for reconstruction: concatenating
the five region texts exactly reconstructs `window.text`. Do not concatenate
token whitespace across region boundaries to reconstruct it.

For `Rain caused flooding by noon.` the surface regions are:

```text
left:       ""
argument_1: "Rain"
between:    " caused "
argument_2: "flooding"
right:      " by noon."
```

Spaces and punctuation are intentionally retained. Arguments are ordered by
surface position, independently of cause/effect or subject/object direction.
`input_slot` and `reference` preserve their original identity after sorting.
Repeated phrases are distinguished by offsets, never by a first substring match.
Multi-sentence input uses the predicate's sentence as its local window; the full
original text remains in the enclosing evidence/output.

## Event arguments and limits

For a discovered proposition complement, O records a context between the parent's
subject span and the child's contiguous dependency subtree, with the child frame
ID. For `The flood caused the library to close.`, the second region is
`the library to close`. A parent with multiple discovered complements gets a
separate context for each. No new attachment or control inference is performed.

The subtree is observed syntax, not a verified semantic event extent. Inherited
subjects may lie outside it, and existing attachment mistakes can affect it.
`span_basis=contiguous_dependency_subtree` makes this distinction explicit.
Ordinary argument descriptors retain their entity/event kind; future state-span
producers can use the same offset interface without changing the partitioner.

Missing arguments, overlapping spans, invalid/out-of-window spans, and
discontinuous proposition spans receive explicit statuses and no invented five-way
partition. Their source window and available descriptors remain inspectable.
The context channel does not resolve N's four documented structural gaps.

## Storage and validation

O uses `diagnostics/parsing_game_O_user_probes.json`, retaining the latest five
independent user inputs. Probe JSON stores claim and proposition contexts. Dated
evaluation diagnostics also carry claim evidence contexts; user probe text stays
in the rolling suite rather than being copied into dated evaluation logs.
The terminal summary remains compact; inspect JSON for full token evidence.

Tests cover lossless reconstruction, active/passive/contrastive wording,
multiword and repeated mentions, Unicode/spacing/punctuation, reversed input
order, local windows with global offsets, missing/overlapping/invalid spans,
event complements, nested ATTEMPT, and rolling serialization. A parity test
compares O's full world-state output with N over the existing evaluation sentences
and additional probes, removing only context fields and the output schema version.

There are no embeddings, learned context features, frequency claims, external
model calls, or new semantic classes in this increment. Feature discovery from
these records remains a future empirical task.
