# Parsing game P: core arguments and attached phrases

P is a separate script; N and O are unchanged. Run:

```sh
python parsing_game_P.py
python -m unittest test_parsing_game_P -v
```

P uses output schema 5, P-prefixed diagnostics, and
`diagnostics/parsing_game_P_user_probes.json` (latest five independent inputs).
The CEM numeric feature names and saved-policy schema remain unchanged.

## Result for the motivating probe

`After the game the players all went home` now yields a draft proposition with:

- subject: `the players`;
- object: absent;
- occurrence: `asserted_in_text`;
- attached `After the game`: temporal;
- attached `home`: destination;
- quantifier `all`: explicit, scope unresolved.

The proposition/event is still ineligible for direct world-state commitment.
Quantification records distinguish an assertion from resolution of the group's
scope; `all`, `every`, and `each` no longer automatically erase assertion status.
Quantified causal claims remain blocked by a separate scope-validation gate.
Negative/existential quantifiers (`no`, `neither`, `any`) retain conservative
handling; this increment does not implement general quantifier logic.

## Extraction policies

Verb/AUX predicates no longer invent objects from nearby words. Arbitrary
prepositional objects are no longer binary relation arguments. Existing licensed
semantic relations can retain an oblique relation argument, and syntactically
marked passives can retain a by-agent for causal direction. These obliques are
separate from the `object` field of structural propositions/events.

Each proposition and corresponding world-state event contains `attachments`
and `quantification`. Attachments preserve dependency/head indices, subtree token
indices, a contiguous span where available, role, policy provenance, and O-style
`argument_context` between the subject and phrase. Missing/overlapping spans keep
their explicit context status. The terminal displays attachment roles and
quantification; JSON retains full evidence.

The initial role policies are deliberately bounded:

- predicate-attached `before`/`after`: temporal;
- `by` with a passive `agent` dependency: agent;
- `to` or adverbial `home` with go/come/return/walk/travel/fly/flee: destination;
- other captured prepositions/adverbs: unresolved.

These are explicit local application policies, not new VerbNet evidence or full
word-sense disambiguation. `The bat flew by the blind man` now has no object; its
by-phrase is preserved with unresolved role, without asserting path/proximity.
The existing event schema's prepositional argument records remain available.

## Regression results and tradeoffs

The P suite reuses N's ATTEMPT tests and M's embedding tests, with 12 contrastive
role sentences and a quantified-causation commitment test. Fixtures remain
excluded from CEM training. Two previous expected failures become ordinary tests:
the by-phrase object error and the conditional `refunds` object leak. Passive
controller recovery and coordinated complement attachment remain expected failures.

The ordinary holdout drops to 7/8: `Insomnia caused by stress yesterday.` now
abstains. spaCy marks its by-phrase as `prep`, its subject as active, and provides
no passive auxiliary. P does not restore a blanket by-means-passive heuristic to
recover this fragment. The gold label is unchanged and the regression is checked
explicitly. Other existing causal suites and the multiple-relation suite retain
their previous scores.

Event-complement links remain available even when no noun pair can be formed;
such causal records remain unresolved and uncommitted. P does not add semantic
classes, cross-sentence reference resolution, or permission to commit draft events.
