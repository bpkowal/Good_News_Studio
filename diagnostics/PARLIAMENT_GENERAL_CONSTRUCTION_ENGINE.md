# Parliament general construction engine

Date: 2026-10-01

## Implemented boundary

Parliament's `ScenarioWorldModel` schema is unchanged. Compilation now executes an
ordered internal catalog of schema-neutral construction rules rather than an opaque
scenario-oriented sequence. Rule metadata is diagnostic only and is not serialized
into the world state.

The catalog covers eight families:

1. event and state identity;
2. participants and semantic roles;
3. modality, negation, and conditional scope;
4. causal and temporal links;
5. resource transfer and quantity;
6. exclusivity and choice;
7. benefits, harms, and stipulated outcomes; and
8. unresolved alternatives and abstention.

No rule identifier contains `medicine`, `antivenom`, `epinephrine`, `trolley`, or
`rescue`. Those scenarios are compositions of the general families.

## Composition tests

- **Allocation:** medicine, antivenom, and epinephrine replays use transfer,
  quantity, exclusivity, choice, and stipulated outcomes. All commit. Constructed
  nonreceipt effects carry deterministic evidence and source quantities.
- **Conditional harm:** a lever action, a condition, a quantified adverse outcome,
  a causal edge, and a temporal edge compose without a trolley-specific rule. The
  condition is propagated to the causal edge and the temporal edge is preserved.
- **Modal rescue:** possible benefit and negated unresolved ability remain possible
  or unresolved. Compilation does not promote either to certain occurrence and
  adds no unsupported effects.
- **Ambiguity:** unsupported and invalid derived atoms continue through the existing
  abstention/quarantine paths instead of being repaired with invented events.

## Quantifier correction

Bare `both` after a same-sentence `X or Y` alternative now closes the alternative
set and does not trigger stripping reconstruction. The complete post-fix matrix is
18/18: seven exclusive-choice cases, five genuine stripping cases, four other
ellipsis cases, and two negative controls. A nominal remnant such as `both parents`
continues to use stripping.

## Verification

- 68 Parliament construction, quantity, status, and clause-segmentation tests pass.
- 66 frozen-parser and ellipsis tests pass.
- 12 parser-to-Parliament bridge tests pass.
- The broader Parliament module runs 514 tests: 503 pass and the same 11 tests fail
  to import optional embedding packages absent from the isolated smoke environment.

## Remaining boundary

Z10 choice sets are still advisory inputs to first-pass drafting. The rule engine
can preserve an unresolved atom and abstain from unsupported closure, but it does
not yet enumerate every compatible Z10 choice-set selection into separate
`ScenarioWorldModel` drafts. That adapter is the next general component; it should
produce typed candidate drafts and feed each through this same rule catalog without
adding a new scenario schema.
