# Live constrained-draft allocation tests

Date: 2026-10-01. Backend: Parliament `OpenAIWorkspaceLLM`, model `o3`, two
attempts per scenario, Z10 advisory mode `EVIDENCE_ONLY`.

## Antivenom

The scenario used one vial of antivenom, two recipients, an explicit
`either ... or ... but not both` choice, and a stipulated recovery consequence
for each branch.

Result: `COMMITTED` after two model attempts. The first draft had the expected six
effects and allocation topology after compilation, but its nonreceipt effects cited
only the comparison clause. The repair added factual provenance to those effects.
The final graph has two direct transfers, two downstream exclusive-allocation
complements, two recovery outcomes, and four causal links.

The final nonreceipt effects do not record `one vial`. The source quantity survived
in the scenario but was not propagated into the constructed effect. This is a
semantic-coverage failure even though admission succeeded.

## Epinephrine injector

The second scenario changed the resource phrase to `a single injector of
epinephrine`, changed all names, and used `must give` with the same explicit
recipient exclusion and stipulated survival consequences.

Result: `REJECTED` after two model attempts. The first draft omitted both
nonrecipient branches. The rebuild added them and their parent links, but cited
only the comparison clause. Validation then rejected both constructed effects for
lacking factual support. A further local patch was requested but unavailable after
the two-attempt limit.

## General finding

The topology rule is broader than the original medicine wording, but construction
is not yet evidence-closed. A derived exclusive-allocation complement needs one
auditable support bundle:

1. the direct transfer parent;
2. the choice or exclusion span;
3. the singular-resource or explicit indivisibility span;
4. the copied source quantity, when present.

The compiler should assemble that bundle deterministically when it is unique and
abstain when the resource or transfer parent is ambiguous. This is a general
`transfer + quantity + exclusivity` construction rule, not a medicine or injector
schema.

Both scenarios also exposed clause segmentation of the abbreviation `Dr.`: the
stored Parliament clauses begin with `Chen` or `Reed`, dropping the title fragment.
The source document remains intact, but exact clause provenance should use an
abbreviation-aware splitter or an alignment-preserving sentence segmentation step.

Frozen Z10 proposed a stripping reconstruction for `but not both` in both cases.
The candidate-to-atom compiler should treat `both` as a choice quantifier here and
leave the reconstructed event unselected.

## Deterministic closure follow-up

The compiler now assembles the unique support bundle described above. Replaying
the rejected epinephrine candidate commits with all six source effects: two direct
transfers, two nonreceipt complements carrying `single injector`, and both stated
survival outcomes. The corresponding antivenom and medicine replays also commit;
their complements carry `one vial` and `one`, respectively. Ambiguous transfer
parents still cause abstention.

The `Dr.` split is fixed with abbreviation-aware sentence segmentation. The full
source clauses now retain `Dr. Reed` and `Dr. Chen` instead of silently dropping
the title fragment.

The choice-quantifier behavior remains frozen. A separate 18-case analysis found
false stripping reconstructions in all seven tested `either/or ... but not both`
variants, while genuine stripping retained 5/5 recall and VP ellipsis, gapping,
and sluicing retained 4/4 expected routing. See
`diagnostics/Z10_BUT_NOT_BOTH_REGRESSION.md` for the proposed narrow guard and
`diagnostics/z10_but_not_both_regression.json` for the complete observations.
