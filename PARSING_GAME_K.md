# Parsing game K

K is a standalone revision of J. J and its tests/data remain unchanged.
Run K with the same environment:

```sh
python parsing_game_K.py
python -m unittest test_parsing_game_K -v
```

The prompt, latest-five rotation, and causal experiment remain available. K uses
its own `diagnostics/parsing_game_K_user_probes.json` and K-prefixed diagnostics.
It does not automatically copy or reinterpret J's saved probes. Its user-probe
file contains both the causal output and the latest event/state output; evicting
a probe removes both. User text is excluded from dated experiment logs.

For a one-off inspection without storing the sentence:

```sh
python parsing_game_K.py --sentence "A courier may send two parcels to a depot."
```

## Parallel representations

```python
from parsing_game_K import train_policy, parse_world_state

policy, history = train_policy()
result = parse_world_state(
    "A courier can carry the package to a customer who ordered it yesterday, "
    "or reroute it to a depot where five workers face identical exposure.",
    policy,
)
```

The result contains:

- `causal_claims`: J's CEM interpretation, unchanged. A transfer is not inferred
  to be a causal relation just because it has an actor and an object.
- `events`: draft predicate frames with semantic roles, assertion annotations,
  modifiers, context, provenance, and unresolved issues. The name includes both
  event and state records, distinguished by `kind`.
- `entities`: bounded mentions with original offsets, head information, explicit
  quantities, and unresolved-reference status.
- `alternatives`: proposed `or` branches with attachment evidence. The word "or"
  does not establish exclusive-or, a chosen branch, or a completed action.

In this example, the frames retain the proposed carrying and rerouting actions,
their destinations, the earlier request, and the workers' exposure description.
The attachment of rerouting remains flagged as ambiguous. Relative descriptions
are associated with the relevant branch through their antecedent mentions.
Each occurrence of "it" remains an unresolved mention: K does not silently
replace it with "the package".

`destination` intentionally preserves both human recipients and physical
destinations of a transfer without requiring an ontology that distinguishes them.
Prepositions and their arguments are retained alongside these role proposals.

## Semantic and epistemic boundaries

A small, explicit frame lexicon recognizes possession, remaining inventory,
transfers, redirection, requests, encounters, and basic copular properties.
Unknown predicates retain syntactic subject/object roles with meaning
`unresolved`. No sentence-specific features were added to the CEM model.
Passive transfers retain the affected item as the theme and an explicit by-agent
as the actor. Unary states do not require a fabricated second argument.

Quantities retain their surface text. Basic explicit integers, number words zero
through ten, and "single" receive numeric values; unsupported expressions retain
`value=None`. Indefinite articles are not silently treated as exact inventory.

**All event/state frames are drafts with `eligible_for_world_state=False`.**
"Asserted" means asserted locally in the text, not externally verified or
necessarily projected out of its surrounding clause. Embedded records carry
their enclosing assertion status; relative descriptions retain branch context.
Possible/denied actions are not converted into completed actions. Causal-claim
eligibility retains J's separate, heuristic interpretation.

K does not perform graph writes, arbitrary discourse inference, general coreference,
normative ranking, counterfactual simulation, or inferred consequences such as
"sending an antidote saves a patient." A rolling suite is a set of independent
probes, not sequential world-state memory. Misspelled or incomplete input is kept
as supplied rather than silently corrected.

## Validation

`test_parsing_game_K.py` checks the existing causal suites plus possession and
remaining states, alternative branches, destinations, quantity extraction,
unresolved pronouns, passive roles, scoped assertions, copular states, unknown
predicates, and separate rotating storage. Causal policy schema remains version 2
because K did not change CEM features or training. Event/state fixtures do not
train CEM and their extraction is not measured by the printed causal accuracy.
