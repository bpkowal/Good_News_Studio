# Candidate maxim increment

Implemented candidate maxims per option, including omission, in the shared native
deontological generation schema and framework ledger. Maxims remain attributed
interpretations; the frozen world is unchanged. Reports expose the proposed
policy and justification without attributing an established intention to Dana.

The localized review packet carries both prior maxims and asks specifically about
means versus side effect, consent, universalization, omission's rescue obligation,
and the distinction between an action and the maxim underlying it.

## Verification

- Four new focused tests pass, including real native generation and recurrent
  review with controlled responses.
- Four negation/boundary regressions pass.
- Thirty-one related review, presentation, and runtime regressions pass.
- An unchanged response claiming `RESOLVED` leaves the native objections open.
- A specific correction retracting the unsupported means inference removes that
  calibration objection while preserving a conflicted judgment.
- A defective maxim does not automatically prohibit an otherwise permissible
  action. Missing fields and attempts to add an established-intention field fail
  the maxim schema.
- All nine locked patches replay successfully from the pinned commit and reproduce
  the modified native modules. Runtime preflight is ready; see
  [runtime verification](runtime_verification.json).

No new provider calls were made. The controlled response augments the saved tram
reply with test maxims; it is not a new model-generated argument. This verifies
the reasoning interface and challenge mechanism, not a live improvement in
philosophical quality. The separate causal-relation handoff issue remains open.

A pre-existing presentation test used an obsolete temporary native checkout; it
now tests the pinned persistent runtime. The initial broader check also used the
compact environment for an application test requiring NumPy; the successful
regression commands use each environment for its intended role.

Design and exact test commands: [Candidate maxims](../../CANDIDATE_MAXIMS.md).
