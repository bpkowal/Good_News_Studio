# Matched argument-quality pilot

Same model, source world, framework commitments, token ceiling and one adapter call per cell.
Counterfactuals change only probabilities or add an explicit promise. No collective verdict is computed.

| Case | Arm | Framework | Submission | Native ledger | Calls |
| --- | --- | --- | --- | --- | --- |
| baseline | standard | utilitarian | VALID | COMMITTED | 1 |
| baseline | standard | deontological | VALID | NONE | 1 |
| baseline | standard | care | VALID | COMMITTED_WITH_UNCERTAINTY | 1 |
| baseline | argument_checklist | utilitarian | VALID | COMMITTED | 1 |
| baseline | argument_checklist | deontological | VALID | COMMITTED_WITH_UNCERTAINTY | 1 |
| baseline | argument_checklist | care | VALID | COMMITTED_WITH_UNCERTAINTY | 1 |
| probability_swap | standard | utilitarian | VALID | COMMITTED | 1 |
| probability_swap | standard | deontological | VALID | COMMITTED | 1 |
| probability_swap | standard | care | VALID | COMMITTED | 1 |
| probability_swap | argument_checklist | utilitarian | VALID | COMMITTED | 1 |
| probability_swap | argument_checklist | deontological | VALID | COMMITTED | 1 |
| probability_swap | argument_checklist | care | VALID | COMMITTED_WITH_UNCERTAINTY | 1 |
| competing_promise | standard | utilitarian | VALID | COMMITTED | 1 |
| competing_promise | standard | deontological | VALID | COMMITTED | 1 |
| competing_promise | standard | care | VALID | COMMITTED_WITH_UNCERTAINTY | 1 |
| competing_promise | argument_checklist | utilitarian | VALID | COMMITTED | 1 |
| competing_promise | argument_checklist | care | VALID | NONE | 1 |

Content review must assess reproducible arithmetic, duty consistency, substantive objection answers,
and explained counterfactual sensitivity. Ledger success alone earns no argument-quality credit.

Full inspectable records: [results](results.json), [design and frozen inputs](manifest.json).
