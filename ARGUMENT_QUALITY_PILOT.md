# Bounded argument-quality experiment

The pilot keeps world construction and native validators fixed. It asks whether
explicit argument instructions improve the reasons produced by the existing
specialists, rather than whether more submissions pass technical checks.

Run it with the existing native Parliament checkout and API environment:

```bash
/tmp/parliament-smoke-env/bin/python run_argument_quality_pilot.py --execute --output-dir diagnostics/argument_quality_pilot_live
```

Without `--execute`, it prepares and re-admits the frozen inputs without API calls.
Completed cells can be resumed; a partially completed cell requires a fresh output
directory. The live pilot makes 18 adapter calls: three frameworks, three cases,
two prompt arms, one call per cell. Provider-internal retries remain unchanged.
There is no peer influence, response repair, or collective verdict in this pilot.

1. Freeze clean intervention identities and the original eight medical effects.
   Re-admit the baseline and both counterfactuals through native admission.
2. Compare the standard prompt against an argument checklist using the same model,
   framework commitments, shared inputs, token ceiling, and call count.
3. Swap the survival probabilities, then retain the original probabilities and
   add an explicit promise to Cara. The promise is visible in the exact source;
   it is not invented as a medical effect or claimed to be fully graph-projected.
4. Review content independently of native admission and recommendation.

Use this evidence rubric; it is a human assessment, not a normative oracle:

| Dimension | 0 | 1 | 2 |
| --- | --- | --- | --- |
| Restricted arithmetic (utilitarian) | Wrong or missing | Correct figures, scope unclear | Reproducible calculation with endpoint and limits explicit |
| Duty reasoning (deontology) | Missing or internally contradictory | Duties classified, priority merely asserted | Consistent duties with defended priority or explicit unresolved conflict |
| Objection handling (all) | Missing | Competing claim named | Substantive response or explanation of why it remains unresolved |
| Counterfactual sensitivity (all) | Relevant change ignored | Change recognized | Explained effect on reasons, including justified retained preference |

Agreement is not evidence of quality. Disagreement is not required. Recognizing
a promise does not automatically require choosing Cara. Expected survivors do
not establish total welfare. A ledger accepted by native review can still contain
an inadequately defended moral inference.

The experiment emits frozen worlds, raw model responses, native ledgers, conflict
graphs, a result table, and checks that shared inputs and commitments match.
The effective treatment prompt is the saved native prompt followed by
`treatment_prompt.txt`. With one sample per cell, these results are exploratory;
they do not establish a reliable effect size or superiority of either prompt.
