# Z10 `but not both` regression analysis

Date: 2026-10-01  
Parser: frozen Z10  
Choice-quantifier code changed: **no**

## Result

The defect is confined to the tail quantifier construction in this matrix. All 7 `either/or ... but not both` variants generated a false `stripping_not_nominal` reconstruction whose remnant is `both` and whose role is `object`. The false positive is invariant to a modal, an honorific, semicolon punctuation, an extra space, and a newline.

Genuine stripping retained 5/5 recall. VP ellipsis, gapping, and sluicing retained 4/4 expected routing. Both negative controls that spell out the predicate or place `not both` inside the clause remained free of stripping reconstructions. The existing ellipsis and Z5-Z10 suite also passed 65/65.

| Family | Pass | Fail | Interpretation |
| --- | ---: | ---: | --- |
| Exclusive-choice quantifier | 0 | 7 | Systematic false stripping reconstruction |
| Genuine stripping | 5 | 0 | Existing behavior preserved |
| VP ellipsis / gapping / sluicing | 4 | 0 | Existing gates preserved |
| Negative controls | 2 | 0 | No false reconstruction |

## Design implication

A future fix can be narrow: before the stripping copy is constructed, recognize bare `both` as the tail of an asserted alternative set when the same coordination contains an `either ... or ...` or equivalent `X or Y` pair. That path should emit choice/exclusivity semantics or a specific unresolved quantifier question, and it should not create a proposition whose object is `both`. The guard must apply only to bare quantifier `both`; `but not both parents` remains an ordinary nominal remnant in this corpus.

The frozen cases and full observations are in `diagnostics/z10_but_not_both_regression.json`. No choice-quantifier behavior was changed during this analysis.

## Implemented follow-up

The approved guard now recognizes bare `both` after a same-sentence alternative
coordination as the closing quantifier, before stripping reconstruction. All seven
exclusive-choice cases now pass. Genuine stripping remains 5/5, the other ellipsis
families remain 4/4, and both negative controls still pass. The guard does not
block a nominal such as `but not both parents`.

Post-fix observations are recorded in
`diagnostics/z10_but_not_both_postfix.json`.
