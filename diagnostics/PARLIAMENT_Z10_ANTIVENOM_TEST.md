# Held-out antivenom allocation test

Scenario: a clinician has one vial of antivenom, two patients need it, either may
receive it but not both, and each has a stipulated recovery outcome if treated.

The initial replay failed because the allocation compiler and validator recognized
`one dose` but not `one vial`. Adding `vial` to a vocabulary would have repeated the
scenario-lock-in problem. The refinement instead composes Parliament's typed
`ExclusiveAllocationEvidence` with source-cited singular resource structure.

After refinement, admission returned `COMMITTED` with six effects and four causal
links. Each action contains a direct antivenom transfer, a downstream nonreceipt
complement for the other patient, and the stipulated recovery outcome. The
nonreceipt effects preserve `one vial`, cite their corresponding transfer parent,
and use `EXCLUSIVE_ALLOCATION_COMPLEMENT`.

The Z10 package was generated successfully. A live model draft was not generated:
the configured Parliament backend stopped before making a request because
`OPENAI_API_KEY` was absent. The committed result is therefore a fresh compiler
and admission replay, not evidence that first-pass model generation has improved.

Z10 also produced one stripping reconstruction and an operator-scope question for
`but not both`. Here `both` closes the recipient choice; it is not an omitted
object receiving a reconstructed `administer` event. Because Z10 is frozen, the
candidate-to-atom compiler should recognize this quantifier/choice construction
and leave that reconstruction unselected. This is the next identified refinement
in candidate conversion, separate from allocation topology.
