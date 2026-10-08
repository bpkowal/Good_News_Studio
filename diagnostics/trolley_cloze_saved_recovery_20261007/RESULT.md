# Saved trolley recovery: verification during concurrent implementation

The exact original live input and saved Cloze completions were replayed, without
new provider calls. Concurrent work was actively changing the Cloze implementation;
this verification did not overwrite those edits. The added test file freezes the
failed input's action boundaries, affected parties, separated harm/benefit outcomes,
condition references, and real native admission.

## Latest replay

- Both omission_harm and conditional_outcome candidates pass native admission.
- The old full-sentence action labels are gone.
- Killing and sparing are distinct effects instead of one combined adverse effect.
- The omission branch retains a process parent, rather than connecting harm
  directly to a person's act. No native validators were weakened by this work.
- The source-copied omission label remains `does not`; the visible graph does not
  yet explain its antecedent clearly to a reader.

See [all candidate graphs](candidate_graphs.md), [native admission results](admission_results.json),
and [recovered proposal records](recovered_candidates.json). Native admission is
separate from semantic completeness; no moral framework run was performed here.

## Regression results and open issues

The latest combined suite ran 52 tests: 47 passed, four skipped, one failed.
All five added trolley tests pass, including native admission. The remaining
failure is the primitive-composition extra-state selection test: a six-effect
proposal was selected where the test expects the seven-effect proposal preserving
the warning-signal outcome. It must be resolved before calling the broader suite
green. See [regression log](regression.log).

The admitted trolley proposal still labels `the five` and `the one` as PERSON,
and omits the trolley and track locations from its topology. Admission now works
for this saved replay, but full scene and quantity/reference coverage remain
unfinished. The earlier rejected live run remains unchanged.
