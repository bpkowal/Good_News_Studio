# Plan to improve draft admission

## Observed cause, 2026-10-03

The fresh trolley run in `diagnostics/trolley_relent_full_pipeline` generated
six core effects plus AV3/AV6, which assert certain averted harms on opposite
branches. `blueprint_derivation_license.apply_derivation_license` itself adds
these effects through `_attach_hypothesized_aversions`; they are not simply bad
model slot fills. Both effects retain three unresolved assumptions and a
`POLARITY_INVERTED` transformation. Parliament refuses such hypotheses as
world effects. The proposal envelope nevertheless marks admission authorized.

A read-only probe on a deep copy removed AV3/AV6 and their action-effect and
counterfactual references. All six core effects were retained. The same strict
parser passed with completeness required, and `_admit_action_source_rows`
returned `status=COMMITTED`, `world_model_status=COMMITTED`. No validator was
relaxed and no source statement or outcome was added. This demonstrates the
fix for this failure, not a corpus-wide success rate.

The older admitted trolley trace has the same speculative benefits but empty
assumption lists. Its successful replay does not establish semantic support for
those additions. Do not restore success by clearing their assumptions.

## Implementation order

1. **Produce a source-supported core plus a hypothesis overlay.** Keep the
   Parliament 1.3 schema unchanged. Store proposed averted harms, assumptions,
   alternative relations, and provenance in the proposal sidecar, and display
   them distinctly in candidate graphs. Submit supported effects and discharged
   structural derivations. Do not silently delete hypotheses or change their
   provenance. Missing branch outcomes remain unknown, not neutral or beneficial.

2. **Compile shared structure deterministically.** Let the model propose copied
   spans, references, role assignments, and interpretations. The compiler assigns
   IDs, joins references, binds quantities and evidence, and builds required
   action/process/outcome paths from those inputs. Remove automatic speculative
   completion from normalization; normalization should not invent outcomes.
   Reuse Z10 analyses and current constructions rather than introducing a schema
   for each scenario.

3. **Admit multiple candidates before selecting.** Run the existing native
   admission path on every filled blueprint core, record results, and select the
   highest-ranked admitted candidate. Preserve interpretation differences and
   show every graph. A candidate with removed speculative overlays is a distinct
   projection with an explicit record, not an invisible repair. If no candidate
   admits, return the specific unresolved construction rather than fabricate
   effects or loop indefinitely.

4. **Target remaining gaps.** Separate mechanical construction failures from
   missing source facts and ethical disagreement. Mechanical failures get
   deterministic fixes; wording/reference ambiguities get alternative fillings;
   absent facts remain explicitly unknown. Test identity consistency, instrument
   versus affected-person roles, and unsupported process paraphrases before
   expanding the blueprint library. Admission alone is not semantic accuracy.

5. **Measure fresh runs end to end.** Freeze source-only trolley, explicit
   survival trolley, rescue, and allocation examples plus wording variations.
   Test generated drafts through native admission, not only envelope validation
   or saved replay. Record first-choice admission, admission of any candidate,
   supported-outcome recall, false committed effects, quantities, roles, and
   unresolved coverage. After deterministic regressions pass, run a small fresh
   API pilot and inspect its graphs before claiming broader reliability.

## First acceptance criteria

- The rejected trolley's supported core admits unchanged under current rules.
- Its two speculative benefits remain visible outside the committed world.
- Explicitly stated survival/benefit outcomes are preserved when present.
- Unknown outcomes cannot be turned into neutral effects or survival.
- A failed preferred blueprint does not prevent an admitted alternative from
  reaching Parliament; the selection and admission records explain the choice.
- Tests exercise a newly constructed graph's admission and full replay identity,
  not merely the existence of exported files.

Implementation update, 2026-10-04: supported-core projection, hypothesis overlays,
native admission of all filled candidates, and selection of an admitted candidate
are implemented. Original candidates are retained alongside additive amendments
and a flexible conditional-branch fallback. Deterministic regression checks cover
the original trolley and a three-branch expansion through native admission and
frozen replay. Broad semantic coverage and a fresh model-generation pilot remain
future work; the fallback currently uses supported explicit conditional outcomes.
