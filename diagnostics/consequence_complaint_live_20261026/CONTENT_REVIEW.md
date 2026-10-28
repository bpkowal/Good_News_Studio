# Consequence-first replies: a local factual gain, unresolved moral bridges

The added consequence record helped the blanket reply preserve the actual risks
and challenge an unsupported reliance premise. It did not establish a stronger
defense of promise override. The cooling reply still inferred unstated severity
and used an inadequately defended tie-breaker. This is a useful but limited gain,
not evidence that the reasoning problem is solved.

This is one non-blind assistant review. It is not an independent normative oracle
or a reliable estimate of a treatment effect. Neither Plan B nor giving Rowan the
blanket is declared morally wrong merely because these particular arguments are
inadequate.

## Test and controls

Four live o3 calls completed, two cases with two fresh response arms each. All
returned with `stop`. Both arms received the same original Contractualist opening,
source and frozen opposing complaint. Neither received a previous response arm,
peer argument, assistant review or intended winner. Both shared the extended
factual-review/warrant schema, low reasoning and 8192-token output ceiling.

The treatment adds the existing person-by-option matrix, source passages and a
fact-first instruction. The baseline uses the focused instruction without that
record. Structured input and instruction change together; their separate effects
are not identified. The new schema also solicits factual review in the baseline,
so this is not an exact rerun of the preceding experiment. The treatment has a
longer input. There is one sample per arm/case, not dependable evidence of
generalization.

All inputs and implementation bytes were frozen before calls. The original
arguments, assignments and source remain unchanged. [Verification](verification.json)
checks matching, hashes, source-span offsets and provider-call counts, with zero
semantic certification. [verify.py](verify.py) reproduces those checks offline.
The 113 related regression tests passed before live calls. No operative ledger,
world schema or acceptance rule changed.

## Comparison

| Dimension | Cooling, Sal consents | Blanket, prior assurance |
| --- | --- | --- |
| Actual complaint | Both answer Jo's exclusion and the twelve households' counterclaim. | Both answer Sasha's assurance-based complaint. |
| Factual fidelity | Treatment preserves Jo's exclusion and Sal's consented eight-week income loss, but still invents comparative severity and a public judgment about the households' needs. | Baseline treats Sasha's risk under A as 40%; treatment correctly states 100% under A and 40% under B and leaves detrimental reliance unestablished. |
| Priority warrant | Both invoke symmetry and numbers when severity is unknown; neither adequately establishes that uncertainty licenses B. | Both propose a qualified promise-override rule; treatment's accurate risks do not by themselves defend its priority over this actual assurance. |
| Uncertainty | Treatment says `PARTLY_ANSWERED`, but still claims B is defensible through an unproved tie-breaker. | Treatment remains provisional but says Rowan's objection “defeats” Sasha's; its own threshold remains unexplained. |

### Blanket: preserve the person's lost chance, then argue about its force

The baseline answer says the choice presents “one person's certain death versus a
40 % risk for the other,” and its comparative justification gives Sasha a 40%
risk under A. This is false: under A Sasha receives no blanket and dies. Forty
percent is her death risk **if she receives** the blanket under B. Exact citation
bookkeeping did not catch this: the baseline has no binding findings.

The consequence-first reply instead states:

> Sasha would die with certainty under Action A and faces only a 40 % risk under Action B.

It also says the source does not establish that Sasha “had time to form plans or
take actions in reliance on Ellis's promise.” That is a useful distinction:
absence of evidence of detrimental reliance neither proves its absence nor
erases the actual assurance. The original burden matrix already contained the
correct risk cells. This gain comes from exposing that information to the reply,
not a newly authored case-specific probability rule.

However, the normative argument remains insufficient. It contrasts its qualified
override rule with “keep-the-promise-no-matter-what.” Sasha's complaint need not
endorse that absolute rule; she can accept some overrides and reject this one.
The reply still must explain why **this** 20-point difference defeats a prior
assurance when both excluded individuals face certain death. It must defend the
priority of the different lost chances rather than redescribe Rowan alone as
facing lethal sacrifice. It also asserts that Rowan could not have anticipated
or mitigated his position; that knowledge is not supplied. A proposed duty of
apology/reparation does not establish that it can compensate Sasha's death or
settle her rejection.

Thus the treatment better preserves the objection's factual basis but does not
establish that its proposed priority is one Sasha must accept.

### Cooling: an unknown comparison still becomes an asserted tie-breaker

The treatment avoids the baseline's assertion that households are “now scheduled”
to receive protection. It retains the consented economic loss and correctly
identifies the twelve households who would gain cooling under B and lose that
opportunity under C. But it says those homes were selected “precisely because
their exposure was judged serious enough to merit public aid.” No such public
assessment is stipulated.

It further concedes that C “imposes no non-consensual basic-needs loss on any
single person.” The source does not establish the medical severity of leaving
the other households uncooled. Sal's consent concerns the shop closure; it cannot
establish the absence of basic-needs loss to those households. Missing severity
does not prove moderate severity, equality or a tie.

The decisive bridge is still: because neither complaint clearly overrides the
other, assisting more people is permissible. That is a proposed normative rule
requiring justification, not a consequence of uncertainty. The final assertion
that no individual can “decisively prove” B unreasonable also shifts the burden
away from justifying B. Its priority-premise sentence about permitting several
people to forgo a benefit does not cleanly support the later defense of B.

Calling the complaint `PARTLY_ANSWERED` is more restrained than the baseline's
`ANSWERED`, but the reply has not correspondingly limited the decisive inference.

## Binding findings and limits

Three of four replies have review-pointer or quote-binding findings. Some target
`/complaint` even though the saved assignment nests it at
`/raw_assignment/complaint`; cooling treatment quotations contain ellipses or
extra quotation marks. These remain saved, uncorrected findings. They do not
prevent reading or assessing the answers, and a populated correction field is
not treated as factual verification. Conversely, the baseline blanket reply's
clean bindings do not validate its mistaken risk comparison.

All recommendations remain provisional and unchanged: B in cooling and A in
blanket. Three replies self-report `ANSWERED`; cooling treatment reports
`PARTLY_ANSWERED`. No objection was retired on those labels.

The refinement is now available in the experimental follow-up runner. Its
remaining limit is substantive: a model can preserve the correct consequence
cells and still jump from an unresolved comparison to its preferred norm. A
further useful challenge would target that exact priority warrant and ask what
survives without it, rather than request another general elaboration. That
additional round was not run here.

## Unedited outputs

| Case | Consequence record | Focused baseline | Consequences first |
| --- | --- | --- | --- |
| Cooling, consent | [Table](cooling_contributor_consents/consequences.md) | [Reply](cooling_contributor_consents/focused_baseline/reply.md) | [Reply](cooling_contributor_consents/consequences_first/reply.md) |
| Blanket, assurance | [Table](blanket_prior_assurance/consequences.md) | [Reply](blanket_prior_assurance/focused_baseline/reply.md) | [Reply](blanket_prior_assurance/consequences_first/reply.md) |

Adjacent JSON files preserve the exact requests, raw provider responses, all
premise annotations and binding findings. [Transport summary](RESULT.md).
