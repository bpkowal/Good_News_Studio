# Burden-preserving comparison pilot: content review

The 16 planned live o3 generations completed: four cases, four frameworks, one
independent response per cell. There were no repairs, peer answers, collective
votes, or world-state admission. All eight cooling answers selected B. All eight
blanket answers selected Rowan (A). Agreement is not the success criterion.

The intervention produced a narrower improvement, not a general cure. The
Contractualist now retains Jo's unmet cooling need and calls B provisional in
the original cooling case. All eight blanket position tables preserve both
people's mortality risks, including risk that remains for the recipient. But
priority arguments remain weak, several complaints are misattributed or replaced,
and provisionality labels do not consistently constrain the surrounding prose.

This is one non-blind assistant review, not an independent normative oracle.
The authors of the scenarios also designed the scaffold. The before/after cooling
comparison changes format instructions and reasoning scaffolding together and
uses one sample per condition; it is not a treatment-effect estimate. No moral
winner or preferred distribution was stipulated as a scoring answer.

## What was changed, and what was not

Every specialist received the same common schema, burden-position procedure,
exact source, available options, o3 model, low reasoning setting and 8192-token
ceiling. Framework-local instructions intentionally differ. All requests and
implementation bytes were frozen before any call. Evaluator notes, pairing
labels, prior model answers and the earlier criticism were excluded from requests.

In cooling, A cools twenty households but makes Mira homeless against her refusal;
B cools twelve and displaces nobody; C cools Jo's most exposed home but closes
Sal's shop for eight weeks. Jo remains uncooled under A and B. The contrast changes
only Sal's refusal to informed, unpressured agreement. In the blanket cases,
A gives Rowan an 80% survival chance and leaves Sasha with zero; B gives Sasha
60% and leaves Rowan with zero. The contrast replaces the prior assurance to
Sasha with an explicit absence of any assurance.

## Contractualist results

| Case | Stated choice | What improved or was preserved | Main remaining problem |
| --- | --- | --- | --- |
| Cooling, refusal | B, provisional | Jo remains uncooled in the matrix; B versus C remains explicitly unresolved | The strongest objection is attributed to Sal even though its burden arises under C, not B. The answer then dismisses that objection instead of answering Jo. |
| Cooling, consent | B, provisional | Sal's agreement changes the proposed warrant while income loss remains PRESENT | The answer appeals to cumulative burdens and unsupported vulnerable household members. It calls B uniquely defensible among settled comparisons while leaving its competitor open. |
| Blanket, assurance | Rowan, provisional | Individual foregone prospects and Sasha's assurance remain visible | The reply asserts that losing 80 points outweighs losing 60 plus the assurance, without adequately defending the priority. Regret and explanation are added as conditions without showing that they discharge the complaint. |
| Blanket, no assurance | Rowan, unique | The two mortality distributions are correctly retained; no promise is invented | The conclusion claims uniqueness while the residual disagreement says the decisive 20-point difference remains open to reasonable dispute. The structural metadata does not capture that prose contradiction. |

The original cooling reply said B avoids both Jo's and Sal's burdens and is the
only nonrejectable option. The new reply does not repeat the false claim that B
cools Jo. Its B-versus-C comparison explicitly describes Jo's relief under C and
new burdens on Sal and twelve households. That is a useful local change. However,
its `remaining_complaint_ids` retains Sal's complaint against C, while the strongest
objection to B should address Jo's continuing need. The objection field starts:

> Sal can reasonably reject Plan B’s priority

Sal's own income burden is absent under B according to that same reply. A personal-complaint method
cannot answer Jo simply by showing Sal is unharmed by the preferred option.

In the consent variation, the reply explicitly recognizes Sal's agreement and
retains the financial cost. Yet it compares Jo with the "many moderate burdens"
and a "cumulative burden" on households. A defensible contractualist account may
argue that numbers matter in some circumstances, but this reply does not supply
that argument; it implicitly imports aggregation. The source also does not state
that the uncooled households contain especially vulnerable members.

In the assurance case, the individual opportunity-loss comparison is recognizably
different from summing expected survivors. Nevertheless, the agent's normative
premise merely says that a substantially greater expected loss supports rejection.
It does not explain why twenty percentage points are substantial enough here to
defeat a specific assurance. The preference may be defensible, but it is not
established by writing that premise down.

In the no-assurance case, the source-supported distributions are correct and the
comparative premise is explicit. But the reply says:

> Whether a 20 % difference in lost survival chance is enough to make one complaint decisively stronger remains open to reasonable dispute.

That unresolved premise limits the same reply's claim of a uniquely defensible
answer. The model marked the pair favorable rather than unresolved, so the
deterministic annotation check reports no mismatch. This is a concrete example
of structurally consistent metadata failing to guarantee consistent reasoning.

## Other frameworks and contrast sensitivity

**Utilitarian.** Both cooling replies choose B using assumptions about the welfare
costs of homelessness, income loss and continued heat. Some assumptions are made
explicit, which is useful, but absent evidence of lethality is still treated as
support for a lesser heat burden. The consent reply does not acknowledge Sal's
agreement in its factual statements or positions. Both blanket replies correctly
compare 0.8 versus 0.6 expected survivors but overextend that restricted endpoint
to welfare. The assurance reply also records Ellis's distress as PRESENT, although
the source gives no mental-state evidence, and assumes promise costs are smaller
than the survival gain. The no-assurance reply claims uniqueness under an explicit
expected-survivor norm; that norm is narrower than total utilitarian evaluation.

**Deontological.** The cooling pair recognizes consent: involuntary livelihood
interference is a proposed prohibition in the refusal case; in the consent case
the agent acknowledges that C involves consensual hardship. It still gives B a
unique ranking by favoring aid to twelve over Jo without adequately deriving that
priority from its deontological commitments. With the promise present, it recognizes
fidelity but simply declares the probability difference sufficient to override it.
Without the promise it invokes effective beneficence and considers randomization,
but extends absence of an assurance into absence of every other special duty.
No automatic priority of rescue or fidelity is established by this review.

**Care.** The cooling refusal reply directly acknowledges Jo's unmet need but
asserts priority for avoiding newly imposed harms. The consent reply is more
balanced: it keeps C defensible, preserves the economic cost despite consent,
and names the unresolved severity comparison. Both blanket replies conflate
greater rescue effectiveness with greater urgency or vulnerability: a higher
survival chance with the blanket does not establish that Rowan's need is more
acute. The no-assurance reply assumes relational equality from missing relational
facts. Unknown relationships must remain unknown, even if equality is adopted
as a conditional interpretive assumption.

## Separate 0–2 rubric assessment

These are fallible assistant judgments using the prewritten rubric: 0 missing or
wrong, 1 partial, 2 explicit and adequately defended for the particular dimension.
No total score is used. Contrast ratings assess a framework's paired reasons and
are repeated in both rows, not counted as independent observations. A 2 for
complaint recognition does not certify the agent's answer or moral priority.

| Case / framework | Fidelity | Principle | Complaint | Comparison | Defeater | Contrast |
| --- | --- | --- | --- | --- | --- | --- |
| Cooling refusal / Utilitarian | 1 | 1 | 1 | 1 | 1 | 0 |
| Cooling refusal / Deontological | 1 | 1 | 1 | 1 | 1 | 2 |
| Cooling refusal / Care | 1 | 1 | 2 | 1 | 1 | 2 |
| Cooling refusal / Contractualist | 1 | 1 | 1 | 1 | 0 | 1 |
| Cooling consent / Utilitarian | 1 | 1 | 1 | 1 | 1 | 0 |
| Cooling consent / Deontological | 1 | 1 | 1 | 1 | 1 | 2 |
| Cooling consent / Care | 2 | 1 | 2 | 1 | 2 | 2 |
| Cooling consent / Contractualist | 1 | 1 | 1 | 0 | 1 | 1 |
| Blanket assurance / Utilitarian | 1 | 1 | 1 | 1 | 1 | 1 |
| Blanket assurance / Deontological | 1 | 1 | 1 | 1 | 1 | 1 |
| Blanket assurance / Care | 1 | 1 | 1 | 0 | 1 | 1 |
| Blanket assurance / Contractualist | 1 | 1 | 2 | 1 | 1 | 1 |
| Blanket no assurance / Utilitarian | 1 | 1 | 1 | 1 | 1 | 1 |
| Blanket no assurance / Deontological | 1 | 1 | 2 | 1 | 1 | 1 |
| Blanket no assurance / Care | 1 | 1 | 2 | 1 | 1 | 1 |
| Blanket no assurance / Contractualist | 2 | 1 | 2 | 1 | 0 | 1 |

Contractualism contributes an identifiable individual opportunity-loss comparison
in the blanket cases. It has not yet demonstrated a distinct, adequately defended
priority argument beyond the other frameworks. The cooling-consent answer's
unargued aggregation further weakens the claim of theoretical independence.

## Technical results are not moral results

All 16 provider completions finished normally with JSON responses. One argument
had zero binding issues; fifteen compiled partially. Fourteen had comparison
annotation findings, including reversed introduced/relieved lists, omitted
retained burdens, and wrong complaint references. These are not fourteen rejected
worlds or a moral failure rate. Counts include cascading and repeated bindings.
No submission was silently repaired, erased, admitted to a world, or rerun.

After the cohort, `complaint-comparison/0.2` added a deterministic projection of
changes from each saved matrix. This eliminates the need for a reader to trust
the model's duplicated transition lists, while preserving its raw lists and
ranking. Unknown or duplicate matrix cells remain unknown, and quantitative
descriptions survive. This zero-call post-hoc projection does not verify the
matrix against the source or improve the agent's original moral argument.

Inspect [all technical outputs](RESULT.md), [verification](verification.json),
the original per-cell `candidate_arguments.md` Mermaid graphs, and
[derived positions for the Contractualist cooling comparison](comparison_projection/cooling_three_options__contractualist.md).

## Consequence for the next increment

The scaffold helps preserve burdens but does not solve normative comparison.
Asking the model to duplicate reference bindings and transition calculations
also creates avoidable errors. The next generative comparison stage should consume
compiled option positions rather than recreate them, and respond to the actual
affected person's strongest objection using the exact priority premise at issue.
It should produce a defended conditional conclusion when it cannot establish a
priority, rather than turn uncertainty into unique defensibility. That further
model stage has not been run here. No roster replacement is supported yet.
