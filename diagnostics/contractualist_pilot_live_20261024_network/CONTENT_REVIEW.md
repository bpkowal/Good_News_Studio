# First neutral-adapter cooling probe

Four independent o3 calls completed, one per framework, using the same source,
three action options, common schema, low reasoning setting and 8192-token ceiling.
No peer answers, repair calls, collective verdict or world admission occurred.
All four responses preferred B. This is one non-blind assistant review, not an
independent normative assessment or evidence of general framework superiority.

## Technical results, separately

All four provider outputs were complete JSON responses. All four compiled only
partially: Utilitarian had 19 binding/construction issues, Deontological 18, Care
1, and Contractualist 41. Counts include cascading and repeated bindings; they
are not counts of independent reasoning failures or a moral-quality score.
Good statements and support routes survive alongside every raw proposal.

- Utilitarian used SOURCE for derived comparisons and ASSUMPTION for conclusions
  with premises. These operations have different meanings in the existing compiler.
- Deontology supplied a quotation with an internal ellipsis; it was not an exact
  evidence span. Some remaining statement references were disconnected.
- Care's comparison did not depend on all the reasons its annotation named.
- Contractualism supplied self-dependent source/assumption routes and referred
  to reason/conclusion handles it had not declared. Thus its prose is readable,
  but much of its intended support graph is missing.

The shared format instructions have since been clarified for all frameworks:
explicit empty-input roots, declared handles, contiguous quotations, operation
meanings and terminal judgments. No original response has been rewritten and
there has been no fresh generation to improve these results. That clarification
is not yet evidence of better live binding. `implementation_at_execution/`
preserves the code used here, verified against hashes frozen before all calls.

## Content findings

**Utilitarian.** It treats homelessness and uncompensated livelihood loss as
outweighing marginal cooling benefits. This can support a conditional argument
for B, but the source does not supply welfare magnitudes. Calling Jo's need only
comfort, and inferring that absence of stated lethality favors B, overstates the
evidence. Missing medical risk is unknown, not evidence of minor risk.

**Deontological.** It identifies refusal, displacement and involuntary burdens,
but invokes categorical priority of stringent rights over aid without adequately
defending the particular rights, exceptions or duty comparisons. The claim that
using Mira's home necessarily treats Mira merely as a means is underargued.
This remains a framework argument needing justification, not a source fact.

**Care.** It recognizes Jo's strongest complaint and uncertainty about Sal's
economic margin. However, it describes Jo's problem as temporary discomfort and
invokes Sal's continued caring relationships, neither established in the source.
It should retain a conditional comparison rather than assert those facts.

**Contractualist.** It explicitly compares Mira's, Sal's and Jo's complaints and
names competing principles, which is a useful form for the proposed specialist.
Its central conclusion nevertheless fails on its own terms. It says B is the
only option that no party can reasonably reject, while leaving Jo's objection
unresolved. Its answer says B "avoids imposing either one" of Jo's and Sal's
burdens, but B leaves Jo's home uncooled. It cannot use an unresolved B-versus-C
comparison as proof that B is nonrejectable. Its claim about longer-term business
jeopardy is a hypothesis, not an established effect of the stated closure.

The contractualist's individual-complaint form is identifiable, but this sample
does not yet establish a distinct *defensible* contribution beyond the other
agents. A different reasoning structure can expose an inference error; merely
using that structure does not make the inference good. Agreement on B proves
neither quality nor redundancy. The warranted next comparison remains the
prewritten paired consent/assurance cases, not extra calls until B looks better.

## Inspect the preserved records

- [Technical summary](RESULT.md)
- [Utilitarian raw answer](cooling_three_options/utilitarian/response.json)
  and [partial argument graph](cooling_three_options/utilitarian/candidate_arguments.md)
- [Deontological raw answer](cooling_three_options/deontological/response.json)
  and [partial argument graph](cooling_three_options/deontological/candidate_arguments.md)
- [Care raw answer](cooling_three_options/care/response.json)
  and [partial argument graph](cooling_three_options/care/candidate_arguments.md)
- [Contractualist raw answer](cooling_three_options/contractualist/response.json)
  and [partial argument graph](cooling_three_options/contractualist/candidate_arguments.md)

The preceding sandboxed run in `../contractualist_pilot_live_20261024/` recorded
four connection errors and no model replies. It remains separate from this run.
