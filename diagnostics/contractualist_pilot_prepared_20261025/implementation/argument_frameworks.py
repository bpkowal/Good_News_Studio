"""Framework-local tasks; shared argument mechanics do not prescribe a theory.

Contractualist structure informed by Marcos-Vidal, Abbo & Belpaeme (2026),
arXiv:2608.01937v1, sections 5.1-5.3. See CONTRACTUALIST_RESEARCH.md for the
attribution, adaptations, and deliberately unimplemented formal semantics.
"""

VERSION = 'argument-framework-instructions/0.1'

FRAMEWORK_GUIDANCE = {
    'utilitarian': (
        'Evaluate consequences for everyone affected and defend the welfare comparison. '
        'Distinguish restricted endpoints such as expected survivors or beneficiary counts '
        'from total welfare. State assumptions about unprovided effects or comparability '
        'as assumptions. Address the strongest objection to aggregation or priority.'),
    'deontological': (
        'Evaluate candidate maxims, applicable duties, consent and competing obligations. '
        'Distinguish a stated rule from its moral authority. Defend any priority among duties '
        'or retain the conflict. For instrumental-use claims distinguish causal contribution '
        'from an end, instrumental belief and purposeful employment; using someone as a '
        'means does not alone establish merely-as-means treatment or prohibition. '
        'A stated source purpose is not necessarily the sole purpose.'),
    'care': (
        'Evaluate concrete needs, vulnerability, relationships, responsiveness and the '
        'competence of care. Do not invent special relationships or turn a sequence of '
        'care phases into an automatic priority rule. Explain how excluded people are '
        'considered and defend any priority among their claims.'),
    'contractualist': (
        'Evaluate general principles by their justifiability to affected individuals. '
        'A principle may permit multiple actions and an action may be permitted by multiple '
        'principles. Form individual, personal reasons expressed in terms available to anyone '
        'in a relevantly similar position. Include excluded recipients and involuntary '
        'contributors. Tie each personal reason to one individual or generic individual '
        'position, one relevant value and the principle/action challenged or supported. '
        'These are reconstructed arguments, not reported preferences or speech. Compare '
        'objections to alternative principles with explicit grounds and warrants. Neither '
        'count reasons nor maximize a minimum welfare score. Do not invent strength weights '
        'or assume actual unanimity or that any objection is a veto. Defend which rejections '
        'are reasonable or preserve the comparison as unresolved. For risk distinguish '
        'ex ante prospects from ex post burdens. An unfulfilled assurance does not by itself '
        'establish wrongful betrayal or actual reliance.'),
    'virtue': (
        'Evaluate practical wisdom, relevant virtues and their application to the available '
        'actions. Explain competing considerations without treating a virtue label as proof '
        'of good character, factual success, or a uniquely right action.'),
    'rawlsian': (
        'Identify the institutional rule or basic-structure question, if any. State the '
        'limits of extending Rawls to an individual emergency. Keep affected persons and '
        'their distributions distinct; a recipient success probability is not the minimum '
        'prospect of everyone affected. Defend the metric and interpretation used.'),
}

SHARED_GENERATION = (
    'Construct small framework-grounded arguments from the exact source and the supplied '
    'framework instruction. Identify factual grounds, interpretive assumptions, evaluative '
    'warrants, inferential support and the strongest relevant objection. Give a substantive '
    'response or identify the precise unresolved premise and a relevant reversal condition. '
    'Preserve polarity, uncertainty and the distinction between possible and certain outcomes. '
    'Do not invent facts or treat a framework commitment as source evidence. '
    'Use exact source quotations and explicit support routes; Python assigns canonical IDs. '
    'Use FACTUAL_PREMISE for descriptive evidence, PRINCIPLE for a proposed general norm, '
    'INTERPRETIVE_PREMISE for a defeasible reading, and NORMATIVE_PREMISE for an evaluative '
    'assumption. MAXIM is available when the framework calls for it, never mandatory. '
    'Declare assumption roots before statements referring to them. Assumptions remain '
    'conditional; citations establish traceability, not semantic entailment. '
)

SHARED_CRITIQUE = (
    'Review this argument against the same source and assigned framework. The proposer may '
    'be right. Assess specific support links: source entailment, polarity, scope, factual '
    'applicability, and the evaluative or comparative warrant. Do not demand another '
    'framework\'s vocabulary or inferential method. Supply small explicit revisions: '
    'withdraw, qualify, supersede, assert or add_support. Changed statements need new handles; '
    'do not rewrite old statements. Preserve good original reasoning and independent support '
    'routes. Explain the strongest remaining conclusion and address the strongest relevant '
    'objection. Conditional assumptions must remain conditional in dependent conclusions. '
    'Do not manufacture facts or a working physical mechanism. '
)


def framework_instruction(source):
    """Explicit identity selects a task; legacy free-text commitments still work."""
    identity = source.get('framework_id', source.get('framework', ''))
    return FRAMEWORK_GUIDANCE.get(identity, 'Apply the supplied framework commitment; '
                                  'do not assume a particular theory from this shared format.')
