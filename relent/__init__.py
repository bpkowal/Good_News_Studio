"""RelEnt — portable relational entailment kernel.

Calculator island for operator-scope and relational rules (precision,
equivalence, status transfer). Does not own world state, deliberation,
or LLM calls. Parliament consumes via ``global_workspace.relent_adapt``.
"""
from __future__ import annotations

from .algebra import (
    RelEdge,
    closure_edges,
    derived_only,
    directionality_errors,
    forbidden_composites,
    function_transfer_errors,
    relational_entailment_errors,
)
from .operators import (
    AllOf,
    AnyOf,
    Conditional,
    ExceptionRule,
    Fact,
    Modal,
    Not,
    consequent_licensed,
    evaluate,
    operator_from_dict,
    operator_to_dict,
)
from .paraphrase import (
    action_families_conflict,
    licensed_action_paraphrase,
    status_conserving_paraphrase_errors,
)
from .precision import quantity_precision_escalation_errors
from .quantity_typing import (
    classify_quantity_span,
    magnitude_quantity_spans,
    pseudo_quantity_errors,
    sanitize_recorded_quantities,
)
from .relations import (
    RELATION_SPECS,
    RELATION_TAGS,
    RelationSpec,
    RelationTag,
    quantity_may_transfer,
    relation_properties,
    status_transfers,
)
from .scope import (
    and_not_inheritance_errors,
    exception_scope_errors,
    modal_scope_errors,
    operator_scope_conservation_errors,
)

__all__ = [
    "RELATION_SPECS",
    "RELATION_TAGS",
    "RelEdge",
    "RelationSpec",
    "RelationTag",
    "AllOf",
    "AnyOf",
    "Conditional",
    "ExceptionRule",
    "Fact",
    "Modal",
    "Not",
    "action_families_conflict",
    "and_not_inheritance_errors",
    "classify_quantity_span",
    "closure_edges",
    "consequent_licensed",
    "derived_only",
    "directionality_errors",
    "evaluate",
    "exception_scope_errors",
    "forbidden_composites",
    "function_transfer_errors",
    "licensed_action_paraphrase",
    "magnitude_quantity_spans",
    "modal_scope_errors",
    "operator_from_dict",
    "operator_scope_conservation_errors",
    "operator_to_dict",
    "pseudo_quantity_errors",
    "quantity_may_transfer",
    "quantity_precision_escalation_errors",
    "relation_properties",
    "relational_entailment_errors",
    "sanitize_recorded_quantities",
    "status_conserving_paraphrase_errors",
    "status_transfers",
]
