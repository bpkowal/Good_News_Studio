"""Project a proposal into a supported world without imposing a template topology.

The original proposal is immutable. Unsupported effects and dependent relations
remain in a diagnostic overlay, never converted into neutral or certain facts.
Native Parliament admission remains the authority for the projected world.
"""
from copy import deepcopy

from blueprint_evidence_graph import overlay_hypotheses
from blueprint_kind_license import overlay_unlicensed_atoms


def supported_core(proposal: dict) -> tuple[dict, dict]:
    projected = deepcopy(proposal)
    world = (projected.get("candidate") or {}).get("world_model")
    overlay = {"effects": [], "relations": [], "reasons": {},
               "status": "NOT_SUBMITTED_AS_WORLD", "unknown_branch_outcomes": []}
    if world is None:
        return projected, overlay
    excluded = set()
    parties = world.get("parties") or []
    for effect in world.get("effects", []):
        reasons = []
        if effect.get("derivation_assumptions"):
            reasons.append("unresolved_derivation_assumptions")
        if effect.get("outcome_type_transformation", "PRESERVED") != "PRESERVED":
            reasons.append("unsupported_outcome_transformation")
        if effect.get("derivation_operation") == "AVERTED_ALTERNATIVE_HARM":
            reasons.append("hypothesized_averted_harm")
        reasons.extend(overlay_unlicensed_atoms(effect, parties))
        if reasons:
            excluded.add(effect["effect_id"])
            overlay["reasons"][effect["effect_id"]] = reasons
    # A copied dependent effect cannot survive merely because its parent was
    # speculative. Preserve it in the overlay along with its provenance.
    changed = True
    while changed:
        changed = False
        for effect in world.get("effects", []):
            ident = effect["effect_id"]
            if ident not in excluded and excluded.intersection(effect.get("source_effect_ids", [])):
                excluded.add(ident)
                overlay["reasons"][ident] = ["depends_on_excluded_effect"]
                changed = True
    overlay["effects"] = [e for e in world.get("effects", []) if e["effect_id"] in excluded]
    world["effects"] = [e for e in world.get("effects", []) if e["effect_id"] not in excluded]
    for action in world.get("actions", []):
        action["effect_ids"] = [e for e in action.get("effect_ids", []) if e not in excluded]
    for collection in ("causal_links", "temporal_relations", "counterfactual_links"):
        kept = []
        for relation in world.get(collection, []):
            endpoints = [relation.get(k) for k in (
                "source_id", "target_id", "source_effect_id", "target_effect_id",
                "alternative_effect_id", "earlier_effect_id", "later_effect_id")]
            if excluded.intersection(e for e in endpoints if isinstance(e, str)):
                overlay["relations"].append({"collection": collection, "record": relation,
                                             "reason": "excluded_endpoint"})
            else:
                kept.append(relation)
        world[collection] = kept
    overlay["unknown_branch_outcomes"] = [
        {"action_id": e.get("action_id"), "party_id": e.get("party_id"),
         "excluded_effect_id": e["effect_id"], "status": "UNKNOWN"}
        for e in overlay["effects"]
    ]
    overlay["hypotheses"] = overlay_hypotheses(
        projected.get("evidence_graph") or [], world)
    return projected, overlay
