"""Small, inspectable composition pilot: choice plus conditional outcomes.

This inventory precedes macro ranking. It is not a new Parliament schema or
a general semantic parser. Existing conditional construction/union code is reused.
"""
from copy import deepcopy

from blueprint_graph_amendments import branch_world, conditional_inventory, merge_world
from blueprint_proposal_contract import proposal, candidate, world_model


def extract_inventory(text, package):
    branches, unresolved = conditional_inventory(text, package)
    candidates = {c["id"]: c for c in package.get("candidates", [])}
    conditions = []
    for branch in branches:
        source = candidates[branch["candidate_id"]]
        conditions.append({
            "type": "conditional_outcome", "branch": branch,
            "candidate_ids": [source["id"]], "requires": list(source["requires"]),
            "evidence_ids": list(source["evidence_ids"]),
            "scope": deepcopy(source["scope"]),
        })
    choices = []
    for group in package.get("choice_sets", []):
        if group["kind"] != "scenario_option":
            continue
        options = [candidates[i] for i in group["candidate_ids"]
                   if candidates[i]["type"] == "OPTION_OF"]
        if options:
            choices.append({
                "type": "choice", "choice_set_id": group["id"],
                "selection_rule": group["selection_rule"], "exhaustive": group["exhaustive"],
                "evidence_ids": list(group.get("evidence_ids", [])),
                "options": [{"candidate_id": c["id"], "arguments": deepcopy(c["arguments"]),
                             "scope": deepcopy(c["scope"])} for c in options],
            })
    supporting_ids = set()
    def include(ident):
        if ident in supporting_ids:
            return
        supporting_ids.add(ident)
        for dependency in candidates[ident]["requires"]:
            include(dependency)
    for row in conditions:
        for ident in row["candidate_ids"]:
            include(ident)
    for row in choices:
        for option in row["options"]:
            include(option["candidate_id"])
    from blueprint_semantic_coverage import extend_inventory
    return extend_inventory({"version": "primitive-composition/0.2", "package_id": package["package_id"],
            "constructions": choices + conditions, "branches": branches,
            "supporting_candidates": [deepcopy(c) for c in package.get("candidates", [])
                                      if c["id"] in supporting_ids],
            "nodes": deepcopy(package.get("nodes", [])), "evidence": deepcopy(package.get("evidence", [])),
            "unresolved": unresolved,
            "limits": "Choice and conditional outcomes; at most two constructed actions. "
                      "Other Z10 semantics remain in the original package."}, package)


def compose(text, inventory, question):
    """Build from branch data, without a family lookup or cloze slot filling."""
    from blueprint_cloze_chooser import _construction_provenance, _relation_alternatives
    from blueprint_proposal_contract import withheld_proposal
    from z10_world_model_adapter import segment_source_clauses
    from blueprint_semantic_coverage import attach_coverage
    def retain_source(proposal):
        return attach_coverage(inventory, {"candidate_attempts": [{"proposal": proposal}]})["candidate_attempts"][0]["proposal"]
    clauses = [{"clause_id": r["clause_id"], "text": r["text"]}
               for r in segment_source_clauses(text)]
    world = world_model([], [], [], [])
    for branch in inventory["branches"]:
        addition, _ = branch_world(text, branch)
        world, _ = merge_world(world, addition)
    reasons = []
    if not world["actions"]:
        reasons.append("No supported conditional branch was extracted.")
    if len(world["actions"]) > 2:
        reasons.append("This composition pilot covers at most two actions.")
    common = dict(proposal_id="primitive_composition_0", blueprint_id="primitive_composition",
                  clauses=clauses, slot_bindings={"inventory_version": inventory["version"]},
                  unresolved_readings=inventory["unresolved"], pre_world_assessment=question)
    if reasons:
        return retain_source(withheld_proposal(**common, assignment=[], unfilled_required_slots=[],
                                 construction_problems=[{"code": "pilot_scope", "message": r}
                                                        for r in reasons]))
    constructed = proposal(
        **common, status="FILLED", assignment_kind="intervention_text",
        assignment=[a["intervention"] for a in world["actions"]], selection=None,
        selection_validation={"status": "not_assessed", "contract_valid": None,
                              "reason": "Source-backed construction; no resolved Z10 selection."},
        candidate_value=candidate({a["action_id"]: {"clause_ids": a["clause_ids"],
                                    "reason": "Constructed from conditional primitive."}
                                   for a in world["actions"]}, world),
        construction_provenance=_construction_provenance(world, clauses, {}),
        relation_alternatives=_relation_alternatives(world, clauses),
        notes=["Choice alternatives are retained in the inventory. Listing options does not "
               "establish exclusivity or actual occurrence."],
    )
    # Reuse the established projection: the legacy derivation helper can append
    # alternative-harm hypotheses while closing provenance. They are not sourced
    # primitives and must not enter this pilot's candidate world.
    from blueprint_admission_core import supported_core
    return retain_source(supported_core(constructed)[0])


def append_composition(text, inventory, blueprint):
    result = deepcopy(blueprint)
    composed = compose(text, inventory, result["question"])
    attempts = result.setdefault("candidate_attempts", [])
    attempts.append({"rank": max((a["rank"] for a in attempts), default=0) + 1,
                     "blueprint_id": "primitive_composition", "variant": "composition",
                     "selected": False, "template_status": composed["status"],
                     "contract_valid": True, "unfilled_slots": [], "proposal": composed})
    # Comparison reports differences, not a requirement for agreement. IDs are
    # omitted so allocation order does not masquerade as a semantic difference.
    def summary(world):
        parties = {p["party_id"]: p["label"] for p in world["parties"]}
        actions = {a["action_id"]: a["intervention"] for a in world["actions"]}
        return sorted((actions[e["action_id"]], parties[e["party_id"]], e["source_proposition"],
                       e["predicate"], e["polarity"], e["modality"], e["effect_kind"])
                      for e in world["effects"])
    def topology(world):
        # Semantic endpoint names rather than local IDs; retain multiplicity.
        names = {p["party_id"]: ("party", p["label"], p["kind"]) for p in world["parties"]}
        names.update({a["action_id"]: ("action", a["intervention"]) for a in world["actions"]})
        names.update({c["condition_id"]: ("condition", c["description"], c["polarity"])
                      for c in world["conditions"]})
        for e in world["effects"]:
            names[e["effect_id"]] = ("effect", names[e["action_id"]], names[e["party_id"]],
                                      e["source_proposition"], e["predicate"], e["directness"])
        def normalize(value):
            if isinstance(value, str):
                return names.get(value, value)
            if isinstance(value, list):
                return [normalize(v) for v in value]
            if isinstance(value, dict):
                return {k: normalize(v) for k, v in value.items()}
            return value
        import json
        return {key: sorted(json.dumps(normalize(row), sort_keys=True) for row in rows)
                for key, rows in world.items() if isinstance(rows, list)}
    composed_world = (composed.get("candidate") or {}).get("world_model")
    comparisons = []
    for attempt in attempts[:-1]:
        from blueprint_admission_core import supported_core
        projected, _ = supported_core(attempt["proposal"])
        other = (projected.get("candidate") or {}).get("world_model")
        if other and composed_world:
            left, right = set(summary(other)), set(summary(composed_world))
            comparisons.append({"proposal_id": attempt["proposal"]["proposal_id"],
                                "effect_readings_match": left == right,
                                "topology_matches": topology(other) == topology(composed_world),
                                "only_in_blueprint": sorted(left - right),
                                "only_in_composition": sorted(right - left)})
    result["primitive_comparison"] = comparisons
    return result
