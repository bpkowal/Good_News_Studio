"""Conservative source accounting, independent of world admission.

Retaining a parser proposal does not resolve it or establish its semantics.
Promise bundles remain outside occurrence until a supported mapping exists.
Mapped commitments keep projection_status that the content is not occurrence.
"""
from copy import deepcopy


def extend_inventory(inventory, package):
    result = deepcopy(inventory)
    result["source_fields_missing"] = [key for key in ("candidates", "nodes", "choice_sets", "open_questions", "coverage", "producer") if key not in package]
    candidates = {c["id"]: c for c in package.get("candidates", [])}
    nodes = {n["id"]: n for n in package.get("nodes", [])}

    def closure(ids):
        found = set()
        def visit(ident):
            if ident in found:
                return
            found.add(ident)
            for dependency in candidates[ident]["requires"]:
                visit(dependency)
        for ident in ids:
            visit(ident)
        return [c["id"] for c in package.get("candidates", []) if c["id"] in found]

    for row in result["constructions"]:
        ids = row.get("candidate_ids", []) or [o["candidate_id"] for o in row.get("options", [])]
        row["candidate_ids"] = closure(ids)
        row["id"] = row["type"] + ":" + (row.get("choice_set_id") or ids[0])
    for predication in package.get("candidates", []):
        if predication["type"] != "PREDICATION":
            continue
        anchor = predication["arguments"]["proposition"]
        if nodes[anchor].get("predicate", "").casefold() != "promise":
            continue
        propositions = {anchor}
        # Retain the supplied complement graph, including unresolved readings;
        # do not manufacture controllers, promisees, reliance or breach.
        while True:
            children = {c["arguments"]["child"] for c in package.get("candidates", [])
                        if c["type"] == "EVENT_LINK" and c["arguments"]["parent"] in propositions}
            if children <= propositions:
                break
            propositions.update(children)
        ids = closure([c["id"] for c in package.get("candidates", [])
                       if propositions & set(c["arguments"].values())])
        bundle = [deepcopy(candidates[i]) for i in ids]
        endpoints = {v for c in bundle for v in c["arguments"].values()}
        evidence_ids = set(e for c in bundle for e in c["evidence_ids"])
        for ident in endpoints:
            evidence_ids.update(nodes[ident]["evidence_ids"])
        for c in bundle:
            for context in c["scope"]["contexts"]:
                evidence_ids.update(context["evidence_ids"])
        question = {"id": "mapping:promise:" + anchor, "kind": "missing_world_mapping",
                    "candidate_ids": ids,
                    "question": "How should this scoped promise and its content map into Parliament? "
                                "Retaining it establishes no reliance, breach, duty, or content occurrence."}
        result["constructions"].append({
            "id": "promise:" + anchor, "type": "promise", "anchor_id": anchor,
            "candidate_ids": ids, "scope": deepcopy(predication["scope"]),
            "provenance": deepcopy(predication["provenance"]),
            "evidence_ids": [e["id"] for e in package["evidence"] if e["id"] in evidence_ids],
            "source_candidates": bundle,
            "source_nodes": [deepcopy(n) for n in package.get("nodes", []) if n["id"] in endpoints],
            "source_evidence": [deepcopy(e) for e in package["evidence"] if e["id"] in evidence_ids],
            "projection_status": "retained_not_projected", "mapping_question": question})
    consumers = {}
    for row in result["constructions"]:
        for ident in row["candidate_ids"]:
            consumers.setdefault(ident, []).append(row["id"])
    result["source_candidates"] = deepcopy(package.get("candidates", []))
    result["source_producer"] = deepcopy(package.get("producer", {}))
    result["source_choice_sets"] = deepcopy(package.get("choice_sets", []))
    result["source_open_questions"] = deepcopy(package.get("open_questions", []))
    result["source_coverage"] = deepcopy(package.get("coverage", {"status": "unknown"}))
    result["candidate_accounting"] = [
        {"candidate_id": c["id"], "consumer_construction_ids": consumers.get(c["id"], []),
         "status": "retained_in_construction" if c["id"] in consumers else "unconsumed",
         "question": None if c["id"] in consumers else {
             "id": "unconsumed:" + c["id"], "kind": "missing_construction",
             "candidate_ids": [c["id"]], "evidence_ids": list(c["evidence_ids"]),
             "question": "Which construction should preserve this " + c["type"] + " reading?"}}
        for c in package.get("candidates", [])]
    result["limits"] += " Promise source bundles are retained outside the world graph; " \
                        "candidate accounting is not a semantic completeness measure."
    return result


def attach_coverage(inventory, result):
    """Annotate every attempt without changing worlds, ranks or admission rules."""
    result = deepcopy(result)
    reports = []
    promises = [c for c in inventory["constructions"] if c["type"] == "promise"]
    unconsumed = [r for r in inventory["candidate_accounting"] if r["status"] == "unconsumed"]
    for attempt in result["candidate_attempts"]:
        proposal = attempt["proposal"]
        world = (proposal.get("candidate") or {}).get("world_model") or {}
        conditions = {c["description"].casefold().rstrip(".!?") for c in world.get("conditions", [])}
        outcomes = {c["source_proposition"].casefold().rstrip(".!?") for c in world.get("effects", [])}
        aligned = [r["id"] for r in inventory["constructions"] if r["type"] == "conditional_outcome"
                   and r["branch"]["condition"].casefold().rstrip(".!?") in conditions
                   and r["branch"]["outcome"].casefold().rstrip(".!?") in outcomes]
        report = {"proposal_id": proposal["proposal_id"],
                  "source_candidate_count": len(inventory["source_candidates"]),
                  "unconsumed_candidate_ids": [r["candidate_id"] for r in unconsumed],
                  "source_aligned_conditional_ids": aligned,
                  "retained_not_projected_construction_ids": [
                      c["id"] for c in promises if not (world.get("commitments") or [])
                  ],
                  "mapped_commitment_construction_ids": [
                      c["id"] for c in promises if world.get("commitments")
                  ],
                  "source_fields_missing": list(inventory["source_fields_missing"]),
                  "unconsumed_does_not_imply_world_omission": True,
                  "semantic_completeness": "not_assessed", "admission_is_separate": True}
        reports.append(report)
        mapped = bool(world.get("commitments"))
        constructions = []
        for construction in promises:
            row = deepcopy(construction)
            if mapped:
                row["projection_status"] = "mapped_commitment_not_occurrence"
                row["status"] = "retained_in_construction"
            constructions.append(row)
        proposal["unresolved_readings"] = [r for r in proposal["unresolved_readings"]
            if not isinstance(r, dict) or r.get("kind") != "semantic_construction_retention"]
        proposal["unresolved_readings"].append({
            "kind": "semantic_construction_retention", "coverage": report,
            "package_id": inventory["package_id"],
            "source_constructions": constructions,
            "source_choice_sets": deepcopy(inventory["source_choice_sets"]),
            "source_producer": deepcopy(inventory["source_producer"]),
            "missing_construction_questions": [deepcopy(r["question"]) for r in unconsumed],
            "source_open_questions": deepcopy(inventory["source_open_questions"]),
            "source_coverage": deepcopy(inventory["source_coverage"])})
    result["semantic_coverage"] = reports
    return result
