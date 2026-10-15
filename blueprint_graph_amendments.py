"""Add source-backed conditional branches without fixing the graph's size.

Existing builders stay unchanged. Amendments are separate candidate attempts;
the original attempt remains available if an amendment fails admission.
"""
from copy import deepcopy
import re

from blueprint_evidence_graph import bind_world_atoms
from blueprint_kind_license import license_kind
from blueprint_proposal_contract import validate_proposal

# Vocabulary may veto a claimed physical_state. It may not assign PROCESS.
_PROCESS_PREDICATES = {"stop", "start", "open", "close", "fail", "activate"}
_PROCESS_BEARER = re.compile(
    r"\b(?:signals?|alarms?|gates?|doors?|pumps?|brakes?|engines?|"
    r"valves?|machines?|switch(?:es)?)$",
    re.I,
)


def branch_world(text, branch):
    from blueprint_cloze_chooser import _conditional_graph, _outcome_reading
    from blueprint_physical_scene import mechanism_from_source, _nominal
    doc, mechanism = mechanism_from_source(text)
    branch = deepcopy(branch)
    if mechanism:
        branch['controllable_process'] = mechanism['instrument']
    world, notes = _conditional_graph(text, branch)
    if mechanism:
        # Expand only a bare negative auxiliary against the unique explicit
        # positive conditional operating the same instrument. Keep its source
        # effect and condition verbatim, as in the macro's VP-ellipsis path.
        antecedents = [(t, obj) for t in doc if t.lemma_ == mechanism['action_predicate']
                       and not any(c.dep_=='neg' for c in t.children)
                       and any(c.lower_=='if' and c.dep_=='mark' for c in t.children)
                       for obj in t.children if obj.dep_ in {'dobj','obj'}
                       and obj.lemma_ == mechanism['instrument_head']]
        for action in world['actions']:
            if branch.get('actor_resolution_evidence'):
                action['intervention'] = re.sub(r'^(?:she|he)\s+', '', action['intervention'], flags=re.I)
            if re.fullmatch(r'(?:does|do|did)\s+not', action['intervention'], re.I) and len(antecedents)==1:
                verb,obj=antecedents[0]
                action['intervention'] += ' '+verb.lemma_+' '+_nominal(obj)
    if branch.get("construction") == "physical_state":
        outcome = world["effects"][-1]
        outcome.update(predicate=branch["predicate"], polarity="NEUTRAL", effect_kind="PHYSICAL_STATE")
        # "will stop" is stipulated on this action branch, just as "will die"
        # is in the existing builder. Do not treat the branch's own "if" as an
        # independent uncertainty gate or promote may/might to certainty.
        if _outcome_reading(branch["outcome"])["modality"] == "CERTAIN":
            outcome.update(modality="CERTAIN", likelihood_qualifiers=[], condition_ids=[])
            world["causal_links"][-1].update(modality="CERTAIN", condition_ids=[])
        bearer = next(p for p in world["parties"] if p["party_id"] == outcome["party_id"])
        licensed = license_kind(
            bearer["label"], role="process", text=text, construction="physical_state")
        bearer["kind"] = licensed["kind"]
        bearer["kind_origin"] = licensed["origin"]
    return world, notes


def conditional_inventory(text, package):
    from parsing_game_S import get_nlp
    doc = get_nlp()(text)
    nodes = {n["id"]: n for n in package.get("nodes", [])}
    roles, predications, objects = {}, {}, {}
    for candidate in package.get("candidates", []):
        if candidate["type"] == "PREDICATION":
            predications[candidate["arguments"]["proposition"]] = candidate
        if candidate["type"] == "PARTICIPANT" and candidate.get("value") == "subject":
            roles.setdefault(candidate["arguments"]["proposition"], []).append(
                nodes[candidate["arguments"]["mention"]]["label"])
        if candidate['type'] == 'PARTICIPANT' and candidate.get('value') in {'object','patient'}:
            objects.setdefault(candidate['arguments']['proposition'], []).append(
                nodes[candidate['arguments']['mention']]['label'])
    from blueprint_physical_scene import mechanism_from_source, counted_antecedent, _nominal
    _, mechanism = mechanism_from_source(text)
    mechanism_actor = ''
    if mechanism:
        roots=[s.root for s in doc.sents if s.text == mechanism['sentence']]
        subjects=[c for root in roots for c in root.children if c.dep_ == 'nsubj']
        names={t.text.casefold() for t in doc if t.pos_=='PROPN'}
        if len(subjects)==1 and subjects[0].pos_=='PROPN' and names=={subjects[0].text.casefold()}:
            mechanism_actor=_nominal(subjects[0])
    branches, unresolved, seen = [], [], set()
    for candidate in package.get("candidates", []):
        if candidate["type"] != "CONDITIONAL_ON":
            continue
        args = candidate["arguments"]
        condition_id, outcome_id = args["condition"], args["consequence"]
        if not re.fullmatch(r"p\d+", condition_id):
            continue
        index = int(condition_id[1:])
        if index >= len(doc):
            continue
        sentence = doc[index].sent.text.strip().rstrip(".!?")
        leading = re.match(r"^(if\b[^,]+),\s*(.+)$", sentence, re.I)
        trailing = re.match(r"^(.+?)(?:,\s*|\s+)(if\b.+)$", sentence, re.I)
        if leading:
            condition, outcome = leading.groups()
        elif trailing:
            outcome, condition = trailing.groups()
        else:
            unresolved.append({"evidence": sentence, "reason": "conditional_shape_not_supported"})
            continue
        signature = (condition, outcome)
        if signature in seen:
            continue
        seen.add(signature)
        actors, bearers = roles.get(condition_id, []), roles.get(outcome_id, [])
        from blueprint_cloze_chooser import _outcome_reading, _outcome_patient, _branch_conjuncts
        reading = _outcome_reading(outcome)
        if reading['kind'] in {'HEALTH_OUTCOME','WELFARE_OUTCOME'} and objects.get(outcome_id):
            bearers=list(dict.fromkeys(objects[outcome_id]))
        resolved = False
        if mechanism_actor and len(actors)==1 and actors[0].lower() in {'she','he'}:
            condition_token=doc[index]
            if condition_token.lemma_ in {mechanism['action_predicate'],'do'}:
                actors=[mechanism_actor];resolved=True
        if mechanism and reading['kind'] in {'HEALTH_OUTCOME','WELFARE_OUTCOME'}:
            patient=_outcome_patient(_branch_conjuncts(outcome)[0])
            if patient:
                bearers=[counted_antecedent(text,patient) or patient]
        coordinated = bool(re.search(r'\b(?:and|or)\b',outcome,re.I))
        supported_bundle = (mechanism is not None and ' or ' not in outcome.lower()
            and all((_outcome_reading(part)['kind'] in {'HEALTH_OUTCOME','WELFARE_OUTCOME'}
                     or _outcome_reading(part)['predicate']=='spare')
                    and _outcome_patient(part) for part in _branch_conjuncts(outcome)))
        scopes = [c.get("scope", {}) for c in (
            candidate, predications.get(condition_id, {}), predications.get(outcome_id, {}))]
        contexts = [context for scope in scopes for context in scope.get("contexts", [])]
        unsupported = any(c.get("kind") in {"attributed", "questioned"} for c in contexts)
        negative_content = predications.get(outcome_id, {}).get("scope", {}).get("polarity") == "negative"
        if (len(actors) != 1 or len(bearers) != 1 or unsupported or negative_content
                or any(re.search(r"\b(?:he|she|they|it|someone)\b", s, re.I)
                       for s in actors + bearers)
                or any(s not in text for s in actors + bearers)
                or re.search(r"\b(?:unless|whether)\b|\?", sentence, re.I)
                or re.search(r"\b(?:tries|tried|decides|decided|promises|believes|reports)\b", condition, re.I)
                or (coordinated and not supported_bundle)):
            unresolved.append({"evidence": sentence, "reason": "scope_or_participants_need_resolution"})
            continue
        # Non-welfare asserted consequences may be proposed as physical_state.
        # Process predicate/bearer lists may veto that claim; they may not mint
        # a process atom from an unmatched sentence, and they never write PERSON.
        from blueprint_cloze_chooser import _outcome_reading
        predicate = nodes.get(outcome_id, {}).get("predicate", "").casefold()
        reading = _outcome_reading(outcome)
        welfare = reading["kind"] in {"HEALTH_OUTCOME", "WELFARE_OUTCOME"}
        process_claim = (
            not welfare
            and predicate in _PROCESS_PREDICATES
            and license_kind(
                bearers[0], role="process", text=text,
                construction="physical_state",
            )["kind"] == "PROCESS"
        )
        if process_claim and (reading["modality"] != "CERTAIN"
                         or re.search(r"\b(?:probably|possibly|perhaps|likely|chance|risk)\b", outcome, re.I)):
            unresolved.append({"evidence": sentence, "reason": "process_state_uncertainty_needs_scope_construction"})
            continue
        if reading["predicate"] == "outcome" and not process_claim:
            unresolved.append({"evidence": sentence, "reason": "outcome_predicate_not_supported"})
            continue
        branch = {"actor": actors[0], "condition": condition,
                         "bearer": bearers[0], "outcome": outcome,
                         "candidate_id": candidate["id"]}
        if resolved:
            # Branch values remain copied strings for the common construction
            # engine's evidence closure; the inventory records the derivation.
            branch['actor_resolution_evidence']=mechanism['sentence']
        if process_claim:
            branch.update(construction="physical_state", predicate=predicate)
        branches.append(branch)
    return branches, unresolved


def merge_world(base, addition):
    """Union compatible records. No template whitelist limits nodes or links."""
    world = deepcopy(base)
    remap, added = {}, []
    shared_actions = {a['intervention'].casefold() for a in base.get('actions', [])} & {
        a['intervention'].casefold() for a in addition.get('actions', [])}
    instrument_labels = set()
    for graph in (base, addition):
        recipients = {pid for a in graph.get('actions', []) if a['intervention'].casefold() in shared_actions
                      for pid in a.get('recipient_party_ids', [])}
        instrument_labels.update(re.sub(r'^(?:a|an|the)\s+', '', p['label'].casefold())
                                 for p in graph.get('parties', []) if p['party_id'] in recipients)
    specifications = (
        ("parties", "party_id", "P"), ("actions", "action_id", "A"),
        ("effects", "effect_id", "E"), ("conditions", "condition_id", "CND"),
    )
    def signature(collection, row):
        if collection == "parties":
            label = row.get("label", "").casefold()
            # Instrument mentions on the same intervention can be introduced
            # as "a lever" and repeated as "the lever". Do not create a second
            # recipient merely because the conditional changes its determiner.
            if row.get('kind') == 'PROCESS' and re.sub(r'^(?:a|an|the)\s+', '', label) in instrument_labels:
                label = re.sub(r'^(?:a|an|the)\s+', '', label)
            return (label, row.get("kind"))
        if collection == "actions":
            return (row.get("intervention", "").casefold(), row.get("actor_party_id"))
        if collection == "conditions":
            return (row.get("description", "").casefold(), row.get("polarity"))
        return tuple(str(row.get(k, "")).casefold() for k in (
            "action_id", "party_id", "source_proposition", "predicate", "polarity",
            "directness", "effect_kind", "modality", "condition_ids"))
    for collection, key, prefix in specifications:
        records = world.setdefault(collection, [])
        for original in addition.get(collection, []):
            row = deepcopy(original)
            for field in ("actor_party_id", "action_id", "party_id"):
                if field in row:
                    row[field] = remap.get(row[field], row[field])
            equivalent = next((r for r in records if signature(collection, r) == signature(collection, row)), None)
            if equivalent:
                remap[original[key]] = equivalent[key]
                continue
            used = {r[key] for r in records}
            number = len(records) if collection == "actions" else len(records) + 1
            while f"{prefix}{number}" in used:
                number += 1
            row[key] = f"{prefix}{number}"
            remap[original[key]] = row[key]
            records.append(row)
            added.append(f"{collection}:{row[key]}")
    # Rewrite references only on new rows, preserving original evidence records.
    for collection, key, _ in specifications:
        for row in world[collection]:
            if f"{collection}:{row[key]}" not in added:
                continue
            for field in ("recipient_party_ids", "effect_ids", "source_effect_ids", "condition_ids"):
                if field in row:
                    row[field] = [remap.get(v, v) for v in row[field]]
    # Existing actions acquire only the newly attached effects of this branch.
    for action in addition.get("actions", []):
        dest = next(r for r in world["actions"] if r["action_id"] == remap[action["action_id"]])
        for ident in action.get("effect_ids", []):
            mapped = remap[ident]
            if mapped not in dest["effect_ids"]:
                dest["effect_ids"].append(mapped)
    for collection in ("causal_links", "temporal_relations", "counterfactual_links"):
        records = world.setdefault(collection, [])
        for original in addition.get(collection, []):
            row = deepcopy(original)
            for key, value in row.items():
                if isinstance(value, str) and key.endswith("_id"):
                    row[key] = remap.get(value, value)
                elif isinstance(value, list) and key.endswith("_ids") and key != "clause_ids":
                    row[key] = [remap.get(v, v) for v in value]
            if row not in records:
                records.append(row)
                added.append(f"{collection}:{len(records)-1}")
    return world, added


def expand_candidates(text, package, blueprint, inventory=None):
    """Append amended attempts and one flexible fallback; preserve originals."""
    from blueprint_cloze_chooser import _conditional_graph, _graph, _construction_provenance, _relation_alternatives
    from blueprint_admission_core import supported_core
    if inventory is None:
        branches, unresolved = conditional_inventory(text, package)
    else:
        branches, unresolved = inventory["branches"], inventory["unresolved"]
    result = deepcopy(blueprint)
    attempts = result.setdefault("candidate_attempts", [])
    if not branches:
        result["amendment_inventory"] = {"branches": [], "unresolved": unresolved}
        return result
    def expand(proposal):
        projected, _ = supported_core(proposal)
        world = projected["candidate"]["world_model"]
        added = []
        for branch in branches:
            # A macro may already contain the full coordinated branch. Match
            # its condition, polarity, predicates and resolved patients before
            # adding records; source pronouns are not extra interventions.
            from blueprint_cloze_chooser import _branch_conjuncts, _outcome_patient, _outcome_reading
            from blueprint_physical_scene import counted_antecedent
            from z10_world_model_adapter import segment_source_clauses
            def nominal(value):return re.sub(r'^(?:a|an|the)\s+', '', value.casefold())
            branch_clauses={c['clause_id'] for c in segment_source_clauses(text)
                            if branch['condition'] in c['text']}
            negative=bool(re.search(r'\bnot\b',branch['condition'],re.I))
            party_labels={p['party_id']:nominal(p['label']) for p in world['parties']}
            existing=[a for a in world['actions'] if bool(re.search(r'\bnot\b',a['intervention'],re.I))==negative
                      and branch_clauses.intersection(a.get('clause_ids',[]))]
            def covers(action):
                effects=[e for e in world['effects'] if e['action_id']==action['action_id']]
                for part in _branch_conjuncts(branch['outcome']):
                    patient=_outcome_patient(part)
                    patient=counted_antecedent(text,patient) or patient
                    if not patient or not any(e['predicate']==_outcome_reading(part)['predicate']
                            and party_labels[e['party_id']]==nominal(patient) for e in effects):return False
                return True
            if any(covers(action) for action in existing):
                continue
            extra, _ = branch_world(text, branch)
            world, changes = merge_world(world, extra)
            added.extend(changes)
        # The same scene/omission completion used by macros applies after
        # union, when both branches are available for identity binding.
        from blueprint_physical_scene import attach_physical_scene
        from blueprint_spatial_projection import branch_choice_cells, spatial_cells
        scene_provenance = []
        world = attach_physical_scene(text, world, scene_provenance)
        projected["candidate"]["world_model"] = world
        projected.setdefault('slot_bindings', {})['branch_choices'] = branch_choice_cells(world)
        cells = spatial_cells(text, world)
        if cells:
            projected['slot_bindings']['physical_scene_cells'] = cells
        projected["assignment"] = [a["intervention"] for a in world["actions"]]
        sources = projected["candidate"].setdefault("actions", {})
        for action in world["actions"]:
            sources.setdefault(action["action_id"], {"clause_ids": action["clause_ids"],
                                                    "reason": "Copied conditional branch."})
            sources[action['action_id']]['clause_ids'] = list(dict.fromkeys(
                sources[action['action_id']].get('clause_ids', []) + action['clause_ids']))
        projected["construction_provenance"] = _construction_provenance(world, projected["clauses"], {})
        projected["construction_provenance"].extend(scene_provenance)
        projected["relation_alternatives"] = _relation_alternatives(world, projected["clauses"])
        projected["evidence_graph"] = bind_world_atoms(
            list(projected.get("evidence_graph") or []), world, projected["clauses"])
        projected["unresolved_readings"].extend(unresolved)
        return projected, added
    # Original proposal rank stays unchanged; amended variants are preferred
    # immediately before their baseline, which remains an admission fallback.
    for attempt in list(attempts):
        proposal = attempt["proposal"]
        if not proposal.get("candidate") or not proposal.get("admission_authorized"):
            continue
        amended, added = expand(proposal)
        if not added:
            continue
        amended["proposal_id"] += "_amended"
        errors = validate_proposal(amended)
        attempts.append({**deepcopy(attempt), "rank": attempt["rank"] - 0.1,
                         "selected": False, "proposal": amended,
                         "variant": "amended", "contract_valid": not errors,
                         "amendment_record": {"added": added, "errors": errors}})
    seed = {"blueprint_id": "conditional_outcome", "slots": branches[0],
            "accepted_evidence": branches[0], "items": [], "semantic_recoveries": {}}
    fallback_seed = _graph(text, seed, result["question"])
    # Seed with the same typed construction used by amendments, so a process
    # state does not leave behind the old builder's generic outcome node.
    fallback_seed["candidate"]["world_model"], _ = branch_world(text, branches[0])
    fallback, _ = expand(fallback_seed)
    fallback.update(proposal_id="flexible_graph_0", blueprint_id="flexible_graph")
    errors = validate_proposal(fallback)
    if not errors:
        rank = max((a["rank"] for a in attempts), default=0) + 1
        attempts.append({"rank": rank, "blueprint_id": "flexible_graph", "selected": False,
                         "template_status": "FILLED", "contract_valid": True,
                         "unfilled_slots": [], "proposal": fallback, "variant": "flexible"})
        if not any(p.get("candidate") and p.get("admission_authorized")
                   for p in result.get("proposals", [])):
            result["chooser_preferred_blueprint_id"] = result.get("chosen_blueprint_id")
            result.update(proposals=[fallback], graph=fallback, chosen_blueprint_id="flexible_graph",
                          status="FILLED", world_withheld=[])
            result["question"]["eligible_for_world_state"] = True
            attempts[-1]["selected"] = True
    result["amendment_inventory"] = {"branches": branches, "unresolved": unresolved,
                                      "fallback_errors": errors}
    return result
