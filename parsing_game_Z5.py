"""Z5 candidate export: "but/and not" plus a nominal is an unresolved copy.

Started from Z4 (parsing_game_Z4.py). Z4 is unchanged.

Changelog
- Z5 stripping: when the ellipsis proposer opens the stripping gate, the
  left verb is copied onto the remnant nominal. The remnant takes the
  contrasting role, polarity is negative, and the copy stays unresolved.
  Each sentence is judged on its own, so a later sentence does not hide
  the remnant. Two matching roles stay an interpretation choice (at most
  one, not exhaustive). The sentence is not rewritten. Verb-phrase ellipsis,
  sluicing, gapping, and coordination stay on the Z4 graph.
"""
import copy
import re

import ellipsis_proposer as proposer
import parsing_game_S as s
import parsing_game_Z4 as z4


def _next_id(items, prefix):
    pattern = re.compile(rf"^{prefix}(\d+)$")
    numbers = [int(match.group(1)) for item in items if (match := pattern.match(item["id"]))]
    return prefix + str(max(numbers, default=-1) + 1)


def _mention(package, index):
    if index is None:
        return None
    ident = f"m{index}"
    if any(node["id"] == ident and node["kind"] == "mention" for node in package["nodes"]):
        return ident
    return None


def _add_evidence(package, text, tokens):
    start = min(token.idx for token in tokens)
    end = max(token.idx + len(token.text) for token in tokens)
    for item in package["evidence"]:
        if item["start"] == start and item["end"] == end:
            return item["id"]
    ident = _next_id(package["evidence"], "e")
    package["evidence"].append(dict(id=ident, start=start, end=end, text=text[start:end]))
    return ident


def _nominal_tokens(head, prep):
    tokens = [head]
    if prep is not None:
        tokens.append(prep)
    for child in head.children:
        if child.dep_ in {"det", "amod", "compound", "nummod", "poss"}:
            tokens.append(child)
    return tokens


def _add_stripping(text, package):
    doc = s.get_nlp()(text)
    decision = proposer.load_proposer().propose(text, nlp=s.get_nlp(), doc=doc)
    proposals = decision.get("stripping") or []
    if not proposals:
        return
    groups = {}
    for item in proposals:
        key = (item["verb_index"], item["not_index"], item["remnant_index"])
        groups.setdefault(key, []).append(item)
    for group in groups.values():
        _add_stripping_group(text, package, doc, group)


def _add_stripping_group(text, package, doc, proposals):
    first = proposals[0]
    if doc[first["verb_index"]].text != first["verb"] or doc[first["not_index"]].lower_ != "not":
        return
    prop_id = f"p{first['verb_index']}"
    original = next((item for item in package["candidates"]
                     if item["type"] == "PREDICATION" and item["arguments"].get("proposition") == prop_id), None)
    if original is None or _mention(package, first["remnant_index"]) is None:
        return
    if any(_mention(package, item["subject_index"]) is None for item in proposals if item["subject_index"] is not None):
        return

    remnant = doc[first["remnant_index"]]
    prep = None if first["prep_index"] is None else doc[first["prep_index"]]
    support = [
        _add_evidence(package, text, [doc[first["verb_index"]]]),
        _add_evidence(package, text, [doc[first["not_index"]]]),
        _add_evidence(package, text, _nominal_tokens(remnant, prep)),
    ]
    siblings = [item for item in package["candidates"]
                if item["type"] == "PARTICIPANT" and item["arguments"].get("proposition") == prop_id]
    sibling = next((item for item in siblings if item["value"] == "subject"), siblings[0] if siblings else None)
    contexts = copy.deepcopy(sibling["scope"]["contexts"] if sibling else original["scope"]["contexts"])
    extra = []
    if sibling is not None:
        extra = [ident for ident in sibling["requires"] if ident != original["id"]]
    inherited = [ident for ident in original["requires"] if ident not in extra]

    def candidate(kind, arguments, value, requires, evidence_ids):
        item = dict(
            id=_next_id(package["candidates"], "c"),
            type=kind,
            arguments=arguments,
            value=value,
            evidence_ids=list(dict.fromkeys(evidence_ids)),
            scope=dict(polarity="negative", contexts=copy.deepcopy(contexts)),
            provenance=[dict(producer="parsing_game_Z5", version="Z5",
                             method="stripping_not_nominal", resource_ids=[])],
            assessment=dict(status="unresolved", score=None),
            requires=list(requires),
            exclusive_with=[],
        )
        package["candidates"].append(item)
        return item

    pred = candidate("PREDICATION", dict(proposition=prop_id), None, inherited + extra, support)
    created = [pred]
    base_requires = [pred["id"]] + inherited + extra
    subject_index = first["subject_index"]
    if subject_index is not None and all(item["subject_index"] == subject_index for item in proposals):
        subject_evidence = _add_evidence(package, text, _nominal_tokens(doc[subject_index], None))
        created.append(candidate(
            "PARTICIPANT", dict(proposition=prop_id, mention=f"m{subject_index}"),
            "subject", base_requires, support + [subject_evidence]))
    roles = []
    for item in proposals:
        if item["role"] == "subject":
            continue
        roles.append(candidate(
            "PARTICIPANT", dict(proposition=prop_id, mention=f"m{item['remnant_index']}"),
            item["role"], base_requires, support))
    created.extend(roles)
    if len(roles) > 1:
        for left in roles:
            left["exclusive_with"] = [other["id"] for other in roles if other is not left]
        package["choice_sets"].append(dict(
            id=_next_id(package["choice_sets"], "choice"),
            kind="interpretation",
            candidate_ids=[item["id"] for item in roles],
            selection_rule="at_most_one",
            exhaustive=False,
            evidence_ids=[support[-1]],
        ))
    package["open_questions"].append(dict(
        id=_next_id(package["open_questions"], "q"),
        kind="missing_representation",
        evidence_ids=list(support),
        candidate_ids=[item["id"] for item in created],
        question="This nominal after not is an unresolved negative copy of the earlier verb.",
        blocking_for=[],
    ))


def export_candidate_graph(text, *, package_id=None):
    package = z4.export_candidate_graph(text, package_id=package_id)
    package["producer"]["name"] = "parsing_game_Z5"
    package["producer"]["version"] = "Z5"
    package["coverage"]["limitations"] = list(package["coverage"]["limitations"]) + [
        "stripping_not_nominal_is_an_unresolved_copy"]
    _add_stripping(text, package)
    return package
