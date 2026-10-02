"""S: evidence-conflict diagnostics and training-population early stopping.

CEM learns relation meaning/direction from reusable evidence channels. Assertion
status is a separate, conservative rule-based annotation (not calibrated
probability). To-attachments use a separate two-stage decision: deterministic
structural candidates, then a logistic classifier over valency and form
features, with unresolved as an explicit abstention. Claims describe what text
says, not verified facts about the world. No graph writes occur here.
Importing this module never trains or opens logs.
"""
import argparse
from dataclasses import dataclass, field, asdict
from functools import lru_cache
from typing import Callable, FrozenSet, Optional
import json
import sys
import tempfile
from pathlib import Path
from datetime import datetime

import numpy as np

REPAIR_ENABLED = True


@lru_cache(maxsize=1)
def load_passive_resource():
    payload = json.loads((Path(__file__).resolve().parent / "resources" /
                          "reduced_passive_Q.json").read_text())
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported reduced-passive resource schema")
    return payload


def temporal_by_object(token):
    resource = load_passive_resource()
    return (token.lower_ in resource["temporal_heads"]
            or any(t.ent_type_ in {"DATE", "TIME"} for t in token.subtree))


def assess_passive_fragment(predicate, subject, objects, preps):
    """Inspectable local checks, including rejected and unresolved alternatives.

    No absence of a lexical warning is treated as proof of semantic resolution.
    The Q construction policy remains provisional; this adds evidence accounting.
    """
    resource = load_passive_resource()
    forms = resource["forms"].get(predicate.lemma_.lower(), [])
    sources = [t for t in preps if t.head.lower_ == "by" and not temporal_by_object(t)
               and t.pos_ in {"NOUN", "PROPN", "PRON"}]
    required = dict(
        licensed_surface_form=predicate.lower_ in forms,
        affected_precedes_predicate=subject is not None and subject.i < predicate.i,
        no_competing_direct_object=not objects,
        one_non_temporal_nominal_source=len(sources) == 1,
        local_root_or_reduced_relative=predicate.dep_ in {"ROOT", "acl"},
        no_auxiliary_or_event_complement=not any(
            t.dep_ in {"xcomp", "ccomp", "aux", "auxpass"} for t in predicate.children),
        no_coordinated_source=not any(t.dep_ == "conj" for s in sources for t in s.children),
    )
    accepted = all(required.values())
    supporting = dict(
        affected_adjacent=subject is not None and subject.i + 1 == predicate.i,
        source_directly_attached=any(s.head.head.i == predicate.i for s in sources),
        parser_participle=predicate.tag_ == "VBN",
        lexical_passive_compatibility=predicate.lower_ in forms,
    )
    competing = [
        dict(reading="temporal_or_deadline", status="detected" if any(
            t.head.lower_ == "by" and temporal_by_object(t) for t in preps) else "not_detected",
             method="bounded_temporal_lexicon_and_ner"),
        dict(reading="event_valued_by", status="detected" if any(
            c.dep_ == "pcomp" for t in predicate.children if t.lower_ == "by" for c in t.children)
             else "not_detected", method="dependency_pcomp"),
        dict(reading="active_or_elliptical", status="unresolved", method="no_disambiguator"),
        dict(reading="path_or_proximity", status="unassessed", method="no_semantic_disambiguator"),
        dict(reading="instrument_or_means", status="unassessed", method="no_semantic_disambiguator"),
        dict(reading="independent_clause_boundary", status="unassessed" if not sources else "not_detected" if
             all(s.sent.start == predicate.sent.start and s.head.head.i == predicate.i for s in sources)
             else "detected", method="dependency_and_sentence_boundary_only"),
    ]
    def mention(token):
        return None if token is None else dict(text=token.text, token_index=token.i,
                                              start=token.idx, end=token.idx + len(token.text))
    return dict(schema_version=1, kind="reduced_passive_hypothesis",
                status="provisional" if accepted else "rejected",
                interpretation_status="provisional" if accepted else "rejected",
                eligible_for_world_state=False,
                affected=mention(subject), agent_or_cause=mention(sources[0]) if len(sources)==1 else None,
                predicate=mention(predicate),
                proposed_direction=dict(source_index=sources[0].i, target_index=subject.i) if accepted else None,
                required_evidence=required, supporting_evidence=supporting,
                competing_interpretations=competing,
                rejection_reasons=[k for k,v in required.items() if not v],
                affected_index=subject.i if subject is not None else None,
                source_index=sources[0].i if len(sources)==1 else None,
                original_tag=predicate.tag_, original_dependency=predicate.dep_,
                alternatives=["active_or_elliptical_reading_unresolved"],
                provenance=dict(source=resource["source"], version=resource["version"]))


def reduced_passive_hypothesis(predicate, subject, objects, preps):
    assessment = assess_passive_fragment(predicate, subject, objects, preps)
    return assessment if assessment["status"] == "provisional" else None

FIRST_CAUSES_SECOND = 0
SECOND_CAUSES_FIRST = 1
NO_CAUSAL_RELATION = 2
UNRESOLVED_RELATION = 3
ACTION_NAMES = {
    0: "FIRST causes SECOND", 1: "SECOND causes FIRST",
    2: "NO causal relation expressed", 3: "UNRESOLVED relationship",
}

TRAIN_EXAMPLES = [
    {
        "sentence": "Heat accelerates erosion.",
        "entity1": "heat",
        "entity2": "erosion",
        "correct_action": UNRESOLVED_RELATION,
    },
    {
        "sentence": "Wind influences erosion.",
        "entity1": "wind",
        "entity2": "erosion",
        "correct_action": UNRESOLVED_RELATION,
    },
    {
        "sentence": "Rain causes flooding.",
        "entity1": "rain",
        "entity2": "flooding",
        "correct_action": FIRST_CAUSES_SECOND,
    },
    {
        "sentence": "Smoke triggers alarm.",
        "entity1": "smoke",
        "entity2": "alarm",
        "correct_action": FIRST_CAUSES_SECOND,
    },
    {
        "sentence": "Heat produces expansion.",
        "entity1": "heat",
        "entity2": "expansion",
        "correct_action": FIRST_CAUSES_SECOND,
    },
    {
        "sentence": "By noon rain caused flooding.",
        "entity1": "rain",
        "entity2": "flooding",
        "correct_action": FIRST_CAUSES_SECOND,
    },
    {
        "sentence": "By chance smoke triggered alarm.",
        "entity1": "smoke",
        "entity2": "alarm",
        "correct_action": FIRST_CAUSES_SECOND,
    },
    {
        "sentence": "Rain caused flooding by blocking drains.",
        "entity1": "rain",
        "entity2": "flooding",
        "correct_action": FIRST_CAUSES_SECOND,
    },
    {
        "sentence": "Smoke triggered alarm by heating sensors.",
        "entity1": "smoke",
        "entity2": "alarm",
        "correct_action": FIRST_CAUSES_SECOND,
    },
    {
        "sentence": "Rain not snow causes flooding.",
        "entity1": "rain",
        "entity2": "flooding",
        "correct_action": FIRST_CAUSES_SECOND,
    },
    # SECOND -> FIRST
    {
        "sentence": "Flooding was caused by rain.",
        "entity1": "flooding",
        "entity2": "rain",
        "correct_action": SECOND_CAUSES_FIRST,
    },
    {
        "sentence": "Alarm was triggered by smoke.",
        "entity1": "alarm",
        "entity2": "smoke",
        "correct_action": SECOND_CAUSES_FIRST,
    },
    {
        "sentence": "Expansion was produced by heat.",
        "entity1": "expansion",
        "entity2": "heat",
        "correct_action": SECOND_CAUSES_FIRST,
    },
    {
        "sentence": "Flooding caused by rain today.",
        "entity1": "flooding",
        "entity2": "rain",
        "correct_action": SECOND_CAUSES_FIRST,
    },
    {
        "sentence": "Alarm triggered by smoke suddenly.",
        "entity1": "alarm",
        "entity2": "smoke",
        "correct_action": SECOND_CAUSES_FIRST,
    },
    # NO CAUSAL RELATION
    {
        "sentence": "Rain does not cause flooding.",
        "entity1": "rain",
        "entity2": "flooding",
        "correct_action": NO_CAUSAL_RELATION,
    },
    {
        "sentence": "Smoke never triggers alarm.",
        "entity1": "smoke",
        "entity2": "alarm",
        "correct_action": NO_CAUSAL_RELATION,
    },
    {
        "sentence": "Heat does not produce cold.",
        "entity1": "heat",
        "entity2": "cold",
        "correct_action": NO_CAUSAL_RELATION,
    },
    {
        "sentence": "Rain is unrelated to gravity.",
        "entity1": "rain",
        "entity2": "gravity",
        "correct_action": NO_CAUSAL_RELATION,
    },
    {
        "sentence": "Music is independent from rainfall.",
        "entity1": "music",
        "entity2": "rainfall",
        "correct_action": NO_CAUSAL_RELATION,
    },
]
# ============================================================
# ORDINARY HOLDOUT
# ============================================================
TEST_EXAMPLES = [
    {
        "sentence": "Stress causes insomnia.",
        "entity1": "stress",
        "entity2": "insomnia",
        "correct_action": FIRST_CAUSES_SECOND,
    },
    {
        "sentence": "By midnight stress caused insomnia.",
        "entity1": "stress",
        "entity2": "insomnia",
        "correct_action": FIRST_CAUSES_SECOND,
    },
    {
        "sentence": "Stress caused insomnia by disrupting sleep.",
        "entity1": "stress",
        "entity2": "insomnia",
        "correct_action": FIRST_CAUSES_SECOND,
    },
    {
        "sentence": "Insomnia was caused by stress.",
        "entity1": "insomnia",
        "entity2": "stress",
        "correct_action": SECOND_CAUSES_FIRST,
    },
    {
        "sentence": "Insomnia caused by stress yesterday.",
        "entity1": "insomnia",
        "entity2": "stress",
        "correct_action": SECOND_CAUSES_FIRST,
    },
    {
        "sentence": "Stress does not cause insomnia.",
        "entity1": "stress",
        "entity2": "insomnia",
        "correct_action": NO_CAUSAL_RELATION,
    },
    {
        "sentence": "Stress not caffeine causes insomnia.",
        "entity1": "stress",
        "entity2": "insomnia",
        "correct_action": FIRST_CAUSES_SECOND,
    },
    {
        "sentence": "Music is unrelated to insomnia.",
        "entity1": "music",
        "entity2": "insomnia",
        "correct_action": NO_CAUSAL_RELATION,
    },
]
# ============================================================
# NEGATION GENERALIZATION
# ============================================================
NEGATION_GENERALIZATION_EXAMPLES = [
    {
        "family": "active_positive",
        "sentence": "Rain caused flooding.",
        "entity1": "rain",
        "entity2": "flooding",
        "correct_action": FIRST_CAUSES_SECOND,
    },
    {
        "family": "active_negative",
        "sentence": "Rain did not cause flooding.",
        "entity1": "rain",
        "entity2": "flooding",
        "correct_action": NO_CAUSAL_RELATION,
    },
    {
        "family": "passive_positive",
        "sentence": "Flooding was caused by rain.",
        "entity1": "flooding",
        "entity2": "rain",
        "correct_action": SECOND_CAUSES_FIRST,
    },
    {
        "family": "passive_negative",
        "sentence": "Flooding was not caused by rain.",
        "entity1": "flooding",
        "entity2": "rain",
        "correct_action": NO_CAUSAL_RELATION,
    },
    {
        "family": "contrastive_not",
        "sentence": "Rain not snow caused flooding.",
        "entity1": "rain",
        "entity2": "flooding",
        "correct_action": FIRST_CAUSES_SECOND,
    },
    {
        "family": "object_contrast",
        "sentence": "Rain caused flooding not drought.",
        "entity1": "rain",
        "entity2": "flooding",
        "correct_action": FIRST_CAUSES_SECOND,
    },
]
# ============================================================
# ROBUSTNESS
# ============================================================
ROBUSTNESS_EXAMPLES = [
    {
        "family": "unresolved_relation",
        "sentence": "Rain accompanies flooding.",
        "entity1": "rain",
        "entity2": "flooding",
        "correct_action": UNRESOLVED_RELATION,
    },
    {
        "family": "negation_scope",
        "sentence": "Flooding was not caused by rain.",
        "entity1": "flooding",
        "entity2": "rain",
        "correct_action": NO_CAUSAL_RELATION,
    },
    {
        "family": "negation_scope",
        "sentence": "Rain not snow caused flooding.",
        "entity1": "rain",
        "entity2": "flooding",
        "correct_action": FIRST_CAUSES_SECOND,
    },
    {
        "family": "negation_scope",
        "sentence": "Rain caused flooding not drought.",
        "entity1": "rain",
        "entity2": "flooding",
        "correct_action": FIRST_CAUSES_SECOND,
    },
    {
        "family": "distance_invariance",
        "sentence": "Rain very often unexpectedly causes severe flooding.",
        "entity1": "rain",
        "entity2": "flooding",
        "correct_action": FIRST_CAUSES_SECOND,
    },
    {
        "family": "distance_invariance",
        "sentence": "Flooding was frequently and unexpectedly caused directly by rain.",
        "entity1": "flooding",
        "entity2": "rain",
        "correct_action": SECOND_CAUSES_FIRST,
    },
    {
        "family": "marker_role",
        "sentence": "Rain caused flooding by noon.",
        "entity1": "rain",
        "entity2": "flooding",
        "correct_action": FIRST_CAUSES_SECOND,
    },
    {
        "family": "marker_role",
        "sentence": "Flooding caused by rain continued.",
        "entity1": "flooding",
        "entity2": "rain",
        "correct_action": SECOND_CAUSES_FIRST,
    },
    {
        "family": "marker_role",
        "sentence": "Rain caused flooding by blocking drains.",
        "entity1": "rain",
        "entity2": "flooding",
        "correct_action": FIRST_CAUSES_SECOND,
    },
    {
        "family": "lexical_ood",
        "sentence": "Rain leads to flooding.",
        "entity1": "rain",
        "entity2": "flooding",
        "correct_action": FIRST_CAUSES_SECOND,
    },
    {
        "family": "lexical_ood",
        "sentence": "Flooding results from rain.",
        "entity1": "flooding",
        "entity2": "rain",
        "correct_action": SECOND_CAUSES_FIRST,
    },
]

# These are human annotations of the original examples, independent of parser
# output. Direction now describes the proposition even when it is denied.
DENIED_GOLD = {
    "Rain does not cause flooding.": FIRST_CAUSES_SECOND,
    "Smoke never triggers alarm.": FIRST_CAUSES_SECOND,
    "Heat does not produce cold.": FIRST_CAUSES_SECOND,
    "Stress does not cause insomnia.": FIRST_CAUSES_SECOND,
    "Rain did not cause flooding.": FIRST_CAUSES_SECOND,
    "Flooding was not caused by rain.": SECOND_CAUSES_FIRST,
}
for _suite in (TRAIN_EXAMPLES, TEST_EXAMPLES, NEGATION_GENERALIZATION_EXAMPLES,
               ROBUSTNESS_EXAMPLES):
    for _example in _suite:
        _example["legacy_action"] = _example["correct_action"]
        _example["assertion_status"] = "asserted"
        if _example["sentence"] in DENIED_GOLD:
            _example["correct_action"] = DENIED_GOLD[_example["sentence"]]
            _example["assertion_status"] = "denied"


def example(sentence, first, second, action, status="asserted"):
    return dict(sentence=sentence, entity1=first, entity2=second,
                correct_action=action, assertion_status=status)


# Teach configurations, not held-out predicate identities.
TRAIN_EXAMPLES += [
    example("Heat leads to expansion.", "heat", "expansion", 0),
    example("Expansion results from heat.", "expansion", "heat", 1),
    example("Rain correlates with flooding.", "rain", "flooding", 2),
    example("Rain may cause flooding.", "rain", "flooding", 0, "possible"),
    example("If rain causes flooding, roads close.", "rain", "flooding", 0,
            "conditional"),
    example("Observers say rain causes flooding.", "rain", "flooding", 0,
            "attributed"),
    example("Flooding was not caused by rain.", "flooding", "rain", 1, "denied"),
]

# All content words here are absent from CEM training, while induce, stem from,
# and associate with are explicitly known to the independent semantic lexicon.
LEXICAL_HOLDOUT = [
    example("Exposure induces disease.", "exposure", "disease", 0),
    example("Disease was induced by exposure.", "disease", "exposure", 1),
    example("Disease was not induced by exposure.", "disease", "exposure", 1, "denied"),
    example("Exposure does not induce disease.", "exposure", "disease", 0, "denied"),
    example("Disease stems from exposure.", "disease", "exposure", 1),
    example("Exposure associates with disease.", "exposure", "disease", 2),
    example("Exposure may induce disease.", "exposure", "disease", 0, "possible"),
]
UNKNOWN_HOLDOUT = [
    example("Dust accompanies corrosion.", "dust", "corrosion", 3),
    example("Light modulates growth.", "light", "growth", 3),
    example("Vibration foobulates sediment.", "vibration", "sediment", 3),
]
EPISTEMIC_HOLDOUT = [
    example("Exposure might not induce disease.", "exposure", "disease", 0,
            "possible"),
    example("If exposure induces disease, treatment changes.", "exposure", "disease",
            0, "conditional"),
    example("Scientists say exposure induces disease.", "exposure", "disease",
            0, "attributed"),
    example("Does exposure induce disease?", "exposure", "disease", 0, "questioned"),
]

# Multi-claim evaluation is separate from CEM fitting. Exact lists, rather than
# sets, detect missing, duplicated, or spurious relations.
MULTI_RELATION_HOLDOUT = [
    ("Exposure induces severe disease and vibration produces structural damage.",
     [("Exposure", "severe disease", "asserted", True),
      ("vibration", "structural damage", "asserted", True)]),
    ("Exposure induces disease and produces inflammation.",
     [("Exposure", "disease", "asserted", True),
      ("Exposure", "inflammation", "asserted", True)]),
    ("Exposure may induce disease, but vibration produces damage.",
     [("Exposure", "disease", "possible", False),
      ("vibration", "damage", "asserted", True)]),
    ("Disease was not induced by exposure, but vibration produces damage.",
     [("exposure", "Disease", "denied", False),
      ("vibration", "damage", "asserted", True)]),
    ("Exposure induces disease, which triggers inflammation.",
     [("Exposure", "disease", "asserted", True),
      ("disease", "inflammation", "asserted", True)]),
]


@dataclass(frozen=True)
class Lexeme:
    forms: tuple
    tail: tuple = ()
    kind: str = "causal"
    direction: str = "forward"


# Lexical knowledge belongs here, never in feature names or sentence patches.
# Surface forms supplement spaCy lemmas; neither POS nor lemma errors veto them.
LEXICON = (
    Lexeme(("cause", "causes", "caused", "causing")),
    Lexeme(("trigger", "triggers", "triggered", "triggering")),
    Lexeme(("produce", "produces", "produced", "producing")),
    Lexeme(("induce", "induces", "induced", "inducing")),
    Lexeme(("lead", "leads", "led", "leading"), ("to",)),
    Lexeme(("result", "results", "resulted", "resulting"), ("in",)),
    Lexeme(("result", "results", "resulted", "resulting"), ("from",),
           direction="reverse"),
    Lexeme(("stem", "stems", "stemmed", "stemming"), ("from",),
           direction="reverse"),
    Lexeme(("give", "gives", "gave", "given", "giving"), ("rise", "to")),
    Lexeme(("correlate", "correlates", "correlated", "correlating"), ("with",),
           kind="association", direction="none"),
    Lexeme(("associate", "associates", "associated", "associating"), ("with",),
           kind="association", direction="none"),
    Lexeme(("unrelated",), kind="explicit_no_relation", direction="none"),
    Lexeme(("independent",), kind="explicit_no_relation", direction="none"),
)

# Each channel describes reusable semantic or structural evidence, not a word.
FEATURE_NAMES = (
    "semantic_causal", "semantic_noncausal", "semantic_unknown",
    "semantic_reverse", "passive_by_topology", "dependency_passive",
    "causal_reverse_structure", "predicate_pos_verb",
    "dependency_subject_object", "relation_between_arguments", "bias",
)


@lru_cache(maxsize=1)
def get_nlp():
    import spacy
    try:
        return spacy.load("en_core_web_sm")
    except OSError as error:
        raise RuntimeError(
            "Install the local model: python -m spacy download en_core_web_sm"
        ) from error


def semantic_candidates(doc, lexicon=LEXICON):
    """Propose meanings independently of POS and dependency decisions."""
    candidates = []
    tokens = [t for t in doc if not t.is_punct and not t.is_space]
    for position, token in enumerate(tokens):
        for entry in lexicon:
            if token.lower_ not in entry.forms and token.lemma_.lower() not in entry.forms:
                continue
            following = tokens[position + 1:position + 1 + len(entry.tail)]
            if tuple(t.lower_ for t in following) != entry.tail:
                continue
            if any(t.sent.start != token.sent.start for t in following):
                continue
            candidates.append(dict(
                index=token.i,
                end=(following[-1].i + 1 if following else token.i + 1),
                semantic_class=entry.kind, lexical_direction=entry.direction,
                source="local_lexicon",
                matched_pattern=" ".join((entry.forms[0],) + entry.tail),
            ))
    return candidates


def _clause_bounds(doc, candidate, candidates):
    """Use relation anchors and coordinators to bound fallbacks, not POS alone."""
    predicate = doc[candidate["index"]]
    peers = sorted(c["index"] for c in candidates
                   if doc[c["index"]].sent.start == predicate.sent.start)
    previous = max((i for i in peers if i < predicate.i), default=None)
    following = min((i for i in peers if i > predicate.i), default=None)
    left, right = predicate.sent.start, predicate.sent.end
    if previous is not None:
        separators = [t.i for t in doc[previous + 1:predicate.i]
                      if t.lower_ in {"and", "but", "or", "yet"} or t.text == ";"]
        left = separators[-1] + 1 if separators else previous + 1
    if following is not None:
        separators = [t.i for t in doc[predicate.i + 1:following]
                      if t.lower_ in {"and", "but", "or", "yet"} or t.text == ";"]
        right = separators[-1] if separators else following
    candidate["clause_start"], candidate["clause_end"] = left, right
    candidate["previous_predicate"] = previous


@lru_cache(maxsize=1)
def load_attempt_adapter():
    path = Path(__file__).resolve().parent / "resources" / "verbnet_attempt_adapter.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported ATTEMPT adapter schema")
    return payload


def _attempt_frame_matcher(parent, child):
    return (child.dep_ == "xcomp" and child.head.i == parent.i
            and any(t.lower_ == "to" and t.dep_ in {"aux", "mark"} for t in child.children)
            and not any(t.dep_ in {"nsubj", "nsubjpass", "csubj"} for t in child.children)
            and not any(t.dep_ in {"obj", "dobj", "dative"} for t in parent.children)
            and any(t.dep_ == "nsubj" for t in parent.children))


def _parent_subject_controller(parent, child):
    return next((t for t in parent.children if t.dep_ == "nsubj"), None)


def _same_recovered_subject(parent, child):
    return (parent.subject is not None and child.subject is not None
            and parent.subject["token_index"] == child.subject["token_index"])


@dataclass(frozen=True)
class ComplementAdapter:
    namespace: str
    relation_type: str
    child_entailment: str
    active_lemmas: FrozenSet[str]
    documented_members: FrozenSet[str]
    frame_matcher: Callable
    controller: Callable
    recovered_frame_matcher: Callable
    provenance: dict
    match_reason: str

    def matches(self, parent, child):
        return (parent.lemma_.lower() in self.active_lemmas
                and self.frame_matcher(parent, child))


@lru_cache(maxsize=1)
def get_complement_adapters():
    """Registered policies; lexical evidence alone never activates a relation."""
    payload = load_attempt_adapter()
    return (ComplementAdapter(
        namespace=payload["namespace"], relation_type=payload["relation_type"],
        child_entailment=payload["adapter_child_entailment"],
        active_lemmas=frozenset(payload["active_lemmas"]),
        documented_members=frozenset(payload["documented_members"]),
        frame_matcher=_attempt_frame_matcher, controller=_parent_subject_controller,
        recovered_frame_matcher=_same_recovered_subject,
        provenance={k: payload[k] for k in ("resource", "version", "class_id", "source_url")},
        match_reason="supported_infinitival_subject_control",
    ),)


def matching_complement_adapters(parent, child):
    return [a for a in get_complement_adapters() if a.matches(parent, child)]


def matches_attempt_frame(parent, child):
    """Compatibility helper; extraction and interpretation use the registry."""
    return any(a.relation_type == "ATTEMPT"
               for a in matching_complement_adapters(parent, child))


def _argument_pair(doc, candidate, resolved=None):
    """Recover syntactic roles first; sort for the CEM text-order convention last."""
    resolved = {} if resolved is None else resolved
    predicate = doc[candidate["index"]]
    lo = candidate.get("clause_start", predicate.sent.start)
    hi = candidate.get("clause_end", predicate.sent.end)
    subjects = sorted((t for t in predicate.children
                       if t.dep_ in {"nsubj", "nsubjpass", "csubj", "csubjpass"}
                       and lo <= t.i < hi), key=lambda t: t.i)
    objects = sorted((t for t in predicate.children
                      if t.dep_ in {"obj", "dobj", "attr", "oprd"}
                      and lo <= t.i < hi), key=lambda t: t.i)
    # A nominal gerund is an event argument, including its own object.
    objects += [t for t in predicate.children if t.dep_ == "xcomp"
                and t.tag_ == "VBG" and lo <= t.i < hi]
    preps = [obj for prep in predicate.children if prep.dep_ in {"prep", "agent"}
             for obj in prep.children if obj.dep_ == "pobj" and lo <= obj.i < hi]
    if not subjects and predicate.dep_ == "acl" and lo <= predicate.head.i < hi:
        subjects = [predicate.head]
    if not subjects and predicate.head.pos_ == "AUX":
        subjects = [t for t in predicate.head.children
                    if t.dep_ == "nsubj" and lo <= t.i < hi]
    subject = subjects[0] if subjects else None
    # Binary relation evidence may use an explicitly licensed oblique or passive
    # agent. Arbitrary prepositional objects are not predicate objects.
    passive = any(t.dep_ in {"nsubjpass", "auxpass", "agent"} for t in predicate.children)
    licensed_preps = [t for t in preps if
                      (passive and t.head.lower_ == "by"
                       and (not REPAIR_ENABLED or not temporal_by_object(t))) or
                      (candidate["semantic_class"] != "unknown"
                       and t.head.lower_ in {"from", "with", "to", "of"})]
    obj = (objects or licensed_preps or [None])[0]
    if REPAIR_ENABLED:
        if any(t.lower_ == "by" for t in predicate.children):
            candidate["passive_fragment_assessment"] = assess_passive_fragment(
                predicate, subject, objects, preps)
        rejected = [t.i for t in preps if t.head.lower_ == "by" and temporal_by_object(t)]
        if rejected:
            candidate["temporal_by_objects"] = rejected
        if obj is None and not passive:
            hypothesis = reduced_passive_hypothesis(predicate, subject, objects, preps)
            if hypothesis:
                candidate["structural_hypothesis"] = hypothesis
                obj = doc[hypothesis["source_index"]]
                passive = True
    provenance = "dependency"
    adapters = matching_complement_adapters(predicate.head, predicate)
    if subject is None and len(adapters) == 1:
        subject = adapters[0].controller(predicate.head, predicate)
        if subject is not None:
            candidate["subject_control_from"] = predicate.head.i
            provenance = "verbnet_adapter_subject_control"
    # An intransitive first conjunct has no CEM pair, but can still supply
    # an explicit syntactic subject to its directly coordinated predicate.
    if (subject is None and predicate.dep_ == "conj"
            and any(t.dep_ == "cc" and t.lower_ in {"and", "but", "yet"}
                    for t in predicate.head.children)):
        subject = next((t for t in predicate.head.children if t.dep_ == "nsubj"), None)
        if subject is not None:
            candidate["shared_subject_from"] = predicate.head.i
            provenance = "coordinated_subject"
    # Resolve only an explicit, locally attached relative pronoun.
    if subject is not None and subject.lower_ in {"which", "that", "who"}:
        if predicate.dep_ == "relcl" and predicate.head.pos_ in {"NOUN", "PROPN"}:
            candidate["relative_pronoun_index"] = subject.i
            subject = predicate.head
            provenance = "relative_antecedent"
    if subject is None:
        events = [t for t in doc[lo:predicate.i] if t.tag_ == "VBG"
                  and any(c.dep_ in {"obj", "dobj"} for c in t.children)]
        local = [t for t in doc[lo:predicate.i]
                 if t.pos_ in {"NOUN", "PROPN", "PRON"}]
        if not local:
            local = [t for t in doc[lo:predicate.i] if t.is_alpha and not t.is_stop]
        if events or local:
            subject = events[0] if events else local[-1]
            provenance = "positional_fallback"
    # Shared subject only across an explicit coordinator, with no new subject.
    previous = resolved.get(candidate.get("previous_predicate"))
    if subject is None and previous is not None and lo > predicate.sent.start:
        coordinator = doc[lo - 1]
        if coordinator.lower_ in {"and", "but", "or", "yet"}:
            # A coordinator following a relative clause can attach to either the
            # relative predicate or the main predicate. Preserve that ambiguity.
            prior_token = doc[previous["index"]]
            if prior_token.dep_ == "relcl":
                outer = [resolved[t.i] for t in prior_token.ancestors
                         if t.i in resolved and t.pos_ == "VERB"]
                if outer:
                    candidate["subject_attachment_candidates"] = [
                        dict(predicate_index=p["index"], subject_index=p["subject_index"])
                        for p in [previous] + outer]
                    compatible = [p for p in outer if doc[p["index"]].tag_ == predicate.tag_
                                  and any(t.tag_ == "MD" for t in doc[p["index"]].children)]
                    if compatible and prior_token.tag_ != predicate.tag_:
                        previous = compatible[0]
            subject = doc[previous["subject_index"]]
            candidate["shared_subject_from"] = previous["index"]
            provenance = "coordinated_subject"
    if obj is None and predicate.pos_ not in {"VERB", "AUX"}:
        local = [t for t in doc[candidate["end"]:hi]
                 if t.pos_ in {"NOUN", "PROPN", "PRON"}]
        if not local:
            local = [t for t in doc[candidate["end"]:hi] if t.is_alpha and not t.is_stop]
        if local:
            obj = local[0]
            # A noun inside a participial modifier is not the object head:
            # e.g. "life" in "one life-saving antidote".
            if (obj.head.dep_ == "amod" and obj.head.head.pos_ in {"NOUN", "PROPN"}
                    and candidate["end"] <= obj.head.head.i < hi):
                obj = obj.head.head
            while (obj.dep_ == "compound" and candidate["end"] <= obj.head.i < hi
                   and obj.head.pos_ in {"NOUN", "PROPN"}):
                obj = obj.head
            provenance = "positional_fallback"
    # --------------------------------------------------------
    # Preserve partial argument structure even when this
    # predicate cannot form the two-argument pair required
    # by the causal-relation classifier.
    #
    # Example:
    #
    #     the library to close
    #
    # close has:
    #     subject = library
    #     object  = None
    #
    # That is still a perfectly useful proposition frame:
    #
    #     close(library)
    #
    # CEM remains stricter and still receives no pair.
    # --------------------------------------------------------

    if subject is not None:

        candidate[
            "subject_index"
        ] = subject.i


    if obj is not None:

        candidate[
            "object_index"
        ] = obj.i


    if (
        subject is None
        or
        obj is None
        or
        subject.i == obj.i
    ):

        candidate[
            "partial_argument_structure"
        ] = True

        return None   

    # "by" attached to this predicate can precede either argument.
    by_agent = passive and any(t.lower_ == "by" and t.head.i == predicate.i
                   and any(x.i == obj.i for x in t.children) for t in predicate.sent)
    # Lemma/POS errors can lose the agent dependency; bounded topology is evidence.
    by_between = passive and any(t.lower_ == "by" for t in
                     doc[min(predicate.i, obj.i) + 1:max(predicate.i, obj.i)])
    passive = passive or by_agent or by_between or any(
        t.dep_ in {"nsubjpass", "auxpass"} for t in predicate.children)
    candidate["passive_evidence"] = passive
    cause = obj if passive or candidate["lexical_direction"] == "reverse" else subject
    candidate["cause_index"] = cause.i
    return (*sorted((subject, obj), key=lambda t: t.i), provenance)


def _entity_span(doc, head, candidate):
    """Bounded nominal/event mention; never swallow a neighbouring relation."""
    lo = candidate.get("clause_start", head.sent.start)
    hi = candidate.get("clause_end", head.sent.end)
    # Shared subjects and local relative antecedents lie outside this clause.
    if not lo <= head.i < hi:
        lo, hi = head.sent.start, candidate["index"]
    allowed = {"det", "amod", "compound", "poss", "case", "nummod", "quantmod"}
    if head.pos_ == "VERB":
        allowed |= {"dobj", "obj", "prt", "advmod", "neg"}
    indices = {head.i}
    pending = [head]
    while pending:
        current = pending.pop()
        for child in current.children:
            if (lo <= child.i < hi and child.i != candidate["index"]
                    and child.dep_ in allowed and child.i not in indices):
                indices.add(child.i)
                pending.append(child)
    # Recover adjective/compound modifiers in a parser-damaged local noun phrase.
    start = min(indices)
    while start > lo and doc[start - 1].pos_ in {"ADJ", "DET"}:
        start -= 1
    end = max(indices) + 1
    grouped = False
    for conj in head.conjuncts:
        if (lo <= conj.i < hi and conj.i != candidate["index"]
                and conj.pos_ in {"NOUN", "PROPN", "PRON"}):
            grouped = True
            start, end = min(start, conj.i), max(end, conj.i + 1)
            for child in conj.children:
                if child.dep_ in allowed and lo <= child.i < hi:
                    start, end = min(start, child.i), max(end, child.i + 1)
    span = doc[start:end]
    return dict(text=span.text, head_text=head.text, token_index=head.i,
                token_start=start, token_end=end, start=span.start_char,
                end=span.end_char, kind="event" if head.pos_ == "VERB" else "entity",
                coordinated=grouped)


def find_entity_spans(doc, entity_text):
    """Return every exact token-sequence match, including repeated mentions.

    Token boundaries prevent substring false matches (rain vs rainfall). Callers
    choose a mention by offsets; this helper never silently picks the first.
    """
    words = [t.lower_ for t in get_nlp().make_doc(entity_text) if not t.is_space]
    if not words:
        return []
    tokens = [t for t in doc if not t.is_space]
    matches = []
    for i in range(len(tokens) - len(words) + 1):
        if [t.lower_ for t in tokens[i:i + len(words)]] == words:
            span = doc[tokens[i].i:tokens[i + len(words) - 1].i + 1]
            matches.append(dict(text=span.text, start=span.start_char,
                                end=span.end_char, token_start=span.start,
                                token_end=span.end, token_index=span.root.i))
    return matches


def _assertion(doc, candidate, first, second):
    """Conservative scope annotations; multiple statuses are retained."""
    predicate = doc[candidate["index"]]
    lo = candidate.get("clause_start", predicate.sent.start)
    hi = candidate.get("clause_end", predicate.sent.end)
    scope = [predicate] + [t for t in predicate.children
                           if t.dep_ in {"aux", "auxpass"} and lo <= t.i < hi]
    negations = [t for head in scope for t in head.children
                 if (t.dep_ == "neg" or t.lower_ == "never") and lo <= t.i < hi]
    # A generic local fallback when negation attachment is missing, avoiding
    # "rain not snow": intervening nominal content blocks this fallback.
    for t in doc[max(first.i + 1, lo):predicate.i]:
        if t.lower_ in {"not", "never", "n't"} and all(
            x.pos_ in {"AUX", "ADV", "PART"} for x in doc[t.i + 1:predicate.i]
        ):
            negations.append(t)
    modals = [t for t in scope if t.tag_ == "MD"]
    flags = []
    if negations:
        flags.append("denied")
    hedges = [t for t in predicate.children
              if t.lower_ in {"possibly", "probably", "perhaps", "maybe", "likely"}]
    if modals or hedges:
        flags.append("possible")
    # Deliberately conservative: sentence-wide conditions/questions suppress
    # commitment even when we cannot disambiguate their precise scope.
    if any(t.lower_ in {"if", "unless"} for t in predicate.sent):
        flags.append("conditional")
    ancestors = list(predicate.ancestors)
    embedded = predicate.dep_ in {"ccomp", "xcomp"} or any(
        t.dep_ in {"ccomp", "xcomp"} for t in ancestors)
    attribution_tokens = [t for t in predicate.sent if t.lower_ == "allegedly"]
    for t in predicate.sent:
        if t.lemma_ == "accord" and t.i + 1 < len(doc) and doc[t.i + 1].lower_ == "to":
            attribution_tokens.append(t)
    if attribution_tokens:
        flags.append("attributed")
    if embedded:
        reporting = {"say", "report", "claim", "believe", "allege", "suggest"}
        if any(t.lemma_ in reporting for t in ancestors):
            flags.append("attributed")
        elif (candidate.get("clause_start", predicate.sent.start) <= predicate.head.i
              < candidate.get("clause_end", predicate.sent.end)):
            flags.append("unresolved")
    if any(t.lower_ in {"no", "neither", "any"}
           for entity in (first, second) for t in entity.children):
        flags.append("unresolved")
    if any(t.text == "?" for t in predicate.sent):
        flags.append("questioned")
    if any(t.text in {'"', '“', '”'} for t in predicate.sent):
        flags.append("quoted")
    primary = next((s for s in ("unresolved", "questioned", "quoted", "attributed", "conditional",
                               "possible", "denied") if s in flags), "asserted")
    return dict(
        status=primary, statuses=flags or ["asserted"],
        polarity="negative" if negations else "positive",
        negation_tokens=sorted({t.i for t in negations}),
        modal_tokens=[t.i for t in modals],
        hedge_tokens=[t.i for t in hedges],
        attribution_tokens=[t.i for t in attribution_tokens],
        attribution_source=[
            t.text for head in ancestors for t in head.children if t.dep_ == "nsubj"
        ] if "attributed" in flags else [],
        method="conservative_scope_rules",
    )


def _evidence(doc, candidate, pair):
    first, second, pair_source = pair
    predicate = doc[candidate["index"]]
    left_idx, right_idx = sorted((predicate.i, doc[candidate["object_index"]].i))
    between = list(doc[left_idx + 1:right_idx])
    passive_by = any(t.lower_ == "by" for t in between)
    dep_passive = any(t.dep_ in {"auxpass", "nsubjpass"}
                      for t in predicate.children)
    causal = candidate["semantic_class"] == "causal"
    lexical_reverse = candidate["lexical_direction"] == "reverse"
    values = (
        causal,
        candidate["semantic_class"] in {"association", "explicit_no_relation"},
        candidate["semantic_class"] == "unknown",
        lexical_reverse, passive_by, dep_passive,
        causal and candidate["cause_index"] == second.i,
        predicate.pos_ == "VERB",
        first.head.i == predicate.i and second.head.i == predicate.i,
        first.i < predicate.i < second.i, True,
    )
    return dict(zip(FEATURE_NAMES, map(float, values)))


@dataclass
class ArgumentContext:
    status: str
    window: dict
    arguments: list
    regions: dict
    ordering: str = "surface_order_not_semantic_direction"
    observational_only: bool = True
    schema_version: int = 1


def build_argument_context(doc, arguments, predicate_index=None):
    """Lossless local surface evidence; never repair or reinterpret arguments.

    Argument offsets and token indices are document-global, end-exclusive.
    Five disjoint regions exist only for two valid nonoverlapping arguments.
    Even otherwise the window and original argument descriptors are retained.
    """
    start, end = 0, len(doc.text)
    if predicate_index is not None:
        sentence = doc[predicate_index].sent
        start = sentence.start_char if sentence.start else 0
        end = sentence.end_char if sentence.end < len(doc) else len(doc.text)

    def region(lo, hi):
        return dict(text=doc.text[lo:hi], start=lo, end=hi, tokens=[
            dict(text=t.text, lemma=t.lemma_, pos=t.pos_, dep=t.dep_,
                 token_index=t.i, head_index=t.head.i, start=t.idx,
                 end=t.idx + len(t.text), whitespace=t.whitespace_)
            for t in doc if lo <= t.idx and t.idx + len(t.text) <= hi
        ])

    arguments = [dict(a) if a is not None else None for a in arguments]
    window = region(start, end)
    result = ArgumentContext("missing_arguments", window, arguments, {})
    if len(arguments) != 2 or any(a is None for a in arguments):
        return asdict(result)
    boundaries = {t.idx for t in doc} | {t.idx + len(t.text) for t in doc}
    if any(not isinstance(a.get("start"), int) or not isinstance(a.get("end"), int)
           or not start <= a["start"] < a["end"] <= end
           or a["start"] not in boundaries or a["end"] not in boundaries
           for a in arguments):
        result.status = "invalid_or_out_of_window_arguments"
        return asdict(result)
    ordered = sorted(enumerate(arguments), key=lambda item: (item[1]["start"], item[1]["end"]))
    (first_slot, first), (second_slot, second) = ordered
    if first["end"] > second["start"]:
        result.status = "overlapping_arguments"
        return asdict(result)
    result.status = "complete"
    result.regions = dict(
        left=region(start, first["start"]),
        argument_1=dict(region(first["start"], first["end"]), input_slot=first_slot,
                        reference=first),
        between=region(first["end"], second["start"]),
        argument_2=dict(region(second["start"], second["end"]), input_slot=second_slot,
                        reference=second),
        right=region(second["end"], end),
    )
    return asdict(result)


# C1 infinitival complement, C2 directional PP, C3 degree-result infinitive,
# C4 unresolved. C4 is always generated so weak or contradictory features can
# abstain instead of forcing a reading.
TO_ATTACHMENT_CLASSES = (
    "infinitival_complement",
    "directional_pp",
    "degree_result_infinitive",
    "unresolved",
)
TO_ATTACHMENT_CLASS_IDS = {
    "infinitival_complement": "C1",
    "directional_pp": "C2",
    "degree_result_infinitive": "C3",
    "unresolved": "C4",
}
TO_ATTACHMENT_FEATURES = (
    "bias",
    "prior_infinitival_complement",
    "prior_directional_pp",
    "prior_degree_result_infinitive",
    "prior_unresolved",
    "head_licenses_event_argument",
    "head_spatial_compatible",
    "semantic_desire_attitude",
    "semantic_temporal_state",
    "semantic_evaluative_attitude",
    "semantic_tough",
    "semantic_motion",
    "semantic_causative_or_control",
    "semantic_unknown",
    "observed_infinitive_marker",
    "observed_preposition",
    "observed_complement_verb",
    "observed_complement_noun",
    "identity_verb_open",
    "identity_noun_open",
    "dual_pos_ambiguity",
    "candidate_infinitival_complement",
    "candidate_directional_pp",
    "candidate_degree_result_infinitive",
    "complement_has_determiner",
    "complement_has_verbal_dependent",
    "degree_marker",
    "spatial_destination_cue",
    "object_control_pattern",
    "clausal_complement_pattern",
    "cooccur_temporal_noun",
    "cooccur_event_verb",
    "cooccur_degree_verb_only",
    "cooccur_degree_and_directional",
    "cooccur_unknown_ambiguous",
)
TO_ATTACHMENT_MIN_PROBABILITY = 0.55
TO_ATTACHMENT_MIN_MARGIN = 0.12
_TO_ATTACHMENT_SEMANTIC_FEATURES = {
    "desire_attitude": "semantic_desire_attitude",
    "temporal_state": "semantic_temporal_state",
    "evaluative_attitude": "semantic_evaluative_attitude",
    "tough": "semantic_tough",
    "motion": "semantic_motion",
    "causative_or_control": "semantic_causative_or_control",
    "unknown": "semantic_unknown",
}
# Labeled constructions for the attachment classifier only. Not CEM examples.
TO_ATTACHMENT_TRAINING = (
    ("They were eager to work.", "infinitival_complement"),
    ("She was eager to leave.", "infinitival_complement"),
    ("They were glad to help.", "infinitival_complement"),
    ("They were ready to work.", "infinitival_complement"),
    ("They were late to work.", "directional_pp"),
    ("They were late to the office.", "directional_pp"),
    ("They were early to work.", "directional_pp"),
    ("The hikers walked to the shelter.", "directional_pp"),
    ("They went to work.", "directional_pp"),
    ("They were too late to leave.", "degree_result_infinitive"),
    ("They were late enough to leave.", "degree_result_infinitive"),
    ("The task was hard to finish.", "infinitival_complement"),
    ("It is hard to work.", "infinitival_complement"),
    ("They tried to work.", "infinitival_complement"),
    ("The flood caused the library to close.", "infinitival_complement"),
    ("The storm forced the school to close.", "infinitival_complement"),
    ("The manager allowed the workers to leave.", "infinitival_complement"),
    ("They were too late to work.", "unresolved"),
    ("They were late enough to work.", "unresolved"),
    ("They were foobish to work.", "unresolved"),
    ("Rain leads to flooding.", "unresolved"),
)


def _flag(value):
    return 1.0 if value else 0.0


@lru_cache(maxsize=1)
def load_to_attachment_resource():
    path = Path(__file__).resolve().parent / "resources" / "to_attachment_valency.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported to-attachment valency schema")
    if payload.get("classes") != list(TO_ATTACHMENT_CLASSES):
        raise ValueError("To-attachment classes do not match the classifier")
    def check(head, name):
        priors = head.get("priors", {})
        if set(priors) != set(TO_ATTACHMENT_CLASSES):
            raise ValueError(f"Missing valency priors for {name}")
        if abs(sum(priors.values()) - 1.0) > 1e-6:
            raise ValueError(f"Valency priors for {name} must sum to 1")
    check(payload["default_head"], "default_head")
    for lemma, head in payload["heads"].items():
        check(head, lemma)
    return payload


def open_complement_categories(token):
    """Categories still open after local form checks.

    The tagger's single POS tag is one hypothesis. A bare lemma that is both
    a noun and a verb keeps both until a dependent or inflection closes one.
    """
    resource = load_to_attachment_resource()
    forced = set()
    if any(child.dep_ in {"det", "poss", "nummod"} for child in token.children):
        forced.add("noun")
    if any(child.dep_ in {"obj", "dobj", "prt", "nsubj", "nsubjpass", "csubj", "csubjpass"}
           for child in token.children):
        forced.add("verb")
    if token.tag_ in {"VBD", "VBG", "VBN", "VBZ"}:
        forced.add("verb")
    if token.tag_ in {"NNS", "NNPS"}:
        forced.add("noun")
    if forced:
        return forced
    if token.lemma_.lower() in set(resource["dual_pos_lemmas"]):
        return {"noun", "verb"}
    categories = set()
    if token.pos_ in {"VERB", "AUX"} or token.tag_ in {"VB", "VBP"}:
        categories.add("verb")
    if token.pos_ in {"NOUN", "PROPN"} or token.tag_ in {"NN", "NNP"}:
        categories.add("noun")
    if token.lemma_.lower() in set(resource["destination_lemmas"]):
        categories.add("noun")
    if not categories:
        categories.add("verb" if token.pos_ == "VERB" else "noun")
    return categories


def _attachment_head(marker, complement):
    resource = load_to_attachment_resource()
    if complement.dep_ in {"xcomp", "ccomp", "pcomp", "advcl", "acl"}:
        head = complement.head
    else:
        head = marker.head
    markers = set(resource["degree_markers"])
    if head.lemma_.lower() in markers and head.head.i != head.i:
        head = head.head
    return head


def generate_to_attachment_sites(doc):
    """Stage 1: structurally valid readings, including dual POS.

    This stage does not choose eager versus late. It only drops readings the
    local form cannot support, and it always keeps unresolved.
    """
    resource = load_to_attachment_resource()
    destinations = set(resource["destination_lemmas"])
    degree_markers = set(resource["degree_markers"])
    sites = []
    for marker in doc:
        if marker.lower_ != "to" or marker.dep_ not in {"aux", "mark", "prep"}:
            continue
        if marker.dep_ == "prep":
            complement = next((child for child in marker.children if child.dep_ == "pobj"), None)
        else:
            complement = marker.head if marker.head.i != marker.i else None
        if complement is None or complement.i <= marker.i or complement.pos_ == "PUNCT":
            continue
        head = _attachment_head(marker, complement)
        identities = open_complement_categories(complement)
        verb_open = "verb" in identities
        noun_open = "noun" in identities
        finite = complement.tag_ in {"VBD", "VBG", "VBN", "VBZ"}
        degree = any(child.lemma_.lower() in degree_markers and child.dep_ == "advmod"
                     for child in head.children)
        generated = {"unresolved"}
        reasons = {"unresolved": "always_available"}
        if verb_open and not finite:
            generated.add("infinitival_complement")
            reasons["infinitival_complement"] = "verb_identity_open"
        else:
            reasons["infinitival_complement"] = "not_generated_verb_identity_closed"
        if noun_open:
            generated.add("directional_pp")
            reasons["directional_pp"] = "noun_identity_open"
        else:
            reasons["directional_pp"] = "not_generated_noun_identity_closed"
        if degree and "infinitival_complement" in generated:
            generated.add("degree_result_infinitive")
            reasons["degree_result_infinitive"] = "degree_marker_and_infinitive_candidate"
        else:
            reasons["degree_result_infinitive"] = (
                "not_generated_no_degree_marker" if not degree else "not_generated_infinitive_closed")
        valency = resource["heads"].get(head.lemma_.lower(), resource["default_head"])
        verbal_dependent = any(
            child.dep_ in {"obj", "dobj", "prt", "nsubj", "nsubjpass"} for child in complement.children)
        site = dict(
            marker_index=marker.i, complement_index=complement.i, head_index=head.i,
            head_lemma=head.lemma_.lower(), complement_lemma=complement.lemma_.lower(),
            token_identities=sorted(identities),
            structural_candidates=[name for name in TO_ATTACHMENT_CLASSES if name in generated],
            candidate_reasons=reasons,
            degree_marker=degree,
            observed_infinitive_marker=marker.pos_ == "PART" or marker.dep_ in {"aux", "mark"},
            observed_preposition=marker.pos_ == "ADP" or marker.dep_ == "prep",
            observed_complement_pos=complement.pos_,
            complement_dependency=complement.dep_,
            marker_dependency=marker.dep_,
            complement_has_determiner=any(child.dep_ == "det" for child in complement.children),
            complement_has_verbal_dependent=verbal_dependent,
            spatial_destination_cue=complement.lemma_.lower() in destinations,
            object_control_pattern=(
                head.pos_ == "VERB" and complement.dep_ == "xcomp"
                and any(child.dep_ in {"obj", "dobj"} for child in head.children)),
            clausal_complement_pattern=(
                complement.dep_ in {"ccomp", "xcomp"}
                and any(child.dep_ in {"nsubj", "nsubjpass"} for child in complement.children)),
            valency=valency,
        )
        site["features"] = _to_attachment_features(site)
        sites.append(site)
    return sites


def _to_attachment_features(site):
    valency = site["valency"]
    priors = valency["priors"]
    verb_open = "verb" in site["token_identities"]
    noun_open = "noun" in site["token_identities"]
    degree = bool(site["degree_marker"])
    candidates = set(site["structural_candidates"])
    semantic = valency.get("semantic_class", "unknown")
    features = dict.fromkeys(TO_ATTACHMENT_FEATURES, 0.0)
    features["bias"] = 1.0
    for name in TO_ATTACHMENT_CLASSES:
        features[f"prior_{name}"] = float(priors[name])
    features["head_licenses_event_argument"] = _flag(valency.get("licenses_event_argument"))
    features["head_spatial_compatible"] = _flag(valency.get("spatial_compatible"))
    features[_TO_ATTACHMENT_SEMANTIC_FEATURES.get(semantic, "semantic_unknown")] = 1.0
    if semantic not in _TO_ATTACHMENT_SEMANTIC_FEATURES:
        features["semantic_unknown"] = 1.0
    features["observed_infinitive_marker"] = _flag(site["observed_infinitive_marker"])
    features["observed_preposition"] = _flag(site["observed_preposition"])
    features["observed_complement_verb"] = _flag(site["observed_complement_pos"] == "VERB")
    features["observed_complement_noun"] = _flag(site["observed_complement_pos"] == "NOUN")
    features["identity_verb_open"] = _flag(verb_open)
    features["identity_noun_open"] = _flag(noun_open)
    features["dual_pos_ambiguity"] = _flag(verb_open and noun_open)
    features["candidate_infinitival_complement"] = _flag("infinitival_complement" in candidates)
    features["candidate_directional_pp"] = _flag("directional_pp" in candidates)
    features["candidate_degree_result_infinitive"] = _flag("degree_result_infinitive" in candidates)
    features["complement_has_determiner"] = _flag(site["complement_has_determiner"])
    features["complement_has_verbal_dependent"] = _flag(site["complement_has_verbal_dependent"])
    features["degree_marker"] = _flag(degree)
    features["spatial_destination_cue"] = _flag(site["spatial_destination_cue"])
    features["object_control_pattern"] = _flag(site["object_control_pattern"])
    features["clausal_complement_pattern"] = _flag(site["clausal_complement_pattern"])
    features["cooccur_temporal_noun"] = _flag(semantic == "temporal_state" and noun_open)
    features["cooccur_event_verb"] = _flag(valency.get("licenses_event_argument") and verb_open)
    features["cooccur_degree_verb_only"] = _flag(degree and verb_open and not noun_open)
    features["cooccur_degree_and_directional"] = _flag(degree and verb_open and noun_open)
    features["cooccur_unknown_ambiguous"] = _flag(semantic == "unknown" and verb_open and noun_open)
    return features


def _to_attachment_row(features):
    return np.array([features[name] for name in TO_ATTACHMENT_FEATURES], dtype=float)


def _softmax_rows(logits):
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / np.sum(exponent, axis=-1, keepdims=True)


def _fit_to_attachment_weights():
    """Multinomial logistic regression. Valency priors start as the baseline."""
    rows, labels, allowed = [], [], []
    for sentence, label in TO_ATTACHMENT_TRAINING:
        sites = generate_to_attachment_sites(get_nlp()(sentence))
        if len(sites) != 1:
            raise ValueError(f"Expected one to-attachment in {sentence!r}, found {len(sites)}")
        if label not in sites[0]["structural_candidates"]:
            raise ValueError(f"{label} is not a structural candidate for {sentence!r}")
        rows.append(_to_attachment_row(sites[0]["features"]))
        labels.append(TO_ATTACHMENT_CLASSES.index(label))
        allowed.append([name in sites[0]["structural_candidates"] for name in TO_ATTACHMENT_CLASSES])
    features = np.vstack(rows)
    gold = np.asarray(labels)
    mask = np.asarray(allowed, dtype=bool)
    classes, width = len(TO_ATTACHMENT_CLASSES), features.shape[1]
    weights = np.zeros((classes, width))
    for index, name in enumerate(TO_ATTACHMENT_CLASSES):
        weights[index, TO_ATTACHMENT_FEATURES.index(f"prior_{name}")] = 4.0
    target = np.eye(classes)[gold]
    penalty_scale = np.full_like(weights, 0.04)
    penalty_scale[:, 0] = 0.0
    for index, name in enumerate(TO_ATTACHMENT_CLASSES):
        penalty_scale[index, TO_ATTACHMENT_FEATURES.index(f"prior_{name}")] = 0.0
    learning_rate = 0.35
    for _ in range(1600):
        logits = np.where(mask, features @ weights.T, -40.0)
        probabilities = _softmax_rows(logits)
        gradient = (probabilities - target).T @ features / len(features)
        gradient += penalty_scale * weights
        weights -= learning_rate * gradient
    return weights


@lru_cache(maxsize=1)
def get_to_attachment_model():
    """Fit on first use from TO_ATTACHMENT_TRAINING. Import does not fit."""
    return _fit_to_attachment_weights()


def _to_attachment_public(site, probabilities, selected, abstained, reason, preference):
    return dict(
        schema_version=1,
        architecture="deterministic_candidates_then_logistic_classifier",
        marker_index=site["marker_index"],
        complement_index=site["complement_index"],
        head_index=site["head_index"],
        head_lemma=site["head_lemma"],
        complement_lemma=site["complement_lemma"],
        token_identities=list(site["token_identities"]),
        structural_candidates=list(site["structural_candidates"]),
        candidate_reasons=dict(site["candidate_reasons"]),
        observed_complement_pos=site["observed_complement_pos"],
        complement_dependency=site["complement_dependency"],
        marker_dependency=site["marker_dependency"],
        valency_priors={name: float(site["valency"]["priors"][name]) for name in TO_ATTACHMENT_CLASSES},
        head_semantic_class=site["valency"].get("semantic_class", "unknown"),
        features={name: float(site["features"][name]) for name in TO_ATTACHMENT_FEATURES},
        probabilities={name: float(probabilities[index]) for index, name in enumerate(TO_ATTACHMENT_CLASSES)},
        model_preference=preference,
        model_preference_id=TO_ATTACHMENT_CLASS_IDS[preference],
        selected=selected,
        selected_id=TO_ATTACHMENT_CLASS_IDS[selected],
        abstained=abstained,
        abstention_reason=reason,
        thresholds=dict(minimum_probability=TO_ATTACHMENT_MIN_PROBABILITY,
                        minimum_margin=TO_ATTACHMENT_MIN_MARGIN),
        score_interpretation=(
            "softmax probabilities from L2-regularized logistic regression on a "
            "small labeled construction set; the acceptance threshold is an explicit policy"
        ),
        eligible_for_world_state=False,
    )


def to_attachment_summary(decision):
    keys = ("marker_index", "complement_index", "head_index", "head_lemma",
            "complement_lemma", "token_identities", "structural_candidates",
            "selected", "selected_id", "abstained", "abstention_reason",
            "model_preference", "probabilities")
    return {key: decision[key] for key in keys}


def classify_to_attachments(doc):
    """Stage 2: score stage-1 candidates and abstain below the confidence policy."""
    weights = get_to_attachment_model()
    analyses = []
    for site in generate_to_attachment_sites(doc):
        logits = weights @ _to_attachment_row(site["features"])
        allowed = np.array([name in site["structural_candidates"] for name in TO_ATTACHMENT_CLASSES])
        logits = np.where(allowed, logits, -40.0)
        probabilities = _softmax_rows(logits.reshape(1, -1))[0]
        order = np.argsort(probabilities)
        preference = TO_ATTACHMENT_CLASSES[int(order[-1])]
        preference_probability = float(probabilities[order[-1]])
        runner_up = float(probabilities[order[-2]]) if len(order) > 1 else 0.0
        margin = preference_probability - runner_up
        if preference == "unresolved":
            selected, abstained, reason = "unresolved", True, "model_selected_unresolved"
        elif preference_probability < TO_ATTACHMENT_MIN_PROBABILITY:
            selected, abstained, reason = "unresolved", True, "below_probability_threshold"
        elif margin < TO_ATTACHMENT_MIN_MARGIN:
            selected, abstained, reason = "unresolved", True, "below_margin_threshold"
        else:
            selected, abstained, reason = preference, False, None
        analyses.append(_to_attachment_public(
            site, probabilities, selected, abstained, reason, preference))
    return analyses


def rejected_infinitive_indices(doc):
    """Verb tokens whose infinitive reading was not accepted."""
    return {
        decision["complement_index"]
        for decision in classify_to_attachments(doc)
        if decision["selected"] in {"directional_pp", "unresolved"}
        and decision["observed_complement_pos"] == "VERB"
        and decision["complement_dependency"] in {"xcomp", "ccomp", "pcomp", "advcl", "acl"}
    }


def collect_claim_evidence(sentence, lexicon=LEXICON, include_complement_parents=False):
    """One independently scoped evidence record per relation; parse text once."""
    doc = get_nlp()(sentence)
    rejected_infinitives = rejected_infinitive_indices(doc)
    proposals = semantic_candidates(doc, lexicon)
    known_indices = {c["index"] for c in proposals}
    proposals += [
        dict(index=t.i, end=t.i + 1, semantic_class="unknown",
             lexical_direction="unknown", source="dependency_predicate",
             matched_pattern=None)
        for t in doc if t.pos_ == "VERB" and t.i not in known_indices
        and t.i not in rejected_infinitives
        and t.dep_ != "amod"
        and not (t.tag_ == "VBG" and t.dep_ in {"csubj", "xcomp", "pcomp"})
        and (include_complement_parents or not any(
            c.dep_ == "ccomp" and c.tag_ != "VBG" for c in t.children))
        and not (t.tag_ == "VBG" and any(c.dep_ in {"obj", "dobj"} for c in t.children)
                 and any(t.i < i < t.sent.end for i in known_indices))
    ]
    # Only propose uncertain content-word predicates in otherwise unanalysed
    # coordinated regions. Do not turn known clauses' nouns into extra relations.
    regions = []
    for sent in doc.sents:
        boundaries = [sent.start - 1] + [t.i for t in sent
                      if t.lower_ in {"and", "but", "or", "yet"} or t.text == ";"] + [sent.end]
        regions.extend(doc[a + 1:b] for a, b in zip(boundaries, boundaries[1:]))
    for region in regions:
        if any(region.start <= c["index"] < region.end for c in proposals):
            continue
        nouns = [t for t in region if t.pos_ in {"NOUN", "PROPN", "PRON"}]
        if len(nouns) >= 2:
            proposals += [
                dict(index=t.i, end=t.i + 1, semantic_class="unknown",
                     lexical_direction="unknown", source="intervening_content",
                     matched_pattern=None)
                for t in doc[nouns[0].i + 1:nouns[-1].i]
                if t.is_alpha and not t.is_stop and t.pos_ not in {"ADV", "ADJ"}
            ]
    proposals.sort(key=lambda c: c["index"])
    # A lexical argument mis-tagged VERB should not split its own relation.
    anchors = [c for c in proposals if c["semantic_class"] != "unknown"
               or not any(doc[c["index"]].sent.start <= k["index"] < doc[c["index"]].sent.end
                          for k in proposals if k["semantic_class"] != "unknown")
               or c["source"] == "intervening_content"
               or any(t.dep_ in {"nsubj", "nsubjpass"} and t.i > max(
                   (i for i in known_indices if doc[c["index"]].sent.start <= i < c["index"]),
                   default=doc[c["index"]].sent.start - 1)
                      for t in doc[c["index"]].children)
               or doc[c["index"]].dep_ in {"conj", "relcl", "advcl"}]
    tokens = [dict(text=t.text, index=t.i, start=t.idx, end=t.idx + len(t.text),
                   lemma=t.lemma_, pos=t.pos_, dependency=t.dep_, head=t.head.i)
              for t in doc]
    records, resolved = [], {}
    for candidate in anchors:
        _clause_bounds(doc, candidate, anchors)
        # Protect the public causal API even if a child frame cannot be built.
        # Ignore parser attachments that cross an independent coordination.
        predicate = doc[candidate["index"]]
        candidate["complement_predicate_indices"] = [
            c.i for c in predicate.children if c.dep_ in {"xcomp", "ccomp"}
            and not any(t.lower_ in {"and", "but", "or", "yet"} or t.text == ";"
                        for t in doc[min(predicate.i, c.i) + 1:max(predicate.i, c.i)])]
        if (predicate.dep_ in {"xcomp", "ccomp"}
                and not any(t.lower_ in {"and", "but", "or", "yet"} or t.text == ";"
                            for t in doc[min(predicate.i, predicate.head.i) + 1:max(predicate.i, predicate.head.i)])):
            candidate["embedding_parent_index"] = predicate.head.i
        pair = _argument_pair(doc, candidate, resolved)
        if pair is None:
            records.append(dict(sentence=sentence, tokens=tokens, proposals=proposals,
                                selected_candidate=candidate, entities=[],
                                issues=["no_supported_argument_pair"], features=None))
            continue
        first, second, provenance = pair
        resolved[candidate["index"]] = candidate
        entities = [_entity_span(doc, t, candidate) for t in (first, second)]
        issues = []
        if candidate.get("subject_attachment_candidates"):
            issues.append("ambiguous_coordination_attachment")
        if any(t.lower_ == "or" for t in doc[candidate["index"]].sent):
            issues.append("disjunctive_scope_requires_interpretation")
        if any(e["coordinated"] for e in entities):
            issues.append("collective_argument_requires_interpretation")
        if any(doc[e["token_index"]].pos_ == "PRON" for e in entities):
            issues.append("unresolved_reference")
        assertion = _assertion(doc, candidate, first, second)
        previous = next((r for r in records if r.get("selected_candidate", {}).get("index")
                         == candidate.get("shared_subject_from")), None)
        own_aux = any(t.dep_ in {"aux", "auxpass"} for t in doc[candidate["index"]].children)
        if previous and not own_aux and doc[candidate["clause_start"] - 1].lower_ in {"and", "or"}:
            inherited = previous.get("assertion", {})
            flags = assertion["statuses"]
            if inherited.get("modal_tokens"):
                flags = [f for f in flags if f != "asserted"]
                if "possible" not in flags:
                    flags.append("possible")
                assertion["modal_tokens"] = inherited["modal_tokens"]
                assertion["status"] = "possible" if assertion["status"] == "asserted" else assertion["status"]
            if inherited.get("negation_tokens"):
                issues.append("ambiguous_shared_negation_scope")
            assertion["statuses"] = flags
            assertion["inherited_scope_from"] = candidate["shared_subject_from"]
        records.append(dict(
            sentence=sentence, tokens=tokens, proposals=proposals,
            selected_candidate=candidate, pair_source=provenance, entities=entities,
            features=_evidence(doc, candidate, pair), assertion=assertion, issues=issues,
        ))
    if not records:
        records.append(dict(sentence=sentence, tokens=tokens, proposals=proposals,
                            entities=[], issues=["no_supported_argument_pair"], features=None))
    for record in records:
        candidate = record.get("selected_candidate", {})
        if include_complement_parents and candidate.get("object_index") is not None:
            obj = doc[candidate["object_index"]]
            if obj.dep_ == "pobj":
                candidate["relation_argument_index"] = candidate.pop("object_index")
        record["argument_context"] = build_argument_context(
            doc, record["entities"], record.get("selected_candidate", {}).get("index"))
    return records
# ============================================================
# PROPOSITION FRAME EXTRACTION
#
# Converts existing claim-evidence records into a purely
# structural proposition representation.
#
# This layer intentionally does NOT use CEM.
# ============================================================

def extract_proposition_frames(
    sentence,
    lexicon=LEXICON,
):

    evidence_records = (
        collect_claim_evidence(
            sentence,
            lexicon,
            include_complement_parents=True,
        )
    )


    # --------------------------------------------------------
    # Proposition frames may contain incomplete argument
    # structures, so recover spans directly from the parse.
    #
    # collect_claim_evidence() remains strict for CEM.
    # --------------------------------------------------------

    doc = get_nlp()(
        sentence
    )


    frames = []


    for frame_number, evidence in enumerate(
        evidence_records
    ):

        candidate = evidence.get(
            "selected_candidate"
        )


        # No predicate was recovered at all.
        if candidate is None:

            continue


        predicate_index = (
            candidate.get(
                "index"
            )
        )


        if predicate_index is None:

            continue


        predicate_token = (
            doc[
                predicate_index
            ]
        )


        # ====================================================
        # RECOVER STRUCTURAL ARGUMENTS
        #
        # These may be partial.
        #
        # Examples:
        #
        # break(dog, window)
        #
        # close(library)
        #
        # leave(workers)
        # ====================================================

        subject_index = (
            candidate.get(
                "subject_index"
            )
        )


        object_index = (
            candidate.get(
                "object_index"
            )
        )


        subject_entity = None
        object_entity = None


        if subject_index is not None:

            subject_entity = (
                _entity_span(
                    doc,
                    doc[
                        subject_index
                    ],
                    candidate,
                )
            )


        if object_index is not None:

            object_entity = (
                _entity_span(
                    doc,
                    doc[
                        object_index
                    ],
                    candidate,
                )
            )


        # ----------------------------------------------------
        # If absolutely no arguments survived, there isn't
        # enough structure yet to call this a proposition.
        # ----------------------------------------------------

        # Preserve a discovered complement even when its controller is unknown.
        # Missing roles are evidence of uncertainty, not a reason to erase it.


        # ----------------------------------------------------
        # A complete evidence record already has assertion
        # analysis.
        #
        # Partial proposition frames don't necessarily have
        # enough information for the old assertion function,
        # so remain conservative.
        # ----------------------------------------------------

        assertion = (
            evidence.get(
                "assertion"
            )
        )


        if assertion is None:

            assertion = {

                "status":
                    "unresolved",

                "statuses": [
                    "unresolved"
                ],

                "polarity":
                    "unknown",

                "method":
                    "partial_proposition_structure",
            }
            if subject_index is not None:
                assertion = _assertion(doc, candidate, doc[subject_index], doc[subject_index])


        issues = list(
            evidence.get(
                "issues",
                []
            )
        )


        if candidate.get(
            "partial_argument_structure"
        ):

            issues.append(
                "partial_argument_structure"
            )


        frame = PropositionFrame(

            frame_id=
                f"p{frame_number}",


            predicate_text=
                predicate_token.text,

            predicate_lemma=
                predicate_token.lemma_,

            predicate_index=
                predicate_index,


            subject=
                subject_entity,

            object=
                object_entity,


            clause_start=
                candidate.get(
                    "clause_start",
                    predicate_token.sent.start,
                ),

            clause_end=
                candidate.get(
                    "clause_end",
                    predicate_token.sent.end,
                ),


            dependency_role=
                predicate_token.dep_,


            assertion=
                assertion,


            issues=
                issues,


            candidate_source=
                candidate.get(
                    "source"
                ),


            predicate_head_index=
                predicate_token.head.i,
        )


        frames.append(
            frame
        )


    # ========================================================
    # SECOND PASS:
    #
    # Now that partial frames such as close(library) exist,
    # connect embedded propositions to their parent frames.
    # ========================================================

    frames = (
        link_embedded_event_frames(
            frames
        )
    )


    for frame in frames:
        frame.attachments, frame.quantification = extract_attached_evidence(doc, frame)
    accepted_infinitives = {
        decision["complement_index"]: decision
        for decision in classify_to_attachments(doc)
        if decision["selected"] in {"infinitival_complement", "degree_result_infinitive"}
    }
    for frame in frames:
        decision = accepted_infinitives.get(frame.predicate_index)
        if decision is not None:
            frame.to_attachment = to_attachment_summary(decision)
    preserve_event_controllers(doc, frames, evidence_records)
    return frames


def preserve_event_controllers(doc, frames, evidence_records):
    """Preserve recovered child subjects on links, never fill a parent's object.

    A controller candidate is structural metadata, not a new control policy.
    Finite ccomp subjects are not classified as controllers. Recovered subjects
    without direct or adapter evidence remain explicitly unvalidated candidates.
    """
    by_id = {frame.frame_id: frame for frame in frames}
    candidates = {record.get("selected_candidate", {}).get("index"):
                  record.get("selected_candidate", {}) for record in evidence_records}
    for frame in frames:
        frame.event_links = []
    for child in frames:
        parent = by_id.get(child.parent_frame_id)
        if parent is None:
            continue
        candidate = None
        source, support, configuration = None, "missing_child_subject", "unresolved"
        if child.dependency_role != "xcomp":
            support = "not_a_control_construction"
        elif child.subject is not None:
            candidate = dict(child.subject)
            source = "child_subject"
            index = candidate["token_index"]
            child_token = doc[child.predicate_index]
            direct = any(t.i == index and t.dep_ in {"nsubj", "nsubjpass", "csubj", "csubjpass"}
                         for t in child_token.children)
            adapter_control = candidates.get(child.predicate_index, {}).get("subject_control_from") == parent.predicate_index
            support = ("explicit_child_subject" if direct else "adapter_subject_control"
                       if adapter_control else "recovered_child_subject_unvalidated")
            if parent.subject is not None and parent.subject["token_index"] == index:
                configuration = "subject_control" if adapter_control else "shared_subject_candidate"
            elif parent.object is not None and parent.object["token_index"] == index:
                configuration = "parent_object_matches_child_subject"
            else:
                configuration = "distinct_child_subject"
        parent.event_links.append(dict(
            parent_frame=parent.frame_id, child_frame=child.frame_id,
            dependency_role=child.dependency_role,
            controller_candidate=candidate, controller_source=source,
            controller_support=support, configuration=configuration,
            interpretation_status="candidate" if candidate is not None else "unresolved",
            eligible_for_world_state=False,
        ))


def extract_attached_evidence(doc, frame):
    """Keep syntactic attachments even when a semantic role is not licensed."""
    predicate = doc[frame.predicate_index]
    attachments, quantification = [], []
    motion = predicate.lemma_.lower() in {"go", "come", "return", "walk", "travel", "fly", "flee"}
    passive = any(c.dep_ in {"nsubjpass", "auxpass", "agent"} for c in predicate.children)
    for token in predicate.children:
        if token.dep_ not in {"prep", "agent", "advmod", "npadvmod"}:
            continue
        role, basis = "unresolved", "syntax_only"
        if token.dep_ in {"prep", "agent"}:
            if token.lower_ in {"after", "before"}:
                role, basis = "temporal", "local_temporal_preposition_policy"
            elif token.lower_ == "by" and passive and token.dep_ == "agent":
                role, basis = "agent", "passive_agent_dependency"
                if REPAIR_ENABLED and any(temporal_by_object(c) for c in token.children if c.dep_ == "pobj"):
                    role, basis = "temporal", "local_temporal_by_policy"
            elif token.lower_ == "to" and motion:
                role, basis = "destination", "local_motion_construction_policy"
        elif token.lower_ == "home" and motion:
            role, basis = "destination", "local_motion_home_policy"
        elif REPAIR_ENABLED and temporal_by_object(token):
            role, basis = "temporal", "local_temporal_modifier_policy"
        tokens = sorted(token.subtree, key=lambda t: t.i)
        contiguous = [t.i for t in tokens] == list(range(tokens[0].i, tokens[-1].i + 1))
        span = (dict(start=tokens[0].idx, end=tokens[-1].idx + len(tokens[-1].text),
                     kind="attached_phrase", token_index=token.i)
                if contiguous else None)
        attachments.append(dict(
            head_text=token.text, dependency=token.dep_, token_index=token.i,
            attached_to_token_index=predicate.i, role=role, role_basis=basis,
            span=span, token_indices=[t.i for t in tokens],
            argument_context=build_argument_context(doc, [frame.subject, span], predicate.i),
        ))
    for argument in (frame.subject, frame.object):
        if argument is None:
            continue
        head = doc[argument["token_index"]]
        for token in head.children:
            if token.lower_ in {"all", "every", "each", "any", "no", "neither"}:
                quantification.append(dict(text=token.text, token_index=token.i,
                                           argument_token_index=head.i,
                                           scope_status="unresolved", source="explicit_surface_quantifier"))
    return attachments, quantification

# ============================================================
# LINK EMBEDDED EVENT FRAMES
#
# Example:
#
# The flood caused the library to close.
#
# p0 = cause(flood, ...)
# p1 = close(library)
#
# spaCy commonly gives:
#
# close --xcomp--> caused
#
# We turn that structural fact into:
#
# p0.event_object_frame_id = p1
# p1.parent_frame_id = p0
#
# IMPORTANT:
# This does NOT say the relationship is causal.
# It only says one proposition is grammatically embedded
# beneath another proposition.
# ============================================================

def link_embedded_event_frames(
    frames
):

    frame_by_predicate_index = {

        frame.predicate_index:
            frame

        for frame
        in frames
    }


    for child_frame in frames:

        # ----------------------------------------------------
        # Start conservatively.
        #
        # xcomp:
        #     The flood caused the library TO CLOSE.
        #
        # ccomp:
        #     Officials said THAT flooding damaged roads.
        #
        # We will consider advcl/participial clauses later.
        # ----------------------------------------------------

        if child_frame.dependency_role not in {
            "xcomp",
            "ccomp",
        }:

            continue


        parent_predicate_index = (
            child_frame.predicate_head_index
        )


        if parent_predicate_index is None:

            continue


        parent_frame = (
            frame_by_predicate_index.get(
                parent_predicate_index
            )
        )


        if parent_frame is None:

            continue


        # ----------------------------------------------------
        # Record the relationship in BOTH directions.
        # ----------------------------------------------------

        child_frame.parent_frame_id = (
            parent_frame.frame_id
        )


        if (
            child_frame.frame_id
            not in
            parent_frame.complement_frame_ids
        ):

            parent_frame.complement_frame_ids.append(
                child_frame.frame_id
            )


        # ----------------------------------------------------
        # For this first experiment, treat the embedded
        # proposition as the parent's EVENT OBJECT.
        #
        # We preserve parent_frame.object as well because
        # "library" may be the controller of "close".
        # ----------------------------------------------------

        parent_frame.event_object_frame_id = (
            parent_frame.complement_frame_ids[0]
            if len(parent_frame.complement_frame_ids) == 1 else None
        )


    return frames


def collect_evidence(sentence, lexicon=LEXICON):
    """Legacy primary-relation view for single-label training/evaluation only."""
    records = collect_claim_evidence(sentence, lexicon)
    return next((r for r in records if r["features"] is not None and
                 r["selected_candidate"]["semantic_class"] != "unknown"),
                next((r for r in records if r["features"] is not None), records[0]))


def extract_channels(example_or_sentence):
    sentence = (example_or_sentence["sentence"]
                if isinstance(example_or_sentence, dict) else example_or_sentence)
    evidence = collect_evidence(sentence)
    if evidence["features"] is None:
        # Absence of a parsed candidate is unknown, not evidence of no causation.
        features = dict.fromkeys(FEATURE_NAMES, 0.0)
        features["semantic_unknown"] = features["bias"] = 1.0
    else:
        features = evidence["features"]
    return np.array([features[name] for name in FEATURE_NAMES], dtype=np.float64)

@dataclass
class PropositionFrame:

    frame_id: str

    predicate_text: str
    predicate_lemma: str
    predicate_index: int

    subject: Optional[dict]
    object: Optional[dict]

    clause_start: int
    clause_end: int

    dependency_role: str

    assertion: dict

    issues: list

    candidate_source: Optional[str]
    # ========================================================
    # NEW: FRAME-TO-FRAME STRUCTURE
    # ========================================================

    predicate_head_index: Optional[int] = None

    parent_frame_id: Optional[str] = None

    complement_frame_ids: list = field(
        default_factory=list
    )

    event_object_frame_id: Optional[str] = None    

    attachments: list = field(default_factory=list)
    quantification: list = field(default_factory=list)
    event_links: list = field(default_factory=list)
    to_attachment: Optional[dict] = None



@dataclass
class CEMPolicy:
    weights: np.ndarray
    feature_names: tuple = FEATURE_NAMES

    def __post_init__(self):
        self.weights = np.asarray(self.weights, dtype=float)
        if (tuple(self.feature_names) != FEATURE_NAMES or
            self.weights.shape != (4, len(FEATURE_NAMES)) or
            not np.isfinite(self.weights).all()):
            raise ValueError("Policy must match the current finite 4-action feature schema.")

    def save(self, path):
        Path(path).write_text(json.dumps(dict(
            schema_version=2, feature_names=list(self.feature_names),
            weights=self.weights.tolist(),
        ), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema_version") != 2:
            raise ValueError("Unsupported policy schema; retrain for role-aware direction (v2).")
        return cls(np.array(payload["weights"]), tuple(payload["feature_names"]))


def train_policy(examples=None, seed=42, population_size=256, generations=80,
                 perfect_patience=5):
    """Vectorized CEM; only training labels choose weights. No holdout stopping."""
    if population_size < 5 or generations < 1:
        raise ValueError("Use population_size >= 5 and generations >= 1.")
    if type(perfect_patience) is not int or perfect_patience < 0:
        raise ValueError("perfect_patience must be a nonnegative integer; 0 disables early stopping.")
    examples = TRAIN_EXAMPLES if examples is None else examples
    if not examples:
        raise ValueError("Training examples cannot be empty.")
    observations = np.stack([extract_channels(e) for e in examples])
    labels = np.array([e["correct_action"] for e in examples])
    rng = np.random.default_rng(seed)
    mean = np.zeros((4, len(FEATURE_NAMES)))
    std = np.ones_like(mean)
    best_weights, best_accuracy, best_margin = None, -1.0, -np.inf
    history = []
    perfect_generations = 0
    stop_reason = "generation_limit"
    for generation in range(generations):
        population = rng.normal(mean, std, (population_size, *mean.shape))
        predictions = np.einsum("paf,nf->pna", population, observations).argmax(axis=2)
        scores = (predictions == labels).mean(axis=1)
        perfect_generations = perfect_generations + 1 if np.all(scores == 1.0) else 0
        # Training margins break accuracy ties; these are not probabilities.
        logits = np.einsum("paf,nf->pna", population, observations)
        true_scores = logits[:, np.arange(len(labels)), labels]
        other = logits.copy()
        other[:, np.arange(len(labels)), labels] = -np.inf
        margins = (true_scores - other.max(axis=2)).min(axis=1)
        ordering = np.lexsort((margins, scores))
        elite = population[ordering[-max(2, population_size // 5):]]
        mean = elite.mean(axis=0)
        std = np.maximum(elite.std(axis=0), 0.08)
        best_index = ordering[-1]
        if (scores[best_index], margins[best_index]) > (best_accuracy, best_margin):
            best_accuracy = float(scores[best_index])
            best_margin = float(margins[best_index])
            best_weights = population[best_index].copy()
        history.append(dict(event_type="cem_generation", generation=generation + 1,
                            population_mean_accuracy=float(scores.mean()),
                            generation_best_accuracy=float(scores.max()),
                            global_best_accuracy=best_accuracy,
                            consecutive_perfect_populations=perfect_generations))
        if perfect_patience and perfect_generations >= perfect_patience:
            stop_reason = "training_population_saturation"
            break
    history.append(dict(event_type="cem_termination", stop_reason=stop_reason,
                        actual_generations=generation + 1, max_generations=generations,
                        perfect_patience=perfect_patience,
                        consecutive_perfect_populations=perfect_generations,
                        stopping_data="training_only", best_training_accuracy=best_accuracy))
    return CEMPolicy(best_weights), history


def _claim_from_evidence(evidence, policy):
    """Return a JSON-serializable claim; never write to Parliament's graph.

    FIRST/SECOND refer to text order, not agent/patient roles. The commitment
    flag is a conservative candidate gate, not truth verification.
    """
    sentence = evidence["sentence"]
    if evidence["features"] is None:
        candidate = evidence.get("selected_candidate", {})
        event_target = bool(candidate.get("complement_predicate_indices"))
        reasons = list(evidence["issues"])
        if event_target:
            reasons.append("event_complement_requires_proposition_reasoning")
        return dict(sentence=sentence, entities=[], relation_type="unresolved", direction=None,
                    source=None, target=None, assertion_status="unresolved",
                    source_entity=None, target_entity=None,
                    assertion={"status": "unresolved", "statuses": ["unresolved"],
                               "polarity": "unknown"}, decision=None,
                    eligible_for_world_state=False,
                    target_kind="proposition" if event_target else "entity",
                    validation_reasons=reasons, evidence=evidence)
    features = np.array([evidence["features"][n] for n in FEATURE_NAMES])
    scores = policy.weights @ features
    action = int(scores.argmax())
    candidate = evidence["selected_candidate"]
    semantic = candidate["semantic_class"]
    predicted_type = ("causal" if action in (0, 1) else
                      "noncausal" if action == 2 else "unresolved")
    expected_type = ("causal" if semantic == "causal" else
                     "noncausal" if semantic in {"association", "explicit_no_relation"}
                     else "unresolved")
    reasons = list(evidence["issues"])
    if candidate.get("structural_hypothesis"):
        reasons.append("provisional_structure_requires_validation")
    if candidate.get("embedding_parent_index") is not None:
        reasons.append("embedded_proposition_requires_scope_validation")
    has_event_complement = bool(candidate.get("complement_predicate_indices"))
    if has_event_complement:
        reasons.append("event_complement_requires_proposition_reasoning")
    if predicted_type != expected_type:
        reasons.append("policy_semantic_disagreement")
    if np.isclose(np.sort(scores)[-1], np.sort(scores)[-2]):
        reasons.append("tied_policy_scores")
    reverse = bool(evidence["features"]["causal_reverse_structure"])
    if predicted_type == "causal" and action != int(reverse):
        reasons.append("policy_direction_disagreement")
    assertion = evidence["assertion"]
    if assertion["statuses"] != ["asserted"]:
        reasons.append("not_an_unqualified_assertion")
    if semantic != "causal":
        reasons.append("no_resolved_causal_meaning")
    # Expose proposed direction even for a denial; never convert it into a
    # positive edge. A policy/semantic disagreement leaves meaning unresolved.
    resolved = predicted_type == expected_type and not any(
        r in reasons for r in ("tied_policy_scores", "policy_direction_disagreement")
    )
    relation_type = semantic if resolved and semantic != "unknown" else "unresolved"
    direction = ("first_to_second" if action == 0 else "second_to_first"
                 if action == 1 else None) if resolved else None
    entities = evidence["entities"]
    if any(t["text"].lower() in {"all", "every", "each"}
           and t["head"] in {e["token_index"] for e in entities}
           for t in evidence["tokens"]):
        reasons.append("quantified_argument_requires_scope_validation")
    source = entities[action]["text"] if direction is not None else None
    target = entities[1 - action]["text"] if direction is not None else None
    source_entity = entities[action] if direction is not None else None
    target_entity = entities[1 - action] if direction is not None else None
    if has_event_complement:
        # The noun may control the embedded predicate; it is not the effect.
        target, target_entity, direction = None, None, None
        if action != FIRST_CAUSES_SECOND:
            source, source_entity = None, None
    return dict(
        sentence=sentence, entities=entities, relation_type=relation_type,
        direction=direction, source=source, target=target,
        source_entity=source_entity, target_entity=target_entity,
        target_kind="proposition" if has_event_complement else "entity",
        assertion_status=assertion["status"], assertion=assertion,
        eligible_for_world_state=not reasons, validation_reasons=reasons,
        decision=dict(action=action, action_name=ACTION_NAMES[action],
                      scores=scores.tolist(),
                      margin=float(np.sort(scores)[-1] - np.sort(scores)[-2]),
                      score_interpretation="uncalibrated linear scores",
                      contributions={
                          ACTION_NAMES[a]: dict(zip(FEATURE_NAMES,
                                                     (policy.weights[a] * features).tolist()))
                          for a in range(4)}),
        evidence=evidence,
    )


def parse_claims(sentence, policy):
    """Return discovered local relation claims in text order, including abstentions."""
    claims = [dict(_claim_from_evidence(evidence, policy), claim_id=f"claim_{i}")
              for i, evidence in enumerate(collect_claim_evidence(sentence))]
    frames = extract_proposition_frames(sentence)
    by_index = {f.predicate_index: f for f in frames}
    for claim in claims:
        candidate = claim["evidence"].get("selected_candidate", {})
        frame = by_index.get(candidate.get("index"))
        claim["proposition_frame_id"] = frame.frame_id if frame else None
        claim["target_proposition_ids"] = [
            by_index[i].frame_id for i in candidate.get("complement_predicate_indices", [])
            if i in by_index]
    return claims


def parse_sentence(sentence, policy):
    """Compatible single-claim API; multiple claims require explicit consumption.

    For multiple relations the primary diagnostic claim includes a `claims` list
    and cannot itself be committed. Use parse_claims for per-relation eligibility.
    """
    claims = parse_claims(sentence, policy)
    primary = next((c for c in claims if c["relation_type"] != "unresolved"), claims[0])
    if len(claims) == 1:
        return primary
    return dict(primary, claims=claims, eligible_for_world_state=False,
                validation_reasons=primary["validation_reasons"] + ["multiple_claims_use_parse_claims"])


def evaluate_suite(examples, policy):
    records = []
    for item in examples:
        claim = parse_sentence(item["sentence"], policy)
        entities = claim.get("entities", [])
        pair_ok = ([e["head_text"].lower() for e in entities] ==
                   [item["entity1"], item["entity2"]])
        action_ok = (claim.get("decision") or {}).get("action") == item["correct_action"]
        status_ok = claim["assertion_status"] == item["assertion_status"]
        validated = not any(r.startswith("policy_") or r == "tied_policy_scores"
                            for r in claim["validation_reasons"])
        records.append(dict(
            sentence=item["sentence"], direction_correct=action_ok,
            pair_correct=pair_ok, assertion_correct=status_ok,
            end_to_end_correct=pair_ok and action_ok and status_ok and validated,
            expected_action=item["correct_action"],
            expected_assertion=item["assertion_status"], claim=claim,
        ))
    metrics = {
        name: sum(r[name] for r in records) / len(records)
        for name in ("direction_correct", "pair_correct", "assertion_correct",
                     "end_to_end_correct")
    }
    return metrics, records


def find_representation_collisions(examples):
    """Audit identical vectors with conflicting direction labels; never train on this."""
    groups = {}
    for item in examples:
        vector = tuple(extract_channels(item))
        groups.setdefault(vector, []).append(item)
    return [dict(features=dict(zip(FEATURE_NAMES, vector)),
                 sentences=[e["sentence"] for e in group],
                 labels=[e["correct_action"] for e in group])
            for vector, group in groups.items()
            if len({e["correct_action"] for e in group}) > 1]


def evaluate_multi_suite(examples, policy):
    records = []
    for sentence, expected in examples:
        claims = parse_claims(sentence, policy)
        observed = [(c["source"], c["target"], c["assertion_status"],
                     c["eligible_for_world_state"]) for c in claims]
        records.append(dict(sentence=sentence, expected=expected, observed=observed,
                            exact_match=observed == expected, claims=claims))
    return {"exact_claim_list_accuracy": sum(r["exact_match"] for r in records) / len(records)}, records


# The frame lexicon supplies local meanings, independently of CEM's causal
# lexicon. Unknown predicates retain syntactic roles without invented semantics.
FRAME_LEXICON = {
    "be": ("state", "property", "theme", "attribute"),
    "have": ("state", "possession", "holder", "theme"),
    "own": ("state", "possession", "holder", "theme"),
    "remain": ("state", "remaining", "theme", None),
    "send": ("event", "transfer", "actor", "theme"),
    "carry": ("event", "transfer", "actor", "theme"),
    "deliver": ("event", "transfer", "actor", "theme"),
    "redirect": ("event", "redirection", "actor", "theme"),
    "reroute": ("event", "redirection", "actor", "theme"),
    "request": ("event", "request", "requester", "theme"),
    "order": ("event", "request", "requester", "theme"),
    "face": ("state", "encounter", "experiencer", "theme"),
}
CHANGE_DIRECTIONS = {
    "reduce": "decrease", "decrease": "decrease", "lower": "decrease",
    "increase": "increase", "raise": "increase",
}
FRAME_LEXICON.update({lemma: ("event", "quantity_change", "influence", "theme")
                      for lemma in CHANGE_DIRECTIONS})

COUNT_WORDS = dict(zip(
    ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"),
    range(11),
))


def extract_event_state_structure(text):
    """Produce inspectable draft frames, never asserted world facts.

    References are document-local mention IDs. No cross-probe memory, pronoun
    guessing, causal consequences, or normative ranking is performed.
    """
    doc = get_nlp()(text)
    to_by_head, to_by_complement = {}, {}
    for decision in classify_to_attachments(doc):
        to_by_head.setdefault(decision["head_index"], []).append(decision)
        to_by_complement.setdefault(decision["complement_index"], []).append(decision)
    evidence = collect_claim_evidence(text, include_complement_parents=True)
    indices = {r["selected_candidate"]["index"] for r in evidence if r.get("selected_candidate")}
    # Copular states do not need a causal predicate or two nominal arguments.
    for token in doc:
        if token.lemma_ != "be" or token.i in indices or token.dep_ not in {"ROOT", "conj"}:
            continue
        subject = next((c for c in token.children if c.dep_ == "nsubj"), None)
        if subject is None:
            continue
        attribute = next((c for c in token.children if c.dep_ in {"attr", "acomp"}), None)
        candidate = dict(index=token.i, end=token.i + 1, subject_index=subject.i,
                         clause_start=token.sent.start, clause_end=token.sent.end)
        if attribute is not None:
            candidate["object_index"] = attribute.i
        evidence.append(dict(selected_candidate=candidate, issues=[], pair_source="copular_dependency"))
    evidence.sort(key=lambda r: r.get("selected_candidate", {}).get("index", len(doc)))
    mentions, events, alternatives = {}, [], []
    event_by_predicate = {}

    def mention(head, candidate):
        # Expand phrases locally but retain the full right-hand complement when
        # the causal pair detector stopped at an embedded predicate.
        span_candidate = dict(candidate, clause_end=head.sent.end)
        item = _entity_span(doc, head, span_candidate)
        key = f"mention_{item['start']}_{item['end']}"
        if key not in mentions:
            quantities = []
            for token in doc[item["token_start"]:item["token_end"]]:
                if token.dep_ == "nummod" or token.lower_ == "single":
                    value = COUNT_WORDS.get(token.lower_)
                    if token.lower_ == "single":
                        value = 1
                    elif token.like_num and token.text.replace(",", "").isdigit():
                        value = int(token.text.replace(",", ""))
                    quantities.append(dict(text=token.text, value=value,
                                           token_index=token.i, method="explicit_quantity"))
            mentions[key] = dict(
                item, mention_id=key, quantities=quantities,
                reference_status="unresolved" if head.pos_ == "PRON" or any(
                    c.dep_ == "det" and c.lower_ in {"this", "that", "these", "those"}
                    for c in head.children) else "surface_mention",
            )
        return key

    for record in evidence:
        candidate = record.get("selected_candidate")
        if candidate is None:
            continue
        pred = doc[candidate["index"]]
        lemma = pred.lemma_.lower()
        kind, meaning, subject_role, object_role = FRAME_LEXICON.get(
            lemma, ("event_or_state", "unresolved", "subject", "object"))
        subject_index = candidate.get("subject_index")
        object_index = candidate.get("object_index")
        # Unlike a binary causal claim, a unary state may have only a subject.
        if subject_index is None:
            subject_index = next((c.i for c in pred.children
                                  if c.dep_ in {"nsubj", "nsubjpass", "csubj"}), None)
        roles = {}
        if subject_index is not None:
            roles[subject_role] = [mention(doc[subject_index], candidate)]
        if object_index is not None and object_role:
            roles[object_role] = [mention(doc[object_index], candidate)]
        prepositions = []
        passive_subject = any(c.dep_ == "nsubjpass" for c in pred.children)
        attachment_heads = {pred.i: pred}
        for index in (subject_index, object_index):
            if index is not None:
                attachment_heads[index] = doc[index]
        for prep in [child for head in attachment_heads.values() for child in head.children]:
            if prep.dep_ not in {"prep", "agent", "dative"}:
                continue
            for obj in prep.children:
                if obj.dep_ not in {"pobj", "obj"}:
                    continue
                ref = mention(obj, candidate)
                prepositions.append(dict(preposition=prep.text, mention_id=ref,
                                         token_index=prep.i, attached_to_token_index=prep.head.i))
                role = {"to": "destination", "from": "origin", "at": "location",
                        "in": "location", "with": "accompaniment",
                        "across": "scope", "among": "scope"}.get(prep.lower_)
                if prep.head.i != pred.i and role != "scope":
                    role = None
                if prep.lower_ == "by" and passive_subject:
                    role = "actor" if meaning in {"transfer", "redirection"} else "agent"
                if role:
                    roles.setdefault(role, []).append(ref)
        if passive_subject and subject_index is not None and meaning in {"transfer", "redirection"}:
            roles["theme"] = [mention(doc[subject_index], candidate)]
            # Passive subject is not the actor; only an explicit by-agent is.
            roles["actor"] = [p["mention_id"] for p in prepositions
                              if p["preposition"].lower() == "by"]
            if not roles["actor"]:
                del roles["actor"]
        if meaning == "quantity_change" and subject_index is not None:
            if passive_subject:
                roles["theme"] = [mention(doc[subject_index], candidate)]
                roles["influence"] = [p["mention_id"] for p in prepositions
                                      if p["preposition"].lower() == "by"]
                roles.pop("agent", None)
                if not roles["influence"]:
                    del roles["influence"]
            elif object_index is None:
                # "Demand decreased" asserts a change without naming a cause.
                roles["theme"] = roles.pop("influence", [])
        assertion = record.get("assertion")
        if assertion is None and subject_index is not None:
            assertion = _assertion(doc, candidate, doc[subject_index], doc[subject_index])
        if assertion is None:
            assertion = dict(status="unresolved", statuses=["unresolved"], polarity="unknown")
        issues = [i for i in record["issues"] if i not in
                  {"no_supported_argument_pair", "disjunctive_scope_requires_interpretation"}]
        if not roles:
            issues.append("no_roles_recovered")
        if meaning == "unresolved":
            issues.append("frame_semantics_unresolved")
        if any(mentions[r]["reference_status"] == "unresolved"
               for refs in roles.values() for r in refs):
            issues.append("unresolved_reference")
        related = list(to_by_head.get(pred.i, []))
        if object_index is not None:
            related.extend(decision for decision in to_by_head.get(object_index, [])
                           if decision["marker_index"] not in {item["marker_index"] for item in related})
        related.extend(decision for decision in to_by_complement.get(pred.i, [])
                       if decision["marker_index"] not in {item["marker_index"] for item in related})
        event_to_attachments = [to_attachment_summary(decision) for decision in related]
        governing = {pred.i, object_index}
        for decision in related:
            # Abstention matters when an infinitive reading was actually open.
            # A prepositional "to" with a closed noun, as in "leads to flooding",
            # stays with the existing attachment map.
            ambiguous_infinitive = (
                "infinitival_complement" in decision["structural_candidates"]
                and decision["observed_complement_pos"] == "VERB")
            if (decision["selected"] == "unresolved" and decision["head_index"] in governing
                    and ambiguous_infinitive):
                issues.append("to_attachment_unresolved")
            if decision["selected"] != "directional_pp" or decision["head_index"] not in governing:
                continue
            complement = doc[decision["complement_index"]]
            already = next((item for item in prepositions
                            if item["token_index"] == decision["marker_index"]), None)
            if already:
                ref = already["mention_id"]
            else:
                ref = mention(complement, candidate)
                mentions[ref]["kind"] = "entity"
                prepositions.append(dict(
                    preposition="to", mention_id=ref, token_index=decision["marker_index"],
                    attached_to_token_index=decision["head_index"],
                    role="destination", role_basis="to_attachment_classifier",
                ))
            if ref not in roles.setdefault("destination", []):
                roles["destination"].append(ref)
        event_id = f"frame_{pred.i}"
        event = dict(
            frame_id=event_id, kind=kind, meaning=meaning,
            predicate=dict(text=pred.text, lemma=lemma, token_index=pred.i,
                           start=pred.idx, end=pred.idx + len(pred.text)),
            roles=roles, prepositional_arguments=prepositions, assertion=assertion,
            modifiers=[dict(text=c.text, lemma=c.lemma_, token_index=c.i)
                       for c in pred.children if c.dep_ == "advmod"],
            context={}, issues=list(dict.fromkeys(issues)),
            to_attachments=event_to_attachments,
            extraction_status="draft", eligible_for_world_state=False,
            provenance=dict(method="local_frame_lexicon_and_structural_evidence",
                            argument_source=record.get("pair_source", "dependency_unary"),
                            semantic_source="frame_lexicon" if lemma in FRAME_LEXICON else "unknown",
                            subject_attachment_candidates=candidate.get("subject_attachment_candidates", [])),
        )
        events.append(event)
        if meaning == "quantity_change":
            event["change"] = dict(direction=CHANGE_DIRECTIONS[lemma],
                                   theme=roles.get("theme", []),
                                   influence=roles.get("influence", []),
                                   magnitude=None, baseline=None,
                                   interpretation="lexically_expressed_change_not_verified_effect")
        event_by_predicate[pred.i] = event
        if pred.dep_ == "relcl" and pred.head.pos_ in {"NOUN", "PROPN"}:
            event["context"]["relative_to"] = mention(pred.head, candidate)
        if pred.dep_ in {"ccomp", "xcomp", "acl"} and pred.head.i in event_by_predicate:
            event["context"]["embedded_under"] = event_by_predicate[pred.head.i]["frame_id"]
            event["context"]["enclosing_assertion_status"] = event_by_predicate[pred.head.i]["assertion"]["status"]

    # Choice structure is independent of causal direction. Do not assert that
    # either branch happened or assume logical exclusive-or from the word "or".
    for record in evidence:
        candidate = record.get("selected_candidate", {})
        current = event_by_predicate.get(candidate.get("index"))
        lo = candidate.get("clause_start", 0)
        if current is None or lo == 0 or doc[lo - 1].lower_ != "or":
            continue
        previous_index = candidate.get("shared_subject_from", candidate.get("previous_predicate"))
        previous = event_by_predicate.get(previous_index)
        if previous is None:
            continue
        alternatives.append(dict(
            group_id=f"alternatives_{len(alternatives)}", connective="or",
            connective_token_index=lo - 1, branches=[previous["frame_id"], current["frame_id"]],
            exclusivity="unspecified",
            attachment_status="ambiguous" if candidate.get("subject_attachment_candidates") else "proposed",
            attachment_candidates=candidate.get("subject_attachment_candidates", []),
        ))

    # Associate relative descriptions with a branch via mention identity. This
    # preserves the condition/context rather than asserting an actual outcome.
    for group in alternatives:
        for branch_id in group["branches"]:
            branch = next(e for e in events if e["frame_id"] == branch_id)
            branch["context"]["alternative_group"] = group["group_id"]
            branch["context"]["branch_id"] = branch_id
            branch_mentions = {r for refs in branch["roles"].values() for r in refs}
            for event in events:
                if event["context"].get("relative_to") in branch_mentions:
                    event["context"]["alternative_group"] = group["group_id"]
                    event["context"]["branch_id"] = branch_id

    return dict(events=events, entities=list(mentions.values()), alternatives=alternatives)


@dataclass
class ComplementRelation:
    relation_id: str
    namespace: str
    relation_type: str
    parent_frame: str
    child_frame: str
    child_entailment: str
    parent_assertion_status: str
    provenance: dict
    controller_link: Optional[dict] = None


def interpret_complements(text, propositions):
    """Interpret one resource-backed family; do not infer successful outcomes.

    not_entailed is absence of an entailment, NOT evidence that the child failed.
    Occurrence labels report textual commitment, never externally verified truth.
    """
    doc = get_nlp()(text)
    by_id = {p.frame_id: p for p in propositions}
    by_index = {p.predicate_index: p for p in propositions}
    relations = []
    for child in propositions:
        parent = by_id.get(child.parent_frame_id)
        if parent is None:
            continue
        parent_token, child_token = doc[parent.predicate_index], doc[child.predicate_index]
        candidates = matching_complement_adapters(parent_token, child_token)
        matches = [a for a in candidates if a.recovered_frame_matcher(parent, child)]
        adapter = matches[0] if len(matches) == 1 else None
        attempts = [dict(
            a.provenance, namespace=a.namespace, relation_type=a.relation_type,
            lexical_member=parent.predicate_lemma.lower() in a.documented_members,
            resource_match=a in matches,
        ) for a in get_complement_adapters()]
        provenance = dict(
            # Preserve N's single-adapter provenance fields for consumers.
            **(adapter.provenance if adapter else
               get_complement_adapters()[0].provenance if len(attempts) == 1 else {}),
            resource_match=adapter is not None,
            lexical_member=any(a["lexical_member"] for a in attempts),
            reason=(adapter.match_reason if adapter else "ambiguous_adapter_match"
                    if len(matches) > 1 else "unsupported_lemma_or_frame"),
            entailment_source="explicit_application_policy" if adapter else "abstention",
            adapter_evaluations=attempts,
        )
        relations.append(ComplementRelation(
            relation_id=f"complement_{len(relations)}",
            namespace=adapter.namespace if adapter else "parliament:proposition",
            relation_type=adapter.relation_type if adapter else "UNRESOLVED",
            parent_frame=parent.frame_id, child_frame=child.frame_id,
            child_entailment=adapter.child_entailment if adapter else "unknown",
            parent_assertion_status=parent.assertion["status"],
            provenance=provenance,
            controller_link=next((link for link in parent.event_links
                                  if link["child_frame"] == child.frame_id), None),
        ))
    incoming = {r.child_frame: r for r in relations}
    enriched = []
    for frame in propositions:
        status = frame.assertion.get("status", "unresolved")
        relation = incoming.get(frame.frame_id)
        root_asserted = (frame.dependency_role == "ROOT" and frame.parent_frame_id is None)
        head = by_index.get(frame.predicate_head_index)
        token = doc[frame.predicate_index]
        independent_conjunct = (
            frame.dependency_role == "conj" and frame.parent_frame_id is None
            and token.tag_ in {"VBD", "VBP", "VBZ"}
            and head is not None and head.dependency_role == "ROOT"
            and head.assertion.get("statuses") == ["asserted"]
            and any(t.dep_ == "cc" and t.lower_ in {"and", "but", "yet"}
                    for t in token.head.children)
        )
        root_asserted = root_asserted or independent_conjunct
        occurrence = ("asserted_in_text" if root_asserted and status == "asserted" else
                      "denied_in_text" if root_asserted and status == "denied" else "unknown")
        enriched.append(dict(
            asdict(frame), occurrence_status=occurrence,
            child_entailment=relation.child_entailment if relation else "not_applicable",
            occurrence_basis="embedded_event_not_promoted" if relation else "local_textual_scope",
            eligible_for_world_state=False,
        ))
    return enriched, [asdict(r) for r in relations]


def parse_world_state(text, policy):
    """Parallel causal and draft event/state interpretations of the same text."""
    propositions = extract_proposition_frames(text)
    enriched, complement_relations = interpret_complements(text, propositions)
    hypotheses = {r.get("selected_candidate", {}).get("index"):
                  r.get("selected_candidate", {}).get("structural_hypothesis")
                  for r in collect_claim_evidence(text, include_complement_parents=True)}
    for packet in enriched:
        packet["structural_hypothesis"] = hypotheses.get(packet["predicate_index"])
        if packet["structural_hypothesis"]:
            packet["occurrence_status"] = "unknown"
            packet["occurrence_basis"] = "provisional_structural_hypothesis"
    context_doc = get_nlp()(text)
    frame_by_id = {f.frame_id: f for f in propositions}
    for frame, packet in zip(propositions, enriched):
        contexts = []
        for child_id in frame.complement_frame_ids:
            child = frame_by_id[child_id]
            tokens = sorted(context_doc[child.predicate_index].subtree, key=lambda t: t.i)
            # This is an observed contiguous syntactic subtree, not the full
            # semantic extent of an event (inherited controllers may be outside).
            contiguous = [t.i for t in tokens] == list(range(tokens[0].i, tokens[-1].i + 1))
            argument = (dict(kind="proposition", frame_id=child_id,
                             span_basis="contiguous_dependency_subtree",
                             start=tokens[0].idx, end=tokens[-1].idx + len(tokens[-1].text),
                             token_start=tokens[0].i, token_end=tokens[-1].i + 1)
                        if contiguous else None)
            context = build_argument_context(context_doc, [frame.subject, argument], frame.predicate_index)
            context["target_proposition_id"] = child_id
            if not contiguous:
                context["status"] = "discontinuous_proposition_span"
            contexts.append(context)
        if not contexts:
            contexts.append(build_argument_context(
                context_doc, [frame.subject, frame.object], frame.predicate_index))
        packet["argument_contexts"] = contexts
    occurrence_by_id = {p["frame_id"]: p["occurrence_status"] for p in enriched}
    by_index = {f.predicate_index: f for f in propositions}
    structure = extract_event_state_structure(text)
    for event in structure["events"]:
        frame = by_index.get(event["predicate"]["token_index"])
        event["proposition_frame_id"] = frame.frame_id if frame else None
        event["structural_hypothesis"] = hypotheses.get(frame.predicate_index) if frame else None
        if REPAIR_ENABLED:
            temporal_refs = {a["mention_id"] for a in event["prepositional_arguments"]
                             if a["preposition"].lower() == "by"
                             and any(temporal_by_object(t) for t in context_doc[a["token_index"]].children
                                     if t.dep_ == "pobj")}
            for role in ("agent", "actor", "influence"):
                if role in event["roles"]:
                    event["roles"][role] = [r for r in event["roles"][role] if r not in temporal_refs]
                    if not event["roles"][role]:
                        del event["roles"][role]
        event["attachments"] = frame.attachments if frame else []
        event["quantification"] = frame.quantification if frame else []
        event["event_links"] = frame.event_links if frame else []
        event["complement_proposition_ids"] = list(frame.complement_frame_ids) if frame else []
        event["occurrence_status"] = occurrence_by_id.get(event["proposition_frame_id"], "unknown")
        event["complement_relation_types"] = [r["relation_type"] for r in complement_relations
                                               if r["parent_frame"] == event["proposition_frame_id"]]
    return dict(
        schema_version=9, text=text, causal_claims=parse_claims(text, policy),
        to_attachment_analyses=classify_to_attachments(context_doc),
        propositions=enriched, complement_relations=complement_relations,
        proposition_interpretation="structure_and_textual_commitment_not_verified_world_facts",
        **structure,
        limitations=["draft_frames_are_not_world_facts", "no_cross_probe_context",
                     "unresolved_pronouns_are_not_linked", "no_inferred_causal_consequences"],
    )

def print_proposition_frames(
    sentence
):

    frames = (
        extract_proposition_frames(
            sentence
        )
    )


    print(
        f"\nSentence: {sentence}"
    )


    for frame in frames:

        subject = (
            frame.subject[
                "text"
            ]
            if frame.subject
            is not None
            else "?"
        )


        obj = (
            frame.object[
                "text"
            ]
            if frame.object
            is not None
            else "?"
        )


        # ----------------------------------------------------
        # If this proposition takes another proposition/event
        # as its structural object, display that event instead
        # of pretending the local noun is the whole object.
        #
        # Example:
        #
        # The flood caused the library to close.
        #
        # p0: flood --caused--> [EVENT p1]
        # p1: library --close--> ?
        # ----------------------------------------------------

        if (
            frame.event_object_frame_id
            is not None
        ):

            object_display = (
                f"[EVENT "
                f"{frame.event_object_frame_id}]"
            )

        else:

            object_display = (
                obj
            )


        print(
            f"  {frame.frame_id}: "
            f"{subject} "
            f"--{frame.predicate_text}--> "
            f"{object_display}"
        )


        # ----------------------------------------------------
        # Display the syntactic object separately from event-link controllers.
        #
        # In:
        #
        # The flood caused the library to close.
        #
        # "the library" is still structurally important
        # because it becomes the subject/controller of "close".
        # ----------------------------------------------------

        if (
            frame.event_object_frame_id
            is not None
        ):

            print(
                f"      syntactic object="
                f"{obj}"
            )

        for link in frame.event_links:
            controller = link["controller_candidate"]
            print(f"      child={link['child_frame']}; controller_candidate="
                  f"{controller['text'] if controller else '?'}; "
                  f"source={link['controller_source']}; support={link['controller_support']}; "
                  f"configuration={link['configuration']}")


        # ----------------------------------------------------
        # Show frame-to-frame embedding from the child's side.
        # ----------------------------------------------------

        if (
            frame.parent_frame_id
            is not None
        ):

            print(
                f"      embedded_under="
                f"{frame.parent_frame_id}"
            )


        # ----------------------------------------------------
        # Show any complement frames attached to this frame.
        # ----------------------------------------------------

        if (
            frame.complement_frame_ids
        ):

            print(
                f"      complement_frames="
                f"{frame.complement_frame_ids}"
            )


        print(
            f"      assertion="
            f"{frame.assertion.get('status')}"
        )

        if frame.to_attachment is not None:
            attachment = frame.to_attachment
            print(f"      to_attachment={attachment['selected_id']}:"
                  f"{attachment['selected']}")


        print(
            f"      source="
            f"{frame.candidate_source}"
        )


        if frame.issues:

            print(
                f"      issues="
                f"{frame.issues}"
            )

def print_event_states(structure):
    mentions = {m["mention_id"]: m for m in structure["entities"]}
    print("   EVENTS / STATES (draft descriptions; no graph commitment)")
    for event in structure["events"]:
        roles = "; ".join(f"{role}=" + ", ".join(mentions[r]["text"] for r in refs)
                          for role, refs in event["roles"].items())
        print(f"   {event['frame_id']}: {event['predicate']['text']} [{event['meaning']}] | {roles}")
        print(f"     Assertion: {event['assertion']['status']}; context: {event['context'] or 'local clause'}")
        if "occurrence_status" in event:
            print(f"     Textual occurrence commitment: {event['occurrence_status']}")
        if event.get("complement_relation_types"):
            print("     Complement semantics: " + ", ".join(event["complement_relation_types"]))
        for attachment in event.get("to_attachments", []):
            print(f"     To-attachment: {attachment['selected_id']}:{attachment['selected']}"
                  f" ({attachment['head_lemma']} to {attachment['complement_lemma']})")
        for attachment in event.get("attachments", []):
            print(f"     Attachment: {attachment['head_text']} | role={attachment['role']}"
                  f" ({attachment['role_basis']})")
        for quantifier in event.get("quantification", []):
            print(f"     Quantifier: {quantifier['text']} | scope={quantifier['scope_status']}")
        if "change" in event:
            print(f"     Change: {event['change']['direction']}; magnitude/baseline: unspecified")
        if event["modifiers"]:
            print("     Modifiers: " + ", ".join(m["text"] for m in event["modifiers"]))
        for argument in event["prepositional_arguments"]:
            print(f"     {argument['preposition']}: {mentions[argument['mention_id']]['text']}")
        if event["issues"]:
            print("     Flags: " + "; ".join(event["issues"]))
    for group in structure["alternatives"]:
        print("   Alternatives: " + " OR ".join(group["branches"]) +
              f" (attachment {group['attachment_status']}; exclusivity unspecified)")

USER_SUITE_PATH = Path(__file__).resolve().parent / "diagnostics" / "parsing_game_S_user_probes.json"
USER_SUITE_LIMIT = 5


def load_user_suite(path):
    """Read user probes separately from labelled evaluation and training data."""
    path = Path(path)
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (not isinstance(payload, dict) or payload.get("schema_version") != 1
                or not isinstance(payload.get("entries"), list)):
            raise ValueError("invalid user-suite schema")
        entries = payload["entries"]
        if any(not isinstance(e, dict) or not isinstance(e.get("sentence"), str)
               or not e["sentence"].strip() for e in entries):
            raise ValueError("each probe must contain nonempty sentence text")
        return entries[-USER_SUITE_LIMIT:]
    except (ValueError, OSError) as error:
        raise ValueError(f"Cannot read user suite {path}; file left unchanged: {error}") from error


def save_user_suite(path, entries):
    """Atomically replace the rolling file; do not archive evicted user text."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(schema_version=1, entries=entries[-USER_SUITE_LIMIT:])
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def append_user_probe(entries, sentence):
    sentence = sentence.strip()
    if not sentence:
        return entries[-USER_SUITE_LIMIT:]
    return (entries + [dict(sentence=sentence,
                            added_at=datetime.now().astimezone().isoformat())])[-USER_SUITE_LIMIT:]


def evaluate_user_suite(entries, policy):
    """Unlabelled observations, never accuracy scores or CEM training targets."""
    evaluated = []
    for entry in entries:
        result = {k: v for k, v in entry.items() if k != "last_result"}
        try:
            parsed = parse_world_state(entry["sentence"], policy)
            structure = {key: parsed[key] for key in ("events", "entities", "alternatives")}
            result["last_result"] = dict(status="unscored", claims=parsed["causal_claims"],
                                        structure=structure, propositions=parsed["propositions"],
                                        complement_relations=parsed["complement_relations"],
                                        to_attachment_analyses=parsed["to_attachment_analyses"])
        except Exception as error:
            # Keep the failing input inspectable and continue the other probes.
            result["last_result"] = dict(status="error", error_type=type(error).__name__,
                                        message=str(error))
        evaluated.append(result)
    return evaluated


def print_user_suite(entries):
    if not entries:
        return
    print(f"\nYOUR SENTENCE PROBES ({len(entries)}/{USER_SUITE_LIMIT}, oldest first)")
    print("Exploratory output: no expected answers supplied, so no accuracy score.")
    print("Eligibility is a heuristic gate, not verified causality or calibrated confidence.")
    print("Implicit/distributed causality and counterfactuals are not reliably supported.")
    for index, entry in enumerate(entries, 1):
        print(f"\n{index}. {entry['sentence']}")
        result = entry["last_result"]
        if result["status"] == "error":
            print(f"   ERROR: {result['error_type']}: {result['message']}")
            continue
        if "structure" in result:
            print_event_states(result["structure"])
        for analysis in result.get("to_attachment_analyses", []):
            reason = f" ({analysis['abstention_reason']})" if analysis.get("abstained") else ""
            print(f"   To-attachment: {analysis['head_lemma']} to {analysis['complement_lemma']}"
                  f" = {analysis['selected_id']}:{analysis['selected']}{reason}")
        print("   CAUSAL CLAIMS")
        for frame in result.get("propositions", []):
            subject = (frame["subject"] or {}).get("text", "?")
            target = ("[EVENT " + ", ".join(frame["complement_frame_ids"]) + "]"
                      if frame["complement_frame_ids"] else (frame["object"] or {}).get("text", "?"))
            print(f"   Proposition {frame['frame_id']}: {subject} --{frame['predicate_text']}--> {target}")
            print(f"     Occurrence: {frame.get('occurrence_status', 'unknown')}; "
                  f"child entailment: {frame.get('child_entailment', 'unknown')}")
            for link in frame.get("event_links", []):
                controller = link["controller_candidate"]
                print(f"     Child {link['child_frame']}: controller_candidate="
                      f"{controller['text'] if controller else '?'}; "
                      f"source={link['controller_source']}; support={link['controller_support']}; "
                      f"configuration={link['configuration']}")
        for relation in result.get("complement_relations", []):
            print(f"   {relation['namespace']}:{relation['relation_type']}: "
                  f"{relation['parent_frame']} -> {relation['child_frame']}; "
                  f"child={relation['child_entailment']}; parent={relation['parent_assertion_status']}")
        for claim in result["claims"]:
            candidate = claim["evidence"].get("selected_candidate", {})
            predicate_index = candidate.get("index")
            predicate_text = next((t["text"] for t in claim["evidence"]["tokens"]
                                   if t["index"] == predicate_index), "not recovered")
            if claim.get("target_kind") == "proposition":
                relation = f"{claim['source'] or '?'} -> [EVENT {', '.join(claim['target_proposition_ids']) or 'unresolved'}]"
            elif claim["source"] is not None:
                relation = f"{claim['source']} -> {claim['target']}"
            else:
                relation = " / ".join(e["text"] for e in claim["entities"]) or "No argument pair recovered"
            print(f"   Predicate: {predicate_text} | {relation}")
            print(f"   Causal interpretation: {claim['relation_type']}; assertion: {claim['assertion_status']}; "
                  f"eligible: {'yes' if claim['eligible_for_world_state'] else 'no'}")
            print(f"   Evidence: {candidate.get('source', 'none')}; "
                  f"lexical match: {candidate.get('matched_pattern') or 'none'}")
            if claim["validation_reasons"]:
                print("   Flags: " + "; ".join(claim["validation_reasons"]))


def main(argv=None):
    global REPAIR_ENABLED
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--population", type=int, default=256)
    parser.add_argument("--generations", type=int, default=80)
    parser.add_argument("--perfect-patience", type=int, default=5,
                        help="Stop after this many consecutive perfect training populations; 0 disables.")
    parser.add_argument("--output-dir", type=Path, default=Path("diagnostics"))
    parser.add_argument("--load-policy", type=Path)
    parser.add_argument("--sentence", help="Parse a sentence using a loaded or newly trained policy.")
    parser.add_argument("--add-sentence", help="Add a sentence to the rolling user suite without prompting.")
    parser.add_argument("--no-prompt", action="store_true", help="Replay saved probes without asking for input.")
    parser.add_argument("--user-suite", type=Path, default=USER_SUITE_PATH,
                        help="Rolling file containing only the latest five user probes and results.")
    parser.add_argument("--no-repair", action="store_true", help="Disable Q passive repair and temporal-by validation.")
    args = parser.parse_args(argv)
    REPAIR_ENABLED = not args.no_repair
    try:
        user_entries = load_user_suite(args.user_suite)
    except ValueError as error:
        parser.error(str(error))
    new_sentence = args.add_sentence
    if new_sentence is None and not args.no_prompt and args.sentence is None and sys.stdin.isatty():
        try:
            new_sentence = input("\nEnter a new sentence to test (Enter skips; latest five are kept):\n> ")
        except EOFError:
            new_sentence = None
    if new_sentence is not None and new_sentence.strip():
        user_entries = append_user_probe(user_entries, new_sentence)
        save_user_suite(args.user_suite, user_entries)
    if args.load_policy:
        policy, history = CEMPolicy.load(args.load_policy), []
    else:
        policy, history = train_policy(seed=args.seed, population_size=args.population,
                                       generations=args.generations, perfect_patience=args.perfect_patience)
    training_run = (history[-1] if history else
                    dict(event_type="cem_termination", stop_reason="loaded_policy", actual_generations=0))
    print("training_run: " + json.dumps(training_run))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    prefix = args.output_dir / ("parsing_game_S_" + stamp)
    policy.save(str(prefix) + ".policy.json")
    with open(str(prefix) + ".jsonl", "w", encoding="utf-8") as log:
        collisions = find_representation_collisions(
            TRAIN_EXAMPLES + TEST_EXAMPLES + NEGATION_GENERALIZATION_EXAMPLES
            + ROBUSTNESS_EXAMPLES + LEXICAL_HOLDOUT + UNKNOWN_HOLDOUT)
        log.write(json.dumps(dict(event_type="representation_audit",
                                  collisions=collisions)) + "\n")
        for record in history:
            log.write(json.dumps(record) + "\n")
        summary = {}
        for name, suite in (
            ("training", TRAIN_EXAMPLES), ("ordinary_holdout", TEST_EXAMPLES),
            ("negation", NEGATION_GENERALIZATION_EXAMPLES),
            ("robustness", ROBUSTNESS_EXAMPLES), ("lexical_holdout", LEXICAL_HOLDOUT),
            ("unknown_holdout", UNKNOWN_HOLDOUT), ("epistemic_holdout", EPISTEMIC_HOLDOUT),
        ):
            metrics, records = evaluate_suite(suite, policy)
            summary[name] = metrics
            print(name + ": " + json.dumps(metrics))
            for record in records:
                log.write(json.dumps(dict(event_type="evaluation", suite=name, **record)) + "\n")
        metrics, records = evaluate_multi_suite(MULTI_RELATION_HOLDOUT, policy)
        summary["multi_relation_holdout"] = metrics
        print("multi_relation_holdout: " + json.dumps(metrics))
        for record in records:
            log.write(json.dumps(dict(event_type="multi_claim_evaluation", **record)) + "\n")
        log.write(json.dumps(dict(event_type="run_summary", seed=args.seed,
                                 population=args.population, generations=args.generations,
                                 training_run=training_run,
                                 feature_names=FEATURE_NAMES, summary=summary)) + "\n")
    Path(str(prefix) + ".txt").write_text(json.dumps(summary, indent=2) + "\n",
                                         encoding="utf-8")
    if user_entries:
        user_entries = evaluate_user_suite(user_entries, policy)
        # User text/results stay ONLY in this rolling file, never the dated logs.
        save_user_suite(args.user_suite, user_entries)
        print_user_suite(user_entries)
        print("\nRolling user suite: " + str(args.user_suite))
    if args.sentence:
        print(json.dumps(parse_world_state(args.sentence, policy), indent=2))
    print("Diagnostics and saved policy: " + str(prefix))
    # ========================================================
    # PROPOSITION FRAME SMOKE TESTS
    # ========================================================

    print_proposition_frames(
        "Exposure induces disease, which triggers inflammation."
    )

    print_proposition_frames(
        "The flood caused the library to close."
    )

    print_proposition_frames(
        "The storm forced the school to close."
    )

    print_proposition_frames(
        "The manager allowed the workers to leave."
    )

if __name__ == "__main__":
    main()
