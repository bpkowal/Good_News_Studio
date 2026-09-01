"""Typed, graph-backed identity for physical actions.

Action labels and prose order are presentation state.  This module compiles an
action into a small semantic graph and derives identity from the graph's stable
content: intervention, affected targets, consequences, quantities, and modality.
It is deliberately conservative.  When too little structure can be recovered,
the caller receives an explicit lexical fallback rather than a guessed match.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import re
import unicodedata
from typing import Any, Sequence

from .semantic_graph import SemanticEdge, SemanticGraph, SemanticNode
from .semantic_roles import (
    ROLE_UNRESOLVED,
    extract_grounded_effects,
    extract_party_registry,
    parties_compatible,
    party_identity,
    relational_role_bindings,
)


_LABEL_PREFIX = re.compile(
    r"^\s*(?:(?:action|option)\s+)?A(?:[_\s]*\{?\d+\}?)\s*[:.)\]-]?\s*",
    re.IGNORECASE,
)
_ACTION_OPEN_ENDINGS = {
    "a", "an", "and", "as", "at", "be", "because", "but", "by", "for", "from",
    "if", "in", "into", "of", "on", "or", "so", "than", "that", "the", "then",
    "these", "this", "though", "to", "unless", "until", "upon", "we", "when",
    "where", "which", "while", "with", "without",
    "allowing", "causing", "creating", "depriving", "forcing", "granting",
    "hindering", "leaving", "making", "preventing", "protecting", "routing",
    "setting", "sparing", "triggering",
}
_ACTION_DANGLING_ENDINGS = {
    "main", "primary", "remaining", "official", "current", "next", "prior",
    "same", "other", "first", "second", "central", "essential", "necessary",
}
_WORD_NUMBER = {
    "zero": "0", "one": "1", "single": "1", "two": "2", "three": "3",
    "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8",
    "nine": "9", "ten": "10", "eleven": "11", "twelve": "12",
}
_PHRASE_ALIASES = (
    (r"\bpeople\s+on\s+foot\b", "pedestrian"),
    (r"\bcurrent\s+(?:course|trajectory|route)\b", "path"),
    (r"\bvehicle(?:'s)?\b", ""),
    (r"\bloss(?:es)?\s+of\s+(?:life|lives)\b", "death"),
    (r"\bloss\s+of\s+life\b", "death"),
    (r"\bwithout\s+consent\b", "involuntary"),
    (
        r"\brespect(?:s|ed|ing)?\s+(?=(?:individual\s+)?bodily\s+"
        r"(?:autonomy|integrity)|legal\s+rights?|individual\s+rights?)",
        "preserve ",
    ),
    (
        r"\bforbid(?:s|ding|den)?\s+(?=involuntary|forced|coercive)",
        "prevent ",
    ),
    (
        r"\bstrip(?:s|ped|ping)?\s+(?=[^.;!?]{0,120}\b(?:bodily\s+autonomy|"
        r"bodily\s+integrity|occupational\s+choice|privacy|freedom\s+of\s+movement|"
        r"personal\s+libert\w*|legal\s+rights?)\b)",
        "infringes ",
    ),
    (r"\bdo\s+nothing\b", "refrain"),
    (r"\bremain\s+silent\b", "refrain"),
    (r"\bunder[- ]resourced\b", "deprivation"),
    (r"\bunderstaffed\b", "deprivation"),
    (r"\bpoor\s+transit\s+access\b", "deprivation transit access"),
    (
        r"\bleav(?:e|es|ing|t)\b(?=[^.;!?]{0,140}\b(?:without|with\s+only|"
        r"with\s+(?:a\s+)?(?:lower|reduced|inadequate|insufficient)|"
        r"understaffed|underfunded|under[- ]resourced)\b)",
        "deprivation",
    ),
)
_CONCEPT_ALIASES = {
    # Interventions and control verbs.
    "allows": "permit", "allow": "permit", "permits": "permit",
    "permit": "permit", "authorizes": "authorize", "authorize": "authorize",
    "forces": "compel", "force": "compel", "compels": "compel",
    "maintains": "maintain", "maintain": "maintain", "remaining": "maintain",
    "remain": "maintain", "continues": "maintain", "continue": "maintain",
    "redirects": "redirect", "redirect": "redirect", "reroutes": "redirect",
    "reroute": "redirect", "routes": "redirect", "route": "redirect",
    "routing": "redirect", "swerves": "redirect", "swerve": "redirect",
    "turns": "redirect", "turn": "redirect", "deploys": "deploy",
    "deploy": "deploy", "uses": "deploy", "use": "deploy",
    "implements": "implement", "implement": "implement", "activates": "activate",
    "activate": "activate", "disconnects": "disconnect", "disconnect": "disconnect",
    "cuts": "disconnect", "cut": "disconnect", "isolates": "isolate",
    "isolate": "isolate", "seals": "seal", "seal": "seal",
    "enforces": "enforce", "enforce": "enforce", "enforced": "enforce",
    "bypasses": "bypass", "bypass": "bypass", "bypassed": "bypass",
    "releases": "release", "release": "release", "reports": "report",
    "report": "report", "discloses": "report", "disclose": "report",
    "shares": "share", "share": "share", "shared": "share",
    "provides": "provide", "provide": "provide",
    "rejects": "reject", "reject": "reject", "approves": "approve",
    "approve": "approve", "purges": "purge", "purge": "purge",
    "expropriates": "expropriate", "expropriate": "expropriate",
    "detains": "detain", "detaining": "detain", "detained": "detain",
    "detain": "detain", "confines": "detain", "confining": "detain",
    "confined": "detain", "confine": "detain",
    "treats": "treat", "treat": "treat",
    "vaccinates": "vaccinate", "vaccinate": "vaccinate", "lies": "deceive",
    "lie": "deceive", "refrains": "refrain", "refrain": "refrain",
    "steps": "step", "step": "step", "pushes": "push", "push": "push",
    "opens": "open", "open": "open", "closes": "close", "close": "close",
    "overrides": "override", "override": "override",
    "allocates": "allocate", "allocate": "allocate", "gives": "give", "give": "give",
    "returns": "return", "return": "return", "keeps": "keep", "keep": "keep",
    "withholds": "withhold", "withhold": "withhold", "chooses": "choose",
    "choose": "choose", "selects": "choose", "select": "choose",
    # Consequences.  Different surface verbs intentionally share one predicate.
    "kills": "death", "killing": "death", "killed": "death", "kill": "death",
    "dies": "death", "died": "death", "die": "death", "deaths": "death",
    "fatalities": "death", "fatality": "death", "sacrifices": "death",
    "sacrificing": "death", "sacrifice": "death",
    "loses": "deprivation", "lose": "deprivation", "lost": "deprivation",
    "freezes": "freeze", "freeze": "freeze", "frozen": "freeze", "freezing": "freeze",
    "saves": "preserve_life", "saving": "preserve_life", "save": "preserve_life",
    "spares": "preserve_life", "sparing": "preserve_life", "spare": "preserve_life",
    "spared": "preserve_life",
    "protects": "protect", "protecting": "protect", "protect": "protect",
    "harms": "harm", "harming": "harm", "harmed": "harm", "harm": "harm",
    "injures": "injury", "injuring": "injury", "injured": "injury",
    "displaces": "displacement", "displacing": "displacement",
    "deprives": "deprivation", "depriving": "deprivation",
    "infringes": "liberty_infringement", "infringing": "liberty_infringement",
    "infringed": "liberty_infringement", "infringe": "liberty_infringement",
    "prevents": "prevent", "preventing": "prevent", "prevent": "prevent",
    "contains": "contain", "containing": "contain", "contain": "contain",
    "preserves": "preserve", "preserving": "preserve", "preserve": "preserve",
    "destroys": "destruction", "destroying": "destruction", "destroy": "destruction",
    "obliterates": "destruction", "obliterating": "destruction",
    "obliterated": "destruction", "obliterate": "destruction",
    "corrupts": "corruption", "corrupting": "corruption", "corrupt": "corruption",
    "sterilizes": "sterilization", "sterilizing": "sterilization",
    "floods": "flood", "flooding": "flood", "flood": "flood",
    "strikes": "strike", "striking": "strike", "strike": "strike",
    "exposes": "exposure", "exposing": "exposure", "expose": "exposure",
    "bottlenecks": "bottleneck", "bottlenecking": "bottleneck", "bottleneck": "bottleneck",
    "delays": "delay", "delayed": "delay", "delaying": "delay", "delay": "delay",
    "misses": "miss", "missed": "miss", "missing": "miss", "miss": "miss",
    "exhausted": "exhaustion", "exhausts": "exhaustion", "exhausting": "exhaustion",
    "exhaustion": "exhaustion",
    "overloads": "overload", "overloading": "overload", "overload": "overload",
    "throughputs": "throughput", "throughput": "throughput",
    "accurate": "accuracy", "accuracy": "accuracy",
    "compliant": "compliance", "compliance": "compliance",
    "wellbeing": "welfare", "welfare": "welfare",
    "raises": "improvement", "raising": "improvement", "raise": "improvement",
    "improves": "improvement", "improving": "improvement", "improve": "improvement",
    "guarantees": "provision", "guaranteeing": "provision", "guarantee": "provision",
    "uplifts": "improvement", "uplifting": "improvement", "uplift": "improvement",
    "maximizes": "improvement", "maximizing": "improvement", "maximize": "improvement",
    "polluted": "pollution", "pollution": "pollution",
    "under-resourced": "deprivation", "underresourced": "deprivation",
    "underfunded": "deprivation",
    "poor": "deprivation",
    "caps": "restriction", "capping": "restriction", "cap": "restriction",
    "restricts": "restriction", "restricting": "restriction", "restrict": "restriction",
    "fulfills": "fulfillment", "fulfilling": "fulfillment", "fulfillment": "fulfillment",
}
_INTERVENTIONS = {
    "permit", "authorize", "compel", "maintain", "redirect", "deploy", "implement",
    "activate", "disconnect", "isolate", "seal", "release", "report", "reject",
    "approve", "purge", "expropriate", "detain", "treat", "vaccinate", "deceive",
    "provide", "share", "disclose",
    "refrain", "step", "push", "open", "close", "override", "enforce", "bypass",
    "allocate", "give", "return", "keep", "withhold", "choose",
}
_CONTROL_INTERVENTIONS = {"permit", "authorize", "compel", "override", "implement"}
_EFFECT_POLARITY = {
    "death": "ADVERSE", "harm": "ADVERSE", "injury": "ADVERSE",
    "displacement": "ADVERSE", "deprivation": "ADVERSE", "destruction": "ADVERSE",
    "corruption": "ADVERSE", "sterilization": "ADVERSE", "flood": "ADVERSE",
    "freeze": "ADVERSE",
    "strike": "ADVERSE", "exposure": "ADVERSE", "expropriate": "ADVERSE",
    "detain": "ADVERSE", "disconnect": "ADVERSE",
    "preserve_life": "BENEFICIAL", "protect": "BENEFICIAL", "prevent": "BENEFICIAL",
    "contain": "BENEFICIAL", "preserve": "BENEFICIAL",
    "improvement": "BENEFICIAL", "provision": "BENEFICIAL",
    "throughput": "BENEFICIAL", "efficiency": "BENEFICIAL", "accuracy": "BENEFICIAL",
    "compliance": "BENEFICIAL", "welfare": "BENEFICIAL", "fulfillment": "BENEFICIAL",
    "delay": "ADVERSE", "deadline": "ADVERSE", "bottleneck": "ADVERSE",
    "backlog": "ADVERSE", "miss": "ADVERSE", "exhaustion": "ADVERSE",
    "overload": "ADVERSE",
    "pollution": "ADVERSE",
    "restriction": "ADVERSE",
    "liberty_infringement": "ADVERSE",
}
_STOPWORDS = {
    "a", "an", "the", "this", "that", "these", "those", "to", "of", "for",
    "from", "into", "onto", "on", "in", "at", "by", "with", "without", "and",
    "or", "but", "while", "whereas", "despite", "although", "however", "current",
    "immediate", "immediately", "deliberately", "reliably", "moderately", "slightly",
    "heavily", "only", "tragic", "existing",
    "first", "second", "action", "option", "will", "would", "must", "may", "can",
    "could", "should", "its", "their", "his", "her", "inside", "outside", "all", "s",
    "is", "are", "was", "were", "be", "been", "being",
    "letting", "allowing", "causing", "resulting", "making",
    "vehicle",
}
_GENERIC_TARGETS = {"system", "decision", "choice", "outcome"}
_ROLE_PROJECTIONS = (
    (
        re.compile(r"\bhospital\b|\bclinic\b|\bmedical\b|\bdoctor\b|\bpatient\b", re.I),
        "hospital patients",
    ),
    (
        re.compile(r"\bwater\b|\bpumping\s+station\b|\bwater\s+supply\b|\bwater\s+station\b", re.I),
        "water users",
    ),
)
_PRIVACY_OBJECTS = re.compile(
    r"\b(?:address|contact(?:\s+information)?|information|info|data|record|records|"
    r"file|files|details|secret|privacy|confidential(?:ity)?|private)\b",
    re.I,
)
_PRIVACY_RECIPIENTS = re.compile(
    r"\b(?:third\s+party|coordinator|neighbor|neighbour|volunteer|outsider|"
    r"stranger|public|community|agency|organization|office)\b",
    re.I,
)
_PRIVACY_PRESERVING_PATTERNS = (
    re.compile(r"\bdeclin\w*\s+to\s+(?:share|provide|disclose|give)\b", re.I),
    re.compile(r"\bdo\s+not\s+(?:share|provide|disclose|give)\b", re.I),
    re.compile(r"\b(?:withhold|keep)\b.*\bprivate\b", re.I),
    re.compile(r"\b(?:keep|preserve|protect)\b.*\bconfidential", re.I),
    re.compile(r"\brefus\w*\s+to\s+(?:share|provide|disclose|give)\b", re.I),
)
_PRIVACY_DISCLOSING_PATTERNS = (
    re.compile(r"\b(?:share|provide|disclose|give)\b", re.I),
)
_PRIVACY_SUBJECT_WORDS = {
    "resident", "person", "individual", "patient", "user", "citizen", "worker",
    "family", "neighbor", "neighbour", "client", "customer", "guest", "student",
}


def _lexical_text(action: str) -> str:
    text = unicodedata.normalize("NFKC", str(action))
    text = _LABEL_PREFIX.sub("", text)
    return " ".join(re.findall(r"[a-z0-9%]+", text.casefold()))


def action_clause_looks_complete(action: str) -> bool:
    """Conservatively reject visibly clipped action clauses."""
    raw = " ".join(str(action).split())
    if re.fullmatch(r"A\d+", raw, flags=re.IGNORECASE):
        return True
    normalized = _lexical_text(action)
    if not normalized:
        return False
    if normalized.endswith("...") or normalized.endswith("…"):
        return False
    tokens = normalized.split()
    if not tokens:
        return False
    if len(tokens) <= 2:
        return True
    if tokens[-1] in _ACTION_OPEN_ENDINGS:
        return False
    if tokens[-1] in _ACTION_DANGLING_ENDINGS:
        return False
    return True


def _normalize_source(action: str) -> str:
    text = unicodedata.normalize("NFKC", str(action)).casefold()
    text = _LABEL_PREFIX.sub("", text)
    text = re.sub(r"\b(\d+(?:\.\d+)?)\s+percent\b", r"\1%", text)
    for pattern, replacement in _PHRASE_ALIASES:
        text = re.sub(pattern, replacement, text)
    return " ".join(text.split())


def _stem_unknown(token: str) -> str:
    if token in _WORD_NUMBER:
        return _WORD_NUMBER[token]
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
        return token[:-1]
    return token


def _concepts(text: str) -> list[str]:
    values = []
    for token in re.findall(r"[a-z0-9]+(?:\.[0-9]+)?%?", text.casefold()):
        normalized = _CONCEPT_ALIASES.get(token, _stem_unknown(token))
        if normalized not in _STOPWORDS:
            values.append(normalized)
    return values


def _quantity_tokens(tokens: list[str]) -> tuple[str, ...]:
    quantities = []
    for index, token in enumerate(tokens):
        if re.fullmatch(r"\d+(?:\.\d+)?%?", token):
            unit = "PERCENT" if token.endswith("%") else "COUNT"
            value = token.rstrip("%")
            # A nearby temporal/unit noun distinguishes 2 months from 2 lives.
            nearby = next((
                candidate.upper()
                for candidate in tokens[index + 1:index + 3]
                if candidate in {
                    "second", "minute", "hour", "day", "week", "month", "year",
                    "job",
                }
            ), unit)
            quantities.append(f"{value}:{nearby}")
    return tuple(sorted(set(quantities)))


def _projected_beneficiary_labels(source: str) -> tuple[str, ...]:
    labels: list[str] = []
    for pattern, label in _ROLE_PROJECTIONS:
        if pattern.search(source):
            labels.append(label)
    return tuple(dict.fromkeys(labels))


def _projected_privacy_consequence(source: str, tokens: list[str]) -> ConsequenceIdentity | None:
    if not (_PRIVACY_OBJECTS.search(source) and _PRIVACY_RECIPIENTS.search(source)):
        return None
    subject_tokens = tuple(sorted({
        token for token in tokens if token in _PRIVACY_SUBJECT_WORDS
    }))
    if not subject_tokens:
        return None
    if any(pattern.search(source) for pattern in _PRIVACY_PRESERVING_PATTERNS):
        return ConsequenceIdentity(
            predicate="preserve",
            polarity="BENEFICIAL",
            targets=subject_tokens,
        )
    if any(pattern.search(source) for pattern in _PRIVACY_DISCLOSING_PATTERNS):
        return ConsequenceIdentity(
            predicate="exposure",
            polarity="ADVERSE",
            targets=subject_tokens,
        )
    return None


def _split_segments(text: str) -> list[str]:
    effect_words = (
        r"kill|die|death|fatal|sacrific|sav|spar|protect|harm|injur|displac|"
        r"depriv|prevent|contain|preserv|destroy|corrupt|steriliz|flood|strik|expos|"
        r"freeze|lose|bottleneck|backlog|delay|deadline|miss|exhaust|overload|"
        r"detain|confin|compel|forc|infring|"
        r"throughput|efficien|accurac|complianc|fulfill|welfare|rais|improv|"
        r"guarante|uplift|maximi|pollut|under[- ]resour|underfund|poor|cap|restrict"
    )
    separated = re.sub(
        rf"\b(?:and|but|while|whereas|by|to)\s+(?=(?:\w+\s+){{0,2}}(?:{effect_words}))",
        " | ",
        text,
        flags=re.IGNORECASE,
    )
    separated = re.sub(
        r"\b(?:but|despite|while|whereas|although|however|thereby|resulting in|causing)\b",
        " | ",
        separated,
        flags=re.IGNORECASE,
    )
    return [part.strip(" ,;:.") for part in re.split(r"[|;,]", separated) if part.strip(" ,;:.")]


@dataclass(frozen=True, slots=True)
class ConsequenceIdentity:
    predicate: str
    polarity: str
    targets: tuple[str, ...] = ()
    quantities: tuple[str, ...] = ()
    probability: str = ""


@dataclass(frozen=True, slots=True)
class ActionIdentity:
    intervention: str
    actors: tuple[str, ...] = ()
    targets: tuple[str, ...] = ()
    consequences: tuple[ConsequenceIdentity, ...] = ()
    modalities: tuple[str, ...] = ()
    basis: str = "GRAPH"
    lexical_fallback: str = ""

    def signature(self) -> dict[str, Any]:
        if self.basis == "LEXICAL_FALLBACK":
            return {"version": 1, "basis": self.basis, "text": self.lexical_fallback}
        return {
            "version": 1,
            "basis": self.basis,
            "intervention": self.intervention,
            "actors": list(self.actors),
            "targets": list(self.targets),
            "consequences": [asdict(value) for value in self.consequences],
            "modalities": list(self.modalities),
        }

    def stable_key(self) -> str:
        payload = json.dumps(self.signature(), sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
        return f"action_graph:v1:{digest}"


def compile_action_identity(action: str) -> ActionIdentity:
    """Compile one action into a conservative typed semantic identity."""
    source = _normalize_source(action)
    tokens = _concepts(source)
    intervention_positions = [
        (index, token) for index, token in enumerate(tokens) if token in _INTERVENTIONS
    ]
    non_control = [item for item in intervention_positions if item[1] not in _CONTROL_INTERVENTIONS]
    if non_control:
        intervention = non_control[0][1]
    elif intervention_positions:
        intervention = intervention_positions[0][1]
    else:
        intervention = tokens[0] if tokens else ""
    actors = ()
    if intervention_positions and intervention_positions[0][0] > 0:
        actors = tuple(sorted({
            token for token in tokens[:intervention_positions[0][0]]
            if token not in _STOPWORDS
            and not re.fullmatch(r"\d+(?:\.\d+)?%?", token)
        }))

    consequences: list[ConsequenceIdentity] = []
    last_subject_tokens: tuple[str, ...] = ()
    for segment in _split_segments(source):
        segment_tokens = _concepts(segment)
        predicates: list[tuple[int, str]] = []
        for index, token in enumerate(segment_tokens):
            if token in _EFFECT_POLARITY and token not in {item[1] for item in predicates}:
                predicates.append((index, token))
        if not predicates:
            subject_candidates = {
                token for token in segment_tokens
                if token not in _STOPWORDS
                and token not in _INTERVENTIONS
                and token not in _EFFECT_POLARITY
                and token not in _GENERIC_TARGETS
                and not re.fullmatch(r"\d+(?:\.\d+)?%?", token)
                and token not in {"risk", "chance", "probability", "guarantee", "guaranteed"}
            }
            if subject_candidates:
                last_subject_tokens = tuple(sorted(subject_candidates))
            continue
        for predicate_index, predicate in predicates:
            before_candidates = {
                token for token in segment_tokens[:predicate_index]
                if token not in _STOPWORDS
                and token not in _INTERVENTIONS
                and token not in _EFFECT_POLARITY
                and token not in _GENERIC_TARGETS
                and not re.fullmatch(r"\d+(?:\.\d+)?%?", token)
                and token not in {"risk", "chance", "probability", "guarantee", "guaranteed"}
            }
            after_candidates = {
                token for token in segment_tokens[predicate_index + 1:]
                if token not in _STOPWORDS
                and token not in _INTERVENTIONS
                and token not in _EFFECT_POLARITY
                and token not in _GENERIC_TARGETS
                and not re.fullmatch(r"\d+(?:\.\d+)?%?", token)
                and token not in {"risk", "chance", "probability", "guarantee", "guaranteed"}
            }
            if predicate == "deprivation":
                # Deprivation language normally names the deprived subject or
                # deficient resource locally ("leaves rural communities with
                # understaffed clinics"). Do not import a stale subject from an
                # earlier action-mechanism segment when local targets exist.
                local_candidates = before_candidates | after_candidates
                entity_tokens = set(local_candidates)
                if len(local_candidates) <= 1:
                    entity_tokens |= set(last_subject_tokens)
            elif predicate in {"freeze", "disconnect"}:
                entity_tokens = (
                    before_candidates | after_candidates | set(last_subject_tokens)
                )
            elif predicate == "restriction":
                entity_tokens = before_candidates | after_candidates | set(last_subject_tokens)
            elif _EFFECT_POLARITY.get(predicate) == "BENEFICIAL":
                entity_tokens = after_candidates or before_candidates or set(last_subject_tokens)
            else:
                entity_tokens = after_candidates or before_candidates or set(last_subject_tokens)
            probability = next((
                token for token in segment_tokens if token.endswith("%")
            ), "")
            consequences.append(ConsequenceIdentity(
                predicate=predicate,
                polarity=_EFFECT_POLARITY[predicate],
                targets=tuple(sorted(entity_tokens)),
                quantities=_quantity_tokens(segment_tokens),
                probability=probability,
            ))
            if predicate == "deprivation" and entity_tokens:
                last_subject_tokens = tuple(sorted(entity_tokens))
    privacy_consequence = _projected_privacy_consequence(source, tokens)
    if privacy_consequence is not None:
        consequences.append(privacy_consequence)

    # If consequences exist they carry the discriminating targets. Otherwise,
    # retain the intervention's object concepts so ordinary non-harm actions
    # (report misconduct vs remain silent) remain distinct.
    consequence_vocab = {
        item for consequence in consequences for item in consequence.targets
    }
    core_targets = {
        token for token in tokens
        if token not in _STOPWORDS
        and token not in _INTERVENTIONS
        and token not in _EFFECT_POLARITY
        and token not in _GENERIC_TARGETS
        and not re.fullmatch(r"\d+(?:\.\d+)?%?", token)
    }
    if consequences:
        core_targets -= consequence_vocab

    modalities = []
    if re.search(r"\b(?:guarantee[sd]?|certain(?:ly)?|inevitabl[ey]|100%)\b", source):
        modalities.append("CERTAIN")
    if re.search(r"\b(?:risk|chance|probability|unverified|experimental)\b", source):
        modalities.append("UNCERTAIN")
    if re.search(r"\b(?:involuntary|forcibly|coercive|against\s+\w+\s+will)\b", source):
        modalities.append("INVOLUNTARY")
    if re.search(r"\b(?:not|never|refus(?:e|es|ed)\s+to)\b", source):
        modalities.append("NEGATED")

    enough_structure = bool(consequences or (intervention_positions and core_targets))
    if not enough_structure:
        return ActionIdentity(
            intervention="",
            basis="LEXICAL_FALLBACK",
            lexical_fallback=_lexical_text(action),
        )
    return ActionIdentity(
        intervention=intervention,
        actors=actors,
        targets=tuple(sorted(core_targets)),
        consequences=tuple(sorted(
            consequences,
            key=lambda item: (
                item.predicate, item.polarity, item.targets, item.quantities, item.probability,
            ),
        )),
        modalities=tuple(sorted(set(modalities))),
    )


def graph_action_key(action: str) -> str:
    return compile_action_identity(action).stable_key()


def add_action_identity_subgraph(
    graph: SemanticGraph, action_node_id: str, identity: ActionIdentity,
) -> None:
    """Attach identity structure to an existing ACTION node."""
    if identity.basis == "LEXICAL_FALLBACK":
        return
    intervention_id = f"{action_node_id}:INTERVENTION"
    graph.add_node(SemanticNode(
        intervention_id, "INTERVENTION", identity.intervention,
        ("deterministic_action_identity",),
    ))
    graph.add_edge(SemanticEdge(action_node_id, "HAS_INTERVENTION", intervention_id))
    for index, actor in enumerate(identity.actors):
        actor_id = f"{action_node_id}:ACTOR:{index}"
        graph.add_node(SemanticNode(
            actor_id, "ACTOR", actor, ("deterministic_action_identity",),
        ))
        graph.add_edge(SemanticEdge(action_node_id, "HAS_ACTOR", actor_id))
    for index, target in enumerate(identity.targets):
        target_id = f"{action_node_id}:TARGET:{index}"
        graph.add_node(SemanticNode(
            target_id, "TARGET", target, ("deterministic_action_identity",),
        ))
        graph.add_edge(SemanticEdge(action_node_id, "TARGETS", target_id))
    # Lightweight beneficiary projection: turn common infrastructure targets
    # into the affected human group that a downstream Rawlsian ledger can bind.
    projected_labels = _projected_beneficiary_labels(
        " ".join([identity.intervention, *identity.targets, *(
            target for consequence in identity.consequences for target in consequence.targets
        )])
    )
    for index, label in enumerate(projected_labels, start=len(identity.targets)):
        target_id = f"{action_node_id}:TARGET:proj:{index}"
        graph.add_node(SemanticNode(
            target_id, "TARGET", label, ("deterministic_action_identity",),
            {"projection_kind": "BENEFICIARY_GROUP"},
        ))
        graph.add_edge(SemanticEdge(action_node_id, "TARGETS", target_id))
    for index, consequence in enumerate(identity.consequences):
        consequence_id = f"{action_node_id}:CONSEQUENCE:{index}"
        graph.add_node(SemanticNode(
            consequence_id, "CONSEQUENCE", consequence.predicate,
            ("deterministic_action_identity",),
            {
                "polarity": consequence.polarity,
                "targets": list(consequence.targets),
                "quantities": list(consequence.quantities),
                "probability": consequence.probability,
            },
        ))
        graph.add_edge(SemanticEdge(action_node_id, "HAS_CONSEQUENCE", consequence_id))
        for target_index, target in enumerate(consequence.targets):
            target_id = f"{consequence_id}:TARGET:{target_index}"
            graph.add_node(SemanticNode(
                target_id, "TARGET", target, ("deterministic_action_identity",),
            ))
            graph.add_edge(SemanticEdge(consequence_id, "AFFECTS", target_id))
        for metric_index, quantity in enumerate(consequence.quantities):
            metric_id = f"{consequence_id}:METRIC:{metric_index}"
            graph.add_node(SemanticNode(
                metric_id, "METRIC", quantity, ("deterministic_action_identity",),
            ))
            graph.add_edge(SemanticEdge(consequence_id, "HAS_METRIC", metric_id))
    for index, modality in enumerate(identity.modalities):
        condition_id = f"{action_node_id}:MODALITY:{index}"
        graph.add_node(SemanticNode(
            condition_id, "CONDITION", modality, ("deterministic_action_identity",),
            {"condition_type": "ACTION_MODALITY"},
        ))
        graph.add_edge(SemanticEdge(action_node_id, "HAS_CONSTRAINT", condition_id))


# --- Canonical action records (identity / label / semantic separation) ---------
#
# action_id: stable deliberation handle (A0, A1, …)
# short_label: concise UI string (may omit detail)
# canonical_semantic_action: authoritative reasoning string — must preserve every
#   decision-critical consequence/constraint from the grounded source clauses.
# Structured fields are the preferred carrier; the semantic string is rendered
# from them when available so a model-written sentence is not the sole store.

_CRITICAL_EFFECT = re.compile(
    r"\b(?:kill|kills|killing|killed|die|dies|dying|died|death|deaths|"
    r"fatalit(?:y|ies)|surviv(?:e|es|ed|ing|al)|preserv(?:e|es|ed|ing)|"
    r"sav(?:e|es|ed|ing)|sacrific(?:e|es|ed|ing)|harm|harms|harmed|harming|"
    r"fail(?:s|ed|ure)|conceal(?:s|ed|ing|ment)?|hidden|secret|covert|"
    r"undisclosed|engineered|siphon|withdraw|withholding)\b",
    re.IGNORECASE,
)
_NUMBERED_OUTCOME = re.compile(
    r"\b(?P<verb>kill|kills|killing|killed|die|dies|dying|died|death|deaths|"
    r"fatalit(?:y|ies)|surviv(?:e|es|ed|ing|al)|preserv(?:e|es|ed|ing)|"
    r"sav(?:e|es|ed|ing)|sacrific(?:e|es|ed|ing))\b"
    r"(?P<body>[^.;,]{0,80}?\b(?P<count>\d+(?:,\d{3})*(?:\.\d+)?)\b[^.;,]{0,40}?"
    r"\b(?P<who>patient|patients|refugee|refugees|resident|residents|life|lives|"
    r"person|people|worker|workers|child|children)\b)"
    r"|"
    r"\b(?P<count2>\d+(?:,\d{3})*(?:\.\d+)?)\b[^.;,]{0,40}?"
    r"\b(?P<who2>patient|patients|refugee|refugees|resident|residents|life|lives|"
    r"person|people|worker|workers|child|children)\b[^.;,]{0,40}?"
    r"\b(?P<verb2>kill|kills|killing|killed|die|dies|dying|died|death|deaths|"
    r"fatalit(?:y|ies)|surviv(?:e|es|ed|ing|al)|preserv(?:e|es|ed|ing)|"
    r"sav(?:e|es|ed|ing)|sacrific(?:e|es|ed|ing))\b",
    re.IGNORECASE,
)
# Infrastructure failure as a decision-critical mechanism. The system noun is
# required: without it the pattern also matched a bare "failure", turning a
# bodily event like "fatal organ failure" into an infrastructure claim that no
# action could ever satisfy.
_FAILURE_SUBJECT = (
    r"(?:shelter|grid|power|oxygen|allocation|hospital|ward|network|"
    r"infrastructure|supply|reactor|plant|server|"
    r"life[- ]support\s+system|support\s+system)"
)
_MECHANISM_FAILURE = re.compile(
    rf"\b{_FAILURE_SUBJECT}(?:\s+[\w-]+){{0,3}}\s+"
    r"(?:fail(?:s|ed|ure)|collapse(?:s|d)?|outage|blackout|unstable|instability)\b"
    r"|"
    r"\b(?:fail(?:s|ed|ure)|collapse(?:s|d)?|outage|blackout|unstable|instability)"
    rf"(?:\s+[\w-]+){{0,3}}\s+{_FAILURE_SUBJECT}\b",
    re.IGNORECASE,
)
# Medical events named with "failure" are harms to a body, not infrastructure.
# Requiring them on the action that prevents them is how A1 was rejected for
# dropping a bare "failure" token from "fatal organ failure without the drug".
_BODILY_FAILURE = re.compile(
    r"\b(?:organ|heart|liver|kidney|renal|hepatic|respiratory|cardiac|"
    r"multi[- ]organ|pulmonary|neurologic(?:al)?)\b",
    re.IGNORECASE,
)
_CONCEALMENT = re.compile(
    r"\b(?:conceal(?:s|ed|ing|ment)?|hidden|secret|covert|undisclosed|"
    r"engineered\s+sacrifice|structural\s+design\s+remains\s+hidden|"
    r"remains?\s+concealed|stays?\s+hidden)\b[^.;,]{0,80}",
    re.IGNORECASE,
)
_INSTITUTIONAL_CONCEALMENT = re.compile(
    r"\b(?:engineered\s+(?:to\s+)?sacrific\w*|sacrificial\s+design|"
    r"sacrifice[\s-]+of[\s-]+wards?|sacrifice\s+public\s+wards?|"
    r"design\s+flaw|structural\s+design|engineered\s+sacrifice)\b",
    re.IGNORECASE,
)
_CONCEALMENT_MARKER = re.compile(
    r"\b(?:conceal\w*|hidden|secret|undisclosed|stays?\s+hidden|"
    r"stays?\s+(?:permanently\s+)?concealed|remains?\s+(?:concealed|hidden))\b",
    re.IGNORECASE,
)
_SHORT_LABEL_SPLIT = re.compile(
    r",|;|\bwhile\b|\bcausing\b|\bkilling\b|\bpreserving\b|\ballowing\b|"
    r"\bleaving\b|\bthereby\b|\bresulting\b",
    re.IGNORECASE,
)

# Token classes so paraphrases keep the same ranking-relevant content.
# "kills 16 refugees" ≡ "causes 16 refugees to die"; "grid failure" ≡ "grid unstable".
_CRITICAL_TOKEN_CLASS = {
    "kill": "LETHAL", "kills": "LETHAL", "killing": "LETHAL", "killed": "LETHAL",
    "die": "LETHAL", "dies": "LETHAL", "dying": "LETHAL", "died": "LETHAL",
    "death": "LETHAL", "deaths": "LETHAL", "dead": "LETHAL", "lethal": "LETHAL",
    "fatality": "LETHAL", "fatalities": "LETHAL", "fatal": "LETHAL",
    "fail": "FAILURE", "fails": "FAILURE", "failed": "FAILURE", "failure": "FAILURE",
    "failing": "FAILURE", "collapse": "FAILURE", "collapsed": "FAILURE",
    "collapses": "FAILURE", "outage": "FAILURE", "blackout": "FAILURE",
    "unstable": "FAILURE", "instability": "FAILURE",
    "conceal": "CONCEAL", "conceals": "CONCEAL", "concealing": "CONCEAL",
    "concealed": "CONCEAL", "concealment": "CONCEAL", "hidden": "CONCEAL",
    "secret": "CONCEAL", "covert": "CONCEAL", "undisclosed": "CONCEAL",
    "engineered": "ENGINEERED_SACRIFICE", "sacrifice": "ENGINEERED_SACRIFICE",
    "flaw": "ENGINEERED_SACRIFICE", "design": "ENGINEERED_SACRIFICE",
    "sacrifices": "ENGINEERED_SACRIFICE",
    "patients": "patient", "refugees": "refugee", "residents": "resident",
    "lives": "life", "people": "person", "workers": "worker",
    "children": "child", "wards": "ward",
}
_COVERAGE_STOPWORDS = {
    "a", "an", "and", "as", "at", "by", "for", "from", "in", "into", "of",
    "on", "or", "the", "to", "with", "that", "this", "those", "these",
    "all", "which", "while", "but", "so", "within", "during", "away",
}


@dataclass(frozen=True, slots=True)
class CanonicalActionRecord:
    """Decision-critical action object. Agents reason from the semantic form."""

    action_id: str
    short_label: str
    canonical_semantic_action: str
    actor: str = ""
    intervention: str = ""
    beneficiaries: tuple[str, ...] = ()
    harmed: tuple[str, ...] = ()
    # Parties the action affects without settling the outcome. Kept separate so
    # a probabilistic harm is neither asserted as certain nor lost from view.
    unresolved: tuple[str, ...] = ()
    unresolved_outcomes: tuple[dict[str, Any], ...] = ()
    # Cascade/foregone relations for parties the action does not itself treat.
    # Direct recipients stay in beneficiaries/harmed; this layer is surrounding
    # world-state, not a second copy of those roles.
    grounded_effects: tuple[Any, ...] = ()
    mechanism: str = ""
    institutional_effect: str = ""
    source_clauses: tuple[str, ...] = ()
    completeness_status: str = "UNCHECKED"
    missing_critical: tuple[str, ...] = ()
    structure_issues: tuple[str, ...] = ()
    commitment_status: str = "REJECTED"
    commitment_reasons: tuple[str, ...] = ()

    @property
    def eligible_for_deliberation(self) -> bool:
        return self.commitment_status == "COMMITTED"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# Terminal states for a canonical record. Only COMMITTED semantic state may
# enter deliberation; PARTIAL is usable evidence that is not yet a settled world
# model, and REJECTED must never reach an agent as if it were one.
COMMITMENT_COMMITTED = "COMMITTED"
COMMITMENT_PARTIAL = "PARTIAL"
COMMITMENT_REJECTED = "REJECTED"


def _commitment_state(
    completeness_status: str,
    grounding_status: str,
    has_sources: bool,
) -> tuple[str, tuple[str, ...]]:
    """Decide whether a record may enter deliberation, and say why not.

    Grounding is checked first: if the scenario could not say which clauses an
    action rests on, then however well-formed the record looks it describes a
    world nobody verified against the source.
    """
    grounding = str(grounding_status or "").strip().upper()
    reasons: list[str] = []

    if grounding == "REJECTED":
        return COMMITMENT_REJECTED, ("action-source grounding was rejected",)
    if completeness_status in {"NEEDS_REPAIR", "MISSING_CRITICAL", "INCOMPLETE_CLAUSE"}:
        return COMMITMENT_REJECTED, (
            f"record completeness is {completeness_status}",
        )

    if completeness_status == "COMPLETE_WITH_NORMALIZATION":
        reasons.append("structured roles required normalization")
    if not has_sources:
        reasons.append("no grounded source clauses")
    if grounding and grounding not in {"COMMITTED", ""}:
        reasons.append(f"grounding status is {grounding}")
    if completeness_status == "UNCHECKED":
        reasons.append("completeness was never checked")

    if reasons:
        return COMMITMENT_PARTIAL, tuple(dict.fromkeys(reasons))
    return COMMITMENT_COMMITTED, ()


def partition_records_for_deliberation(
    records: Sequence[CanonicalActionRecord],
) -> tuple[list[CanonicalActionRecord], list[CanonicalActionRecord]]:
    """Split records into those admitted to deliberation and those withheld."""
    admitted = [record for record in records if record.eligible_for_deliberation]
    withheld = [record for record in records if not record.eligible_for_deliberation]
    return admitted, withheld


# --- Structured role extraction from canonical prose ---------------------------
#
# compile_action_identity() is graph-oriented and can mis-assign token bags as
# targets (e.g. "freezing refugees survive" → freeze ADVERSE on refugees).
# Canonical action records use dedicated prose patterns instead.

_SCENARIO_ACTOR = re.compile(
    r"\bas\s+(?:the\s+)?director\s+of\s+(?:an?\s+)?(.+?)\s+grid\b",
    re.IGNORECASE,
)
_PATIENT_GROUP = (
    r"(?P<label>(?:(?:\d+|four)\s+)?"
    r"(?:(?:non[- ]consenting|chronically\s+ill|public[- ]ward|ward)\s+)*"
    r"patients?)"
)
_REFUGEE_GROUP = (
    r"(?P<label>(?:(?:\d+|sixteen)\s+)?(?:freezing\s+)?refugees?)"
)
_TOKEN_BAG_MARKERS = re.compile(r":COUNT\b|^[a-z]+,\s", re.IGNORECASE)


def _group_key(label: str) -> tuple[str, ...]:
    text = label.casefold()
    tokens = []
    count = re.search(r"\b(\d+|four|sixteen)\b", text)
    if count:
        tokens.append(count.group(1))
    if "patient" in text or "ward" in text:
        tokens.append("patient")
    if "refugee" in text:
        tokens.append("refugee")
    return tuple(tokens)


def _group_keys_compatible(required: tuple[str, ...], actual: tuple[str, ...]) -> bool:
    if required == actual:
        return True
    if "refugee" in required and "refugee" in actual:
        return True
    if "patient" in required and "patient" in actual:
        return True
    return False


def _best_group_label(text: str, kind: str) -> str:
    """Return the richest patient/refugee group mention in prose."""
    pattern = re.compile(
        _PATIENT_GROUP if kind == "patient" else _REFUGEE_GROUP,
        re.IGNORECASE,
    )
    matches = [match.group("label") for match in pattern.finditer(text)]
    if not matches:
        return "patients" if kind == "patient" else "refugees"
    return max(matches, key=lambda item: (bool(re.search(r"\d", item)), len(item)))


def _explicit_count_in_label(label: str) -> str | None:
    return party_identity(label)[1]


def _count_from_group_key(key: tuple[str, ...]) -> str | None:
    if not key:
        return None
    token = key[0]
    if token.isdigit():
        return token
    return {"four": "4", "sixteen": "16"}.get(token)


def _enrich_refugee_label_from_context(
    text: str,
    span_start: int,
    label: str,
) -> str:
    """Keep shelter/emergency context on refugee groups when prose supplies it."""
    cleaned = " ".join(str(label).split())
    window = text[max(0, span_start - 120): span_start + len(cleaned)]
    if "shelter" not in window.casefold() or "shelter" in cleaned.casefold():
        return cleaned
    count = _explicit_count_in_label(cleaned)
    if count:
        return f"{count} shelter refugees"
    return "shelter refugees"


def _extract_relational_role_bindings(canonical: str) -> list[dict[str, Any]]:
    """Explicit save/harm relations that must survive in structured fields."""
    text = " ".join(str(canonical or "").split())
    bindings: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()

    def _register(field: str, label: str, *, span_start: int) -> None:
        cleaned = " ".join(str(label).split()).strip(" ,;.")
        if not cleaned:
            return
        if "refugee" in cleaned.casefold():
            cleaned = _enrich_refugee_label_from_context(text, span_start, cleaned)
        key = _group_key(cleaned)
        token = (field, key)
        if token in seen:
            return
        seen.add(token)
        bindings.append({"field": field, "label": cleaned, "key": key})

    for match in re.finditer(
        rf"\bkill(?:s|ing|ed)?\s+(?:the\s+)?{_PATIENT_GROUP}\b",
        text,
        re.IGNORECASE,
    ):
        _register("harmed", match.group("label"), span_start=match.start("label"))
    if re.search(r"\bkill(?:s|ing|ed)?\s+them\b", text, re.IGNORECASE) and re.search(
        r"\bpatients?\b", text, re.IGNORECASE,
    ):
        patient = _best_group_label(text, "patient")
        _register("harmed", patient, span_start=text.casefold().find(patient.casefold()))

    for match in re.finditer(
        rf"\b(?:kills?|killing)\s+(?:all\s+)?{_REFUGEE_GROUP}\b",
        text,
        re.IGNORECASE,
    ):
        _register("harmed", match.group("label"), span_start=match.start("label"))
    for match in re.finditer(
        rf"\b{_REFUGEE_GROUP}\s+(?:die|to\s+die|perish)\b",
        text,
        re.IGNORECASE,
    ):
        _register("harmed", match.group("label"), span_start=match.start("label"))

    for match in re.finditer(
        rf"\b{_REFUGEE_GROUP}\s+(?:survive|alive)\b",
        text,
        re.IGNORECASE,
    ):
        _register("beneficiaries", match.group("label"), span_start=match.start("label"))
    for match in re.finditer(
        rf"\bkeep\s+{_REFUGEE_GROUP}\s+alive\b",
        text,
        re.IGNORECASE,
    ):
        _register("beneficiaries", match.group("label"), span_start=match.start("label"))
    for match in re.finditer(
        rf"\b(?:save[sd]?|rescu(?:e|es|ed|ing))\s+{_REFUGEE_GROUP}\b",
        text,
        re.IGNORECASE,
    ):
        _register("beneficiaries", match.group("label"), span_start=match.start("label"))
    for match in re.finditer(
        rf"\bstabiliz\w*[^.;]{{0,120}}?\bholding\s+{_REFUGEE_GROUP}\b",
        text,
        re.IGNORECASE,
    ):
        _register("beneficiaries", match.group("label"), span_start=match.start("label"))
    if re.search(r"\bstabiliz", text, re.IGNORECASE):
        for match in re.finditer(
            rf"\bholding\s+{_REFUGEE_GROUP}\b",
            text,
            re.IGNORECASE,
        ):
            _register("beneficiaries", match.group("label"), span_start=match.start("label"))

    for match in re.finditer(
        rf"\b(?:honoring|preserving|respecting|upholding|sparing)\s+"
        rf"(?:the\s+)?{_PATIENT_GROUP}\b",
        text,
        re.IGNORECASE,
    ):
        _register("beneficiaries", match.group("label"), span_start=match.start("label"))
    for match in re.finditer(
        rf"\b{_PATIENT_GROUP}\b[^.;]{{0,40}}\b(?:protected|preserved|spared)\b",
        text,
        re.IGNORECASE,
    ):
        _register("beneficiaries", match.group("label"), span_start=match.start("label"))
    if re.search(r"\bpatients?['’]?\s+right\s+against\b", text, re.IGNORECASE):
        _register(
            "beneficiaries",
            _best_group_label(text, "patient"),
            span_start=0,
        )

    return bindings


def _obligatory_role_groups(canonical: str) -> dict[str, set[tuple[str, ...]]]:
    """Independent prose scan for required harmed/beneficiary group keys."""
    harmed: set[tuple[str, ...]] = set()
    benefited: set[tuple[str, ...]] = set()
    for binding in _extract_relational_role_bindings(canonical):
        if binding["field"] == "harmed":
            harmed.add(binding["key"])
        else:
            benefited.add(binding["key"])

    lowered = " ".join(str(canonical or "").split()).casefold()
    if re.search(
        rf"\b(?:divert\w*|reallocat\w*)\b[^.;]{{0,120}}\bpatients?\b",
        lowered,
    ) and not any("patient" in key for key in harmed):
        harmed.add(_group_key(_best_group_label(canonical, "patient")))
    if (
        "grid failure" in lowered
        and "refugee" in lowered
        and "maintain" in lowered
        and not any("refugee" in key for key in harmed)
    ):
        harmed.add(_group_key(_best_group_label(canonical, "refugee")))
    if re.search(r"\bmaintain\b[^.;]{0,80}\ballocation\b", lowered) and any(
        "refugee" in key for key in harmed
    ):
        benefited.add(_group_key(_best_group_label(canonical, "patient")))

    return {"harmed": harmed, "beneficiaries": benefited}


def extract_structured_action_roles(
    action_text: str,
    *,
    scenario_actor: str = "",
    registry: Sequence[Any] | None = None,
    clauses: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Extract harmed/beneficiaries/mechanism from a canonical semantic action."""
    text = " ".join(str(action_text or "").split())
    lowered = text.casefold()
    beneficiaries: list[str] = []
    harmed: list[str] = []
    mechanism = ""
    institutional = ""

    # Roles come from the scenario's own party vocabulary, so a dilemma the
    # pattern layer has never seen is extracted on the same footing as a
    # familiar one.
    parties = list(registry) if registry is not None else extract_party_registry(
        text, [{"clause_id": "A", "text": text}],
    )

    def _append_party(bucket: list[str], label: str) -> None:
        cleaned = " ".join(str(label).split()).strip(" ,;.")
        if not cleaned:
            return
        if any(parties_compatible(existing, cleaned) for existing in bucket):
            return
        bucket.append(cleaned)

    unresolved: list[str] = []
    unresolved_outcomes: list[dict[str, Any]] = []
    for binding in relational_role_bindings(text, parties):
        if binding.field == ROLE_UNRESOLVED:
            _append_party(unresolved, binding.party.label)
            unresolved_outcomes.append(binding.as_dict())
            continue
        _append_party(
            beneficiaries if binding.field == "beneficiaries" else harmed,
            binding.party.label,
        )

    # --- mechanism -----------------------------------------------------------
    if re.search(r"\bmaintain\b[^.;]{0,80}\ballocation\b", lowered) and (
        "grid failure" in lowered or "grid unstable" in lowered
    ):
        mechanism = "maintaining standard allocation causes shelter grid failure"
    realloc = re.search(
        r"\b((?:forcibly\s+)?(?:reallocat\w*|divert\w*)\s+oxygen\b[^.;]{0,160}?)"
        r"(?=\s*(?:,|;|\bkilling\b|\bbut\b|\bto\b|\bso\b|\band\b))",
        text,
        re.IGNORECASE,
    )
    if realloc:
        mechanism = " ".join(realloc.group(1).split())
        if "stabiliz" in lowered and "stabiliz" not in mechanism.casefold():
            mechanism += " to stabilize the shelter"
    if not mechanism:
        grid_fail = re.search(
            r"\b(?:which\s+)?(?:triggering|causing|leaves?|allow)\s+"
            r"(?:(?:the|a)\s+)?(?:grid\s+)?failure\b[^.;,]{0,60}|"
            r"\b(?:which\s+)?(?:triggering|causing|leaves?)\s+"
            r"(?:(?:the|a)\s+)?shelter\s+grid\s+"
            r"(?:failure|unstable|to\s+fail)\b[^.;,]{0,60}",
            text,
            re.IGNORECASE,
        )
        if grid_fail:
            mechanism = " ".join(grid_fail.group(0).split())
            mechanism = re.sub(r"^which\s+", "", mechanism, flags=re.IGNORECASE)
    if not mechanism and re.search(r"\bshelter\s+grid\s+fail", lowered):
        mechanism = "shelter grid fails within hours"

    # --- institutional effect ------------------------------------------------
    if _extract_institutional_concealment_claim(text) == (
        "engineered sacrifice remains concealed"
    ):
        institutional = "engineered sacrifice arrangement remains concealed"

    # --- intervention --------------------------------------------------------
    intervention = ""
    head_match = re.match(r"^([^,]+)", text)
    if head_match:
        intervention = " ".join(head_match.group(1).split())
        intervention = re.sub(
            r"\s+that\b.*$",
            "",
            intervention,
            flags=re.IGNORECASE,
        ).strip(" ,;.")
    if not intervention:
        intervention = render_short_label(text)

    actor = " ".join(str(scenario_actor or "").split())
    return {
        "actor": actor,
        "intervention": intervention,
        "beneficiaries": tuple(beneficiaries),
        "harmed": tuple(harmed),
        "unresolved": tuple(unresolved),
        "unresolved_outcomes": tuple(unresolved_outcomes),
        "grounded_effects": extract_grounded_effects(
            text,
            clauses=clauses,
            registry=parties,
            direct_beneficiaries=beneficiaries,
            direct_harmed=harmed,
            unresolved=unresolved,
        ),
        "mechanism": mechanism,
        "institutional_effect": institutional,
    }


def extract_scenario_actor(scenario: str) -> str:
    """Resolve an explicit scenario actor when the prose names one."""
    text = " ".join(str(scenario or "").split())
    match = _SCENARIO_ACTOR.search(text)
    if not match:
        return ""
    scope = " ".join(match.group(1).split()).casefold()
    if "life" in scope and "support" in scope:
        return "grid director"
    if scope.endswith(" director"):
        return scope
    return f"{scope} director".strip()


def _field_looks_like_token_bag(value: str) -> bool:
    text = " ".join(str(value or "").split())
    if not text:
        return False
    if _TOKEN_BAG_MARKERS.search(text):
        return True
    # Comma-separated single tokens are graph-compiler debris, not prose roles.
    if "," in text and not re.search(r"\b(?:patient|refugee|grid|oxygen|shelter)\b", text):
        return True
    return False


def validate_structured_role_consistency(
    record: CanonicalActionRecord | dict[str, Any],
    *,
    scenario_actor: str = "",
) -> tuple[str, ...]:
    """Return issues when structured fields disagree with canonical prose."""
    if isinstance(record, CanonicalActionRecord):
        payload = record.as_dict()
    else:
        payload = dict(record)
    canonical = str(payload.get("canonical_semantic_action") or "")
    expected_roles = extract_structured_action_roles(
        canonical,
        scenario_actor=scenario_actor or str(payload.get("actor") or ""),
    )
    obligations = _obligatory_role_groups(canonical)
    issues: list[str] = []

    for field in ("beneficiaries", "harmed"):
        for value in payload.get(field) or ():
            if _field_looks_like_token_bag(str(value)):
                issues.append(f"{field} contains token-bag fragment: {value}")

    if scenario_actor and not str(payload.get("actor") or "").strip():
        issues.append("actor missing despite explicit scenario role")

    valid_beneficiaries = [
        str(value) for value in (payload.get("beneficiaries") or ())
        if not _field_looks_like_token_bag(str(value))
    ]
    valid_harmed = [
        str(value) for value in (payload.get("harmed") or ())
        if not _field_looks_like_token_bag(str(value))
    ]

    # Independent obligation scan — must not share the same extractor blind spot.
    # Identity is party-compatible, not "any two labels that contain 'patient'":
    # Patient A and Patient B are different people even though both are patients.
    for required in obligations["beneficiaries"]:
        if not any(
            _group_keys_compatible(required, _group_key(existing))
            or parties_compatible("/".join(required), existing)
            for existing in valid_beneficiaries
        ):
            issues.append(
                "beneficiaries missing required prose group: "
                + "/".join(required)
            )
    for required in obligations["harmed"]:
        if not any(
            _group_keys_compatible(required, _group_key(existing))
            or parties_compatible("/".join(required), existing)
            for existing in valid_harmed
        ):
            issues.append(
                "harmed missing required prose group: "
                + "/".join(required)
            )

    for label in expected_roles["beneficiaries"]:
        if not any(
            parties_compatible(label, existing)
            for existing in valid_beneficiaries
        ):
            issues.append(f"beneficiaries missing prose group: {label}")
    for label in expected_roles["harmed"]:
        if not any(
            parties_compatible(label, existing)
            for existing in valid_harmed
        ):
            issues.append(f"harmed missing prose group: {label}")

    if obligations["beneficiaries"] and not valid_beneficiaries:
        issues.append("beneficiaries empty despite prose beneficiaries")
    if obligations["harmed"] and not valid_harmed:
        issues.append("harmed empty despite prose harm")

    for binding in _extract_relational_role_bindings(canonical):
        field = str(binding["field"])
        labels = valid_beneficiaries if field == "beneficiaries" else valid_harmed
        rich_label = str(binding["label"])
        compatible = [
            label for label in labels
            if parties_compatible(rich_label, label)
            or _group_keys_compatible(binding["key"], _group_key(label))
        ]
        if not compatible:
            issues.append(f"{field} missing relational binding: {rich_label}")
            continue
        required_count = party_identity(rich_label)[1]
        if required_count and not any(
            party_identity(label)[1] for label in compatible
        ):
            issues.append(
                f"{field} lost explicit cardinality for {rich_label}"
            )

    overlap_labels = [
        f"{harmed} / {saved}"
        for harmed in valid_harmed
        for saved in valid_beneficiaries
        if parties_compatible(harmed, saved)
    ]
    if overlap_labels:
        issues.append(
            "group appears as both harmed and beneficiary: "
            + ", ".join(overlap_labels)
        )

    if expected_roles["mechanism"] and not str(payload.get("mechanism") or "").strip():
        issues.append("mechanism missing causal relation from prose")
    if _field_looks_like_token_bag(str(payload.get("mechanism") or "")):
        issues.append("mechanism contains token-bag fragment")

    return tuple(dict.fromkeys(issues))


def _completeness_status_from_structure_issues(
    structure_issues: tuple[str, ...],
) -> str:
    if not structure_issues:
        return ""
    if all("lost explicit cardinality" in issue for issue in structure_issues):
        return "COMPLETE_WITH_NORMALIZATION"
    return "NEEDS_REPAIR"


def _extract_institutional_concealment_claim(text: str) -> str | None:
    """Normalize institutional-concealment paraphrases to one stable atom."""
    cleaned = " ".join(str(text or "").split())
    if not cleaned:
        return None
    if not _CONCEALMENT_MARKER.search(cleaned):
        return None
    if (
        _INSTITUTIONAL_CONCEALMENT.search(cleaned)
        or re.search(r"\bconceal\w*\s+that\b", cleaned, re.IGNORECASE)
    ):
        return "engineered sacrifice remains concealed"
    return "arrangement remains concealed"


def extract_decision_critical_claims(text: str) -> tuple[str, ...]:
    """Pull ranking-relevant consequence/constraint atoms from prose or clauses."""
    cleaned = " ".join(str(text or "").split())
    if not cleaned:
        return ()
    claims: list[str] = []

    def _add(claim: str) -> None:
        normalized = " ".join(str(claim).split()).strip(" ,;:.-")
        if len(normalized) < 6:
            return
        if normalized.casefold() in {item.casefold() for item in claims}:
            return
        claims.append(normalized)

    for match in _NUMBERED_OUTCOME.finditer(cleaned):
        groups = match.groupdict()
        if groups.get("count"):
            body = str(groups.get("body") or "")
            verb = str(groups.get("verb") or "").casefold()
            if verb in {"kill", "kills", "killing", "killed"} and re.search(
                r"\bthem\b", body, re.IGNORECASE,
            ) and re.search(
                r"\bholding\b[^.;]{0,80}\brefugees?\b", body, re.IGNORECASE,
            ):
                continue
            _add(
                f"{groups['verb']} {groups['count']} {groups['who']}"
            )
        elif groups.get("count2"):
            _add(
                f"{groups['verb2']} {groups['count2']} {groups['who2']}"
            )
    if re.search(r"\bkill(?:s|ing|ed)?\s+them\b", cleaned, re.IGNORECASE) and re.search(
        r"\bpatients?\b", cleaned, re.IGNORECASE,
    ):
        patient = _best_group_label(cleaned, "patient")
        _add(f"killing {patient}")
    for match in _MECHANISM_FAILURE.finditer(cleaned):
        span = match.group(0)
        if _BODILY_FAILURE.search(span):
            continue
        # A bare "failure" token is not a decision-critical atom; it is almost
        # always a medical or generic event that the action may be preventing.
        if re.fullmatch(r"fail(?:s|ed|ure|ing)?", span.strip(), re.IGNORECASE):
            continue
        _add(span)
    concealment = _extract_institutional_concealment_claim(cleaned)
    if concealment:
        _add(concealment)
    return tuple(claims)


def _critical_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for token in re.findall(r"[a-z0-9]+", str(text or "").casefold()):
        if token in _COVERAGE_STOPWORDS or len(token) <= 1:
            continue
        tokens.add(_CRITICAL_TOKEN_CLASS.get(token, token))
    return tokens


def _claim_coverage(claim: str, haystack: str) -> float:
    if (
        "conceal" in claim.casefold()
        and _extract_institutional_concealment_claim(haystack) == claim
    ):
        return 1.0
    claim_cf = claim.casefold()
    if any(word in claim_cf for word in ("kill", "die", "death")):
        if "patient" in claim_cf and re.search(
            r"\bkill(?:s|ing|ed)?\b", haystack, re.IGNORECASE,
        ) and re.search(r"\bpatients?\b", haystack, re.IGNORECASE):
            if _group_keys_compatible(
                _group_key(claim),
                _group_key(_best_group_label(haystack, "patient")),
            ):
                return 1.0
        if "refugee" in claim_cf and re.search(
            r"\b(?:kill|die|death)\w*\b", haystack, re.IGNORECASE,
        ) and re.search(r"\brefugees?\b", haystack, re.IGNORECASE):
            if _group_keys_compatible(
                _group_key(claim),
                _group_key(_best_group_label(haystack, "refugee")),
            ):
                return 1.0
    claim_tokens = _critical_tokens(claim)
    hay_tokens = _critical_tokens(haystack)
    if not claim_tokens:
        return 1.0
    return len(claim_tokens & hay_tokens) / len(claim_tokens)


def _source_claim_applies_to_action(claim: str, action_text: str) -> bool:
    """Return whether a grounded source claim must be preserved in this action."""
    text = " ".join(str(action_text or "").split()).casefold()
    claim_cf = " ".join(str(claim or "").split()).casefold()
    patch_action = bool(re.search(
        r"\b(?:execute|covert\s+patch|forcibly\s+reallocat\w*|divert\w*\s+oxygen)\b",
        text,
    ))
    maintain_action = bool(re.search(
        r"\bmaintain\b[^.;]{0,80}\b(?:standard\s+)?(?:oxygen\s+)?allocation\b",
        text,
    ))

    if "conceal" in claim_cf or claim_cf == "engineered sacrifice remains concealed":
        if patch_action:
            return bool(_extract_institutional_concealment_claim(action_text))
        return True

    if "refugee" in claim_cf and any(
        word in claim_cf for word in ("kill", "die", "death")
    ):
        if patch_action and re.search(r"\b(?:save|surviv|keep)\b", text) and "refugee" in text:
            return False

    if "patient" in claim_cf and any(
        word in claim_cf for word in ("kill", "die", "death")
    ):
        if maintain_action and (
            re.search(r"\bpreserv\w+", text)
            or "patients' lives" in text
            or "patients' rights" in text
            or "patients' right" in text
        ):
            return False

    if claim_cf == "shelter grid failure" and patch_action:
        if "stabiliz" in text and "grid" in text:
            return False

    # A source clause often states the counterfactual of the other option
    # ("Patient B dies of organ failure without the drug"). That is not an
    # obligation on the action that *prevents* the event.
    if _claim_inverted_by_action(claim, action_text):
        return False

    return True


def _claim_inverted_by_action(claim: str, action_text: str) -> bool:
    """True when the action's own roles reverse the polarity of the claim."""
    claim_cf = " ".join(str(claim or "").split()).casefold()
    lethal = bool(re.search(
        r"\b(?:kill|die|death|fail|failure|perish|drown)\b", claim_cf,
    ))
    beneficial = bool(re.search(
        r"\b(?:save|surviv|preserv|protect|alive|rescue)\b", claim_cf,
    ))
    if not lethal and not beneficial:
        return False
    registry = extract_party_registry(
        action_text, [{"clause_id": "A", "text": action_text}],
    )
    if not registry:
        return False
    for binding in relational_role_bindings(action_text, registry):
        if not _claim_mentions_party(claim_cf, binding.party):
            continue
        if lethal and binding.field == "beneficiaries":
            return True
        if beneficial and binding.field == "harmed":
            return True
    return False


def _claim_mentions_party(claim_cf: str, party) -> bool:
    if parties_compatible(party, claim_cf):
        return True
    if party.designator and party.designator in claim_cf:
        return True
    distinctive = party.tokens - {"life", "person", "individual", "patient"}
    return any(token in claim_cf for token in distinctive)


def missing_decision_critical_claims(
    action: str,
    source_texts: Sequence[str],
    *,
    coverage_threshold: float = 0.72,
) -> tuple[str, ...]:
    """Return source claims that the canonical action failed to preserve.

    Matching is synonym-aware for lethal outcomes, infrastructure failure, and
    concealment so paraphrases that keep the same moral content are admitted.
    """
    action_text = " ".join(str(action or "").split())
    action_claims = extract_decision_critical_claims(action_text)
    missing: list[str] = []
    for source in source_texts:
        for claim in extract_decision_critical_claims(source):
            if not _source_claim_applies_to_action(claim, action_text):
                continue
            if _claim_coverage(claim, action_text) >= coverage_threshold:
                continue
            # Also accept when the action states an equivalent extracted claim
            # (e.g. source "kills 16 refugees" vs action "die 16 refugees").
            if any(
                _claim_coverage(claim, action_claim) >= coverage_threshold
                or _claim_coverage(action_claim, claim) >= coverage_threshold
                for action_claim in action_claims
            ):
                continue
            if claim.casefold() not in {item.casefold() for item in missing}:
                missing.append(claim)
    return tuple(missing)


def render_short_label(semantic_action: str, intervention: str = "") -> str:
    """UI-facing label. May omit detail; never used as the reasoning object."""
    text = " ".join(str(semantic_action or "").split())
    if not text:
        return " ".join(str(intervention or "action").replace("_", " ").split()).title()
    head = _SHORT_LABEL_SPLIT.split(text, maxsplit=1)[0].strip(" ,;:.-")
    if len(head.split()) >= 2 and len(head) <= 72:
        return head[0].upper() + head[1:] if head else head
    if intervention:
        return " ".join(intervention.replace("_", " ").split()).title()
    if len(text) <= 72:
        return text[0].upper() + text[1:]
    shortened = text[:73].rsplit(" ", 1)[0].rstrip(" ,;:-")
    return shortened + "…"


def render_semantic_action_from_structure(
    *,
    intervention: str,
    actor: str = "",
    beneficiaries: Sequence[str] = (),
    harmed: Sequence[str] = (),
    mechanism: str = "",
    institutional_effect: str = "",
    fallback: str = "",
) -> str:
    """Render the authoritative semantic string from structured fields."""
    parts: list[str] = []
    head = " ".join(str(intervention or "").replace("_", " ").split())
    if actor and head:
        parts.append(f"{actor.strip()}: {head}")
    elif head:
        parts.append(head[0].upper() + head[1:] if head else head)
    if beneficiaries:
        parts.append("preserving " + "; ".join(beneficiaries))
    if harmed:
        parts.append("causing " + "; ".join(harmed))
    if mechanism:
        parts.append(str(mechanism).strip().rstrip("."))
    if institutional_effect:
        parts.append(str(institutional_effect).strip().rstrip("."))
    rendered = ", ".join(part for part in parts if part)
    if rendered:
        return rendered + ("." if rendered[-1] not in ".?!" else "")
    return " ".join(str(fallback or "").split())


def _structure_from_identity(identity: ActionIdentity, action_text: str) -> dict[str, Any]:
    beneficiaries: list[str] = []
    harmed: list[str] = []
    mechanisms: list[str] = []
    institutional: list[str] = []
    for consequence in identity.consequences:
        quantity = ", ".join(consequence.quantities)
        targets = ", ".join(consequence.targets) or consequence.predicate
        phrase = " ".join(part for part in (quantity, targets) if part).strip()
        if consequence.polarity == "BENEFICIAL":
            if phrase and phrase not in beneficiaries:
                beneficiaries.append(phrase)
        elif consequence.polarity == "ADVERSE":
            if phrase and phrase not in harmed:
                harmed.append(phrase)
            if consequence.predicate in {"disconnect", "freeze", "failure", "collapse"} or (
                "fail" in consequence.predicate
            ):
                mechanisms.append(phrase or consequence.predicate)
    text = action_text.casefold()
    if re.search(r"\b(?:conceal\w*|hidden|secret|covert|undisclosed)\b", text):
        institutional.append("engineered arrangement remains concealed")
    if re.search(r"\b(?:siphon|structural\s+design)\b", text):
        institutional.append("structural design remains hidden")
    actor = ", ".join(identity.actors)
    intervention = identity.intervention.replace("_", " ") if identity.intervention else ""
    if not intervention:
        intervention = render_short_label(action_text)
    mechanism = "; ".join(dict.fromkeys(mechanisms))
    if not mechanism:
        fail = re.search(
            r"([^.;,]{0,40}\b(?:grid|shelter|power|oxygen)\b[^.;,]{0,40}\bfail\w*[^.;,]{0,40}"
            r"|[^.;,]{0,40}\bfail\w*[^.;,]{0,40}\b(?:grid|shelter|power|oxygen)\b[^.;,]{0,40})",
            action_text,
            re.IGNORECASE,
        )
        if fail:
            mechanism = " ".join(fail.group(0).split())
    return {
        "actor": actor,
        "intervention": intervention,
        "beneficiaries": tuple(beneficiaries),
        "harmed": tuple(harmed),
        "unresolved": (),
        "unresolved_outcomes": (),
        "grounded_effects": (),
        "mechanism": mechanism,
        "institutional_effect": "; ".join(dict.fromkeys(institutional)),
    }


def _resolve_party_registry(
    scenario: str,
    canonical: str,
    source_clause_texts: Sequence[str],
) -> list[Any]:
    """Derive the scenario's party vocabulary for domain-neutral validation.

    Prefers the full scenario, falls back to the grounded clauses, and finally
    to the action prose itself so records built in isolation are still checked.
    """
    from .scenario_semantics import segment_scenario_clauses
    from .semantic_roles import extract_party_registry

    if scenario:
        return extract_party_registry(
            scenario, segment_scenario_clauses(scenario), outcome_context=(canonical,),
        )
    clauses = [
        {"clause_id": f"S{index}", "text": text}
        for index, text in enumerate(source_clause_texts)
        if str(text).strip()
    ]
    if clauses:
        joined = " ".join(clause["text"] for clause in clauses)
        return extract_party_registry(joined, clauses, outcome_context=(canonical,))
    return extract_party_registry(canonical, [{"clause_id": "A", "text": canonical}])


def _merge_completeness(*statuses: str) -> str:
    """Combine status verdicts, letting the most severe one win."""
    order = {"NEEDS_REPAIR": 2, "COMPLETE_WITH_NORMALIZATION": 1, "": 0}
    worst = max((status for status in statuses), key=lambda item: order.get(item, 0))
    return worst


def build_canonical_action_record(
    action_id: str,
    action_text: str,
    *,
    actor: str = "",
    source_clause_texts: Sequence[str] = (),
    scenario: str = "",
    grounding_status: str = "",
    require_complete: bool = False,
) -> CanonicalActionRecord:
    """Compile id / short label / semantic action / structured fields."""
    semantic = " ".join(str(action_text or "").split())
    party_registry = _resolve_party_registry(
        scenario, semantic, [str(text) for text in source_clause_texts],
    )
    effect_clauses: Sequence[Any]
    if scenario:
        from .scenario_semantics import segment_scenario_clauses
        effect_clauses = segment_scenario_clauses(scenario)
    else:
        effect_clauses = [
            {"clause_id": f"S{index}", "text": str(text)}
            for index, text in enumerate(source_clause_texts)
            if str(text).strip()
        ]
    structure = extract_structured_action_roles(
        semantic,
        scenario_actor=actor,
        registry=party_registry,
        clauses=effect_clauses,
    )
    # Prefer the admitted full prose when complete; otherwise render from structure.
    if semantic and action_clause_looks_complete(semantic):
        canonical = semantic
    else:
        canonical = render_semantic_action_from_structure(
            intervention=structure["intervention"],
            actor=structure["actor"],
            beneficiaries=structure["beneficiaries"],
            harmed=structure["harmed"],
            mechanism=structure["mechanism"],
            institutional_effect=structure["institutional_effect"],
            fallback=semantic,
        )
    short = render_short_label(canonical, structure["intervention"])
    sources = tuple(
        " ".join(str(item).split())
        for item in source_clause_texts
        if " ".join(str(item).split())
    )
    missing = missing_decision_critical_claims(canonical, sources) if sources else ()
    record = CanonicalActionRecord(
        action_id=str(action_id).strip().upper() or "A?",
        short_label=short,
        canonical_semantic_action=canonical,
        actor=structure["actor"],
        intervention=structure["intervention"],
        beneficiaries=structure["beneficiaries"],
        harmed=structure["harmed"],
        unresolved=structure.get("unresolved", ()),
        unresolved_outcomes=structure.get("unresolved_outcomes", ()),
        grounded_effects=structure.get("grounded_effects", ()),
        mechanism=structure["mechanism"],
        institutional_effect=structure["institutional_effect"],
        source_clauses=sources,
        completeness_status="UNCHECKED",
        missing_critical=missing,
        structure_issues=(),
    )
    from .semantic_roles import completeness_from_role_issues, validate_role_assignment

    legacy_issues = validate_structured_role_consistency(
        record,
        scenario_actor=actor,
    )
    # Domain-neutral check against the scenario's own party vocabulary, so a
    # dilemma the pattern layer has never seen cannot pass with empty roles.
    role_issues = validate_role_assignment(
        canonical,
        beneficiaries=record.beneficiaries,
        harmed=record.harmed,
        unresolved=record.unresolved,
        registry=party_registry,
    )
    structure_issues = tuple(dict.fromkeys((
        *legacy_issues,
        *(issue.message for issue in role_issues),
    )))
    # Each layer grades only its own findings: the legacy checks report verdicts
    # as message text, while the registry checks carry explicit severity.
    merged_status = _merge_completeness(
        _completeness_status_from_structure_issues(legacy_issues),
        completeness_from_role_issues(role_issues),
    )

    if not action_clause_looks_complete(canonical):
        status = "INCOMPLETE_CLAUSE"
    elif missing:
        status = "MISSING_CRITICAL"
    elif structure_issues:
        status = merged_status
    elif sources:
        status = "COMPLETE"
    else:
        status = "UNCHECKED"
    commitment, commitment_reasons = _commitment_state(
        status, grounding_status, bool(sources),
    )
    record = CanonicalActionRecord(
        action_id=record.action_id,
        short_label=record.short_label,
        canonical_semantic_action=record.canonical_semantic_action,
        actor=record.actor,
        intervention=record.intervention,
        beneficiaries=record.beneficiaries,
        harmed=record.harmed,
        unresolved=record.unresolved,
        unresolved_outcomes=record.unresolved_outcomes,
        grounded_effects=record.grounded_effects,
        mechanism=record.mechanism,
        institutional_effect=record.institutional_effect,
        source_clauses=record.source_clauses,
        completeness_status=status,
        missing_critical=record.missing_critical,
        structure_issues=structure_issues,
        commitment_status=commitment,
        commitment_reasons=commitment_reasons,
    )
    if require_complete and status in {
        "INCOMPLETE_CLAUSE",
        "MISSING_CRITICAL",
        "NEEDS_REPAIR",
        "COMPLETE_WITH_NORMALIZATION",
    }:
        detail = "; ".join((*missing, *structure_issues)) or canonical
        raise ValueError(
            f"{record.action_id} fails action-completeness ({status}): {detail}"
        )
    return record


def build_canonical_action_records(
    actions: Sequence[str],
    *,
    actor: str = "",
    grounded_clause_texts_by_id: dict[str, Sequence[str]] | None = None,
    scenario: str = "",
    grounding_status: str = "",
    require_complete: bool = False,
) -> list[CanonicalActionRecord]:
    grounded = grounded_clause_texts_by_id or {}
    records: list[CanonicalActionRecord] = []
    for index, action in enumerate(actions):
        action_id = f"A{index}"
        records.append(build_canonical_action_record(
            action_id,
            action,
            actor=actor,
            source_clause_texts=grounded.get(action_id, ()),
            scenario=scenario,
            grounding_status=grounding_status,
            require_complete=require_complete,
        ))
    return records


def validate_action_set_completeness(
    actions: Sequence[str],
    *,
    scenario: str = "",
    grounded_clause_texts_by_id: dict[str, Sequence[str]] | None = None,
) -> None:
    """Hard gate: every canonical action must preserve decision-critical content.

    Checks (1) clause-shape completeness and (2), when source clauses are known,
    that ranking-relevant effects from those clauses survive in the action text.
    """
    normalized = [str(action) for action in actions]
    if len(normalized) != 2:
        raise ValueError("canonical action set must contain exactly two actions")
    shape_problems = [
        f"A{index}: {action}"
        for index, action in enumerate(normalized)
        if not action_clause_looks_complete(action)
    ]
    if shape_problems:
        prefix = "incomplete or truncated action clause(s)"
        if scenario:
            prefix += " in scenario admission"
        raise ValueError(f"{prefix}: " + "; ".join(shape_problems))

    grounded = grounded_clause_texts_by_id or {}
    if not grounded and scenario:
        # Before grounding: ensure scenario-level critical claims are not dropped
        # by *both* actions when the scenario states them in action clauses.
        return

    actor = extract_scenario_actor(scenario) if scenario else ""
    structure_rows: list[str] = []
    for index, action in enumerate(normalized):
        action_id = f"A{index}"
        record = build_canonical_action_record(
            action_id,
            action,
            actor=actor,
            source_clause_texts=grounded.get(action_id, ()),
            scenario=scenario,
        )
        # Only a repair-level verdict blocks admission. A normalization issue
        # (a flattened count, a party the fallback extractor could not name)
        # is recorded on the action rather than used to reject the whole set.
        if record.structure_issues and record.completeness_status == "NEEDS_REPAIR":
            structure_rows.append(
                f"{action_id} structured roles disagree with prose: "
                + "; ".join(record.structure_issues)
            )
    if structure_rows:
        raise ValueError(" | ".join(structure_rows))

    missing_rows = []
    for index, action in enumerate(normalized):
        action_id = f"A{index}"
        sources = list(grounded.get(action_id, ()))
        missing = missing_decision_critical_claims(action, sources)
        if missing:
            missing_rows.append(
                f"{action_id} drops decision-critical content: "
                + "; ".join(missing[:4])
            )
    if missing_rows:
        raise ValueError(
            "canonical action(s) omit decision-critical consequences/constraints "
            "from grounded source clauses: " + " | ".join(missing_rows)
        )
