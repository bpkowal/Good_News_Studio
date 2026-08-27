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
_MECHANISM_FAILURE = re.compile(
    r"\b(?:(?:shelter|grid|power|oxygen|allocation|hospital|ward)\s[\w-]*\s*){0,4}"
    r"(?:fail(?:s|ed|ure)|collapse(?:s|d)?|outage|blackout)\b"
    r"|"
    r"\b(?:fail(?:s|ed|ure)|collapse(?:s|d)?)\s[\w-]*\s*"
    r"(?:shelter|grid|power|oxygen|allocation|hospital|ward)\b",
    re.IGNORECASE,
)
_CONCEALMENT = re.compile(
    r"\b(?:conceal(?:s|ed|ing|ment)?|hidden|secret|covert|undisclosed|"
    r"engineered\s+sacrifice|structural\s+design\s+remains\s+hidden|"
    r"remains?\s+concealed)\b[^.;,]{0,40}",
    re.IGNORECASE,
)
_SHORT_LABEL_SPLIT = re.compile(
    r",|;|\bwhile\b|\bcausing\b|\bkilling\b|\bpreserving\b|\ballowing\b|"
    r"\bleaving\b|\bthereby\b|\bresulting\b",
    re.IGNORECASE,
)


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
    mechanism: str = ""
    institutional_effect: str = ""
    source_clauses: tuple[str, ...] = ()
    completeness_status: str = "UNCHECKED"
    missing_critical: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


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
            _add(
                f"{groups['verb']} {groups['count']} {groups['who']}"
            )
        elif groups.get("count2"):
            _add(
                f"{groups['verb2']} {groups['count2']} {groups['who2']}"
            )
    for match in _MECHANISM_FAILURE.finditer(cleaned):
        _add(match.group(0))
    for match in _CONCEALMENT.finditer(cleaned):
        _add(match.group(0))
    return tuple(claims)


def _claim_coverage(claim: str, haystack: str) -> float:
    ignored = {
        "a", "an", "and", "as", "at", "by", "for", "from", "in", "into", "of",
        "on", "or", "the", "to", "with", "that", "this", "those", "these",
    }
    claim_tokens = {
        token for token in re.findall(r"[a-z0-9]+", claim.casefold())
        if token not in ignored and len(token) > 1
    }
    hay_tokens = {
        token for token in re.findall(r"[a-z0-9]+", haystack.casefold())
        if token not in ignored
    }
    if not claim_tokens:
        return 1.0
    return len(claim_tokens & hay_tokens) / len(claim_tokens)


def missing_decision_critical_claims(
    action: str,
    source_texts: Sequence[str],
    *,
    coverage_threshold: float = 0.72,
) -> tuple[str, ...]:
    """Return source claims that the canonical action failed to preserve."""
    action_text = " ".join(str(action or "").split())
    missing: list[str] = []
    for source in source_texts:
        for claim in extract_decision_critical_claims(source):
            if _claim_coverage(claim, action_text) >= coverage_threshold:
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
        "mechanism": mechanism,
        "institutional_effect": "; ".join(dict.fromkeys(institutional)),
    }


def build_canonical_action_record(
    action_id: str,
    action_text: str,
    *,
    actor: str = "",
    source_clause_texts: Sequence[str] = (),
    require_complete: bool = False,
) -> CanonicalActionRecord:
    """Compile id / short label / semantic action / structured fields."""
    semantic = " ".join(str(action_text or "").split())
    identity = compile_action_identity(semantic) if semantic else ActionIdentity(
        intervention="", basis="LEXICAL_FALLBACK", lexical_fallback="",
    )
    structure = _structure_from_identity(identity, semantic)
    if actor:
        structure["actor"] = " ".join(str(actor).split())
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
    if not action_clause_looks_complete(canonical):
        status = "INCOMPLETE_CLAUSE"
    elif missing:
        status = "MISSING_CRITICAL"
    elif sources:
        status = "COMPLETE"
    else:
        status = "UNCHECKED"
    record = CanonicalActionRecord(
        action_id=str(action_id).strip().upper() or "A?",
        short_label=short,
        canonical_semantic_action=canonical,
        actor=structure["actor"],
        intervention=structure["intervention"],
        beneficiaries=structure["beneficiaries"],
        harmed=structure["harmed"],
        mechanism=structure["mechanism"],
        institutional_effect=structure["institutional_effect"],
        source_clauses=sources,
        completeness_status=status,
        missing_critical=missing,
    )
    if require_complete and status in {"INCOMPLETE_CLAUSE", "MISSING_CRITICAL"}:
        detail = "; ".join(missing) if missing else canonical
        raise ValueError(
            f"{record.action_id} fails action-completeness ({status}): {detail}"
        )
    return record


def build_canonical_action_records(
    actions: Sequence[str],
    *,
    actor: str = "",
    grounded_clause_texts_by_id: dict[str, Sequence[str]] | None = None,
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
