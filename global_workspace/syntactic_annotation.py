"""Read-only spaCy dependency evidence for semantic-grounding experiments.

Nothing in this module creates, deletes, owns, or relates world-model nodes.
It emits grammatical observations that downstream experiments may inspect.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


CLAUSE_DEPENDENCIES = {"ROOT", "advcl", "ccomp", "xcomp", "conj", "relcl"}
SUBJECT_DEPENDENCIES = {"nsubj", "nsubjpass", "csubj", "csubjpass", "expl"}
OBJECT_DEPENDENCIES = {"dobj", "obj", "iobj", "pobj", "attr", "oprd"}
MODAL_LEMMAS = {"can", "could", "may", "might", "must", "shall", "should", "will", "would"}
CONDITIONAL_MARKERS = {"if", "unless", "provided", "when"}


def load_english_parser(model: str = "en_core_web_sm") -> Any:
    """Load an optional English dependency model with an actionable failure."""
    try:
        import spacy
    except ImportError as exc:  # pragma: no cover - exercised without optional extra
        raise RuntimeError(
            "spaCy is optional; install requirements-nlp.txt to run syntax audits"
        ) from exc
    try:
        return spacy.load(model)
    except OSError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            f"spaCy model {model!r} is unavailable; install requirements-nlp.txt"
        ) from exc


def _span_text(tokens: Iterable[Any]) -> str:
    rows = list(tokens)
    if not rows:
        return ""
    doc = rows[0].doc
    return doc[rows[0].i:rows[-1].i + 1].text


def annotate_text(text: str, *, nlp: Any) -> dict[str, Any]:
    """Return persisted dependency observations without semantic conclusions."""
    doc = nlp(text)
    sentences: list[dict[str, Any]] = []
    predicates: list[dict[str, Any]] = []
    for sentence_index, sentence in enumerate(doc.sents):
        sentence_predicates: list[str] = []
        for token in sentence:
            is_predicate = token.pos_ in {"VERB", "AUX"} and (
                token.dep_ in CLAUSE_DEPENDENCIES
                or any(child.dep_ in SUBJECT_DEPENDENCIES | OBJECT_DEPENDENCIES for child in token.children)
                or token.pos_ == "VERB"
            )
            if not is_predicate:
                continue
            predicate_id = f"S{sentence_index}_T{token.i}"
            sentence_predicates.append(predicate_id)
            children = list(token.children)
            subtree = list(token.subtree)
            subjects = [child.text for child in children if child.dep_ in SUBJECT_DEPENDENCIES]
            objects = [child.text for child in children if child.dep_ in OBJECT_DEPENDENCIES]
            negations = [child.text for child in children if child.dep_ == "neg"]
            modals = [
                child.text for child in children
                if child.dep_ in {"aux", "auxpass"} and child.lemma_.casefold() in MODAL_LEMMAS
            ]
            subordinate_markers = [
                child.text for child in children if child.dep_ == "mark"
            ]
            markers = [
                child.text for child in children
                if child.text.casefold() in CONDITIONAL_MARKERS
            ]
            conjunctions = [child.text for child in children if child.dep_ == "conj"]
            predicates.append({
                "predicate_id": predicate_id,
                "sentence_index": sentence_index,
                "token_index": token.i,
                "start_char": token.idx,
                "end_char": token.idx + len(token.text),
                "text": token.text,
                "lemma": token.lemma_.casefold(),
                "pos": token.pos_,
                "dependency": token.dep_,
                "head_token_index": token.head.i,
                "head_text": token.head.text,
                "clause_head": token.dep_ in CLAUSE_DEPENDENCIES,
                "subtree_span": _span_text(subtree),
                "subjects": subjects,
                "objects": objects,
                "negations": negations,
                "modals": modals,
                "conditional_markers": markers,
                "subordinate_markers": subordinate_markers,
                "conjunctions": conjunctions,
                "children": [
                    {"text": child.text, "lemma": child.lemma_.casefold(), "dependency": child.dep_}
                    for child in children
                ],
            })
        sentences.append({
            "sentence_index": sentence_index,
            "text": sentence.text,
            "start_char": sentence.start_char,
            "end_char": sentence.end_char,
            "predicate_ids": sentence_predicates,
        })
    return {
        "annotation_version": "1.0",
        "mode": "READ_ONLY_STRUCTURAL",
        "model": nlp.meta.get("name", "unknown"),
        "text": text,
        "sentences": sentences,
        "predicates": predicates,
        "tokens": [{
            "token_index": token.i,
            "start_char": token.idx,
            "end_char": token.idx + len(token.text),
            "text": token.text,
            "lemma": token.lemma_.casefold(),
            "pos": token.pos_,
            "dependency": token.dep_,
            "head_token_index": token.head.i,
        } for token in doc],
    }


def predicate_lemmas(text: str, *, nlp: Any) -> list[str]:
    """Extract the predicate lemmas in a short gold-event description."""
    doc = nlp(text)
    lemmas = [token.lemma_.casefold() for token in doc if token.pos_ == "VERB"]
    if lemmas:
        return list(dict.fromkeys(lemmas))
    # Short evaluator glosses are sometimes parsed as noun fragments even when
    # their root lemma is a verb in the full sentence (for example, "server
    # shuts down"). Preserve the fallback as a separately reported match.
    roots = [token.lemma_.casefold() for token in doc if token.dep_ == "ROOT"]
    return list(dict.fromkeys(roots))
