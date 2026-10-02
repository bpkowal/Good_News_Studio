"""Ellipsis proposer. A separate policy from the four-action relation CEM.

The detector decides whether a gap is present. A verb-phrase copy is offered
only when the remnant is a bare auxiliary, "too"/"either", or "does the same".
The copy is a verb phrase already in the text, not the bare verb and not the
object. A separate gate copies the left verb onto "but/and not" plus a nominal,
with that nominal in the contrasting role and negative polarity. Other remnant
types stay silent. Nothing is rewritten, and a copy is not a world-state
commitment. If the detector abstains, nothing is proposed.

Weights are fit by a cross-entropy search on Hoosier training groups whose
spelled-out form shows the gap. Development groups choose how close a second
candidate must be before it is kept. The test groups and the dilemma probe
are not used to fit either choice.
"""

import json
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

import ellipsis_detector as detector
from ellipsis_corpus import load_examples, rows_for_split, surface

FEATURE_NAMES = (
    "bias",
    "verb_and_nominal",
    "verb_only",
    "nominal_only",
    "nearest",
    "same_sentence",
    "aux_match",
    "abstain_bias",
    "bare_auxiliary",
    "too_either",
    "do_the_same",
    "wh_remnant",
    "coordinate_gap",
    "verbless_fragment",
    "not_nominal",
    "than_nominal",
)
CANDIDATE_WIDTH = 7
# Indices into the detector's feature vector. Other cues stay silent.
VERB_PHRASE_REMNANTS = (0, 1, 2)
MODEL_PATH = Path(__file__).resolve().parent / "resources" / "ellipsis_proposer_v1.json"
_TOKEN = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?|[^\s\w]")


def normalize(text):
    return " ".join(surface(text).casefold().split())


def words(text):
    return [token for token in _TOKEN.findall(text) if token.isalnum() or "'" in token]


def missing_spans(gapped, full):
    """Word sequences present in the spelled-out sentence and absent from the gap."""
    source, target = words(gapped), words(full)
    spans = []
    for tag, _i1, _i2, j1, j2 in SequenceMatcher(a=source, b=target, autojunk=False).get_opcodes():
        if tag in {"insert", "replace"} and j1 < j2:
            span = target[j1:j2]
            if span:
                spans.append(" ".join(span))
    return spans


def _content_tokens(tokens):
    kept = []
    for token in tokens:
        # "War and Peace" stays together. A coordinator of a verb starts another clause.
        if token.dep_ == "cc" and token.head.pos_ in {"VERB", "AUX"}:
            break
        if token.lower_ == "but" and token.dep_ == "cc":
            break
        if token.dep_ == "neg" or token.lemma_ == "not":
            continue
        if token.pos_ == "PUNCT":
            continue
        kept.append(token)
    return kept


def _phrase(verb):
    kept = [verb]
    for child in verb.children:
        if child.dep_ in {"nsubj", "nsubjpass", "aux", "auxpass", "mark", "punct", "cc", "conj", "advcl", "discourse", "neg"}:
            continue
        kept.extend(child.subtree)
    ordered = _content_tokens(sorted(set(kept), key=lambda token: token.i))
    nominals = [token for token in ordered if token.pos_ in {"NOUN", "PROPN", "PRON"} and token is not verb]
    return ordered, nominals


_BARE_COMPLEMENTS = {"attr", "acomp", "dobj", "obj", "xcomp", "ccomp", "prep", "relcl"}
_FINITE_AUX = {"be", "have", "do", "can", "could", "will", "would", "may", "might", "must", "should", "shall"}


def _bare_do(token):
    """A do that stands in for a missing phrase. "do dentists" still has its object."""
    if token.lemma_ != "do" or token.dep_ not in {"ROOT", "advcl", "conj", "ccomp", "relcl", "parataxis"}:
        return False
    if any(child.pos_ == "VERB" and child.dep_ != "aux" for child in token.children):
        return False
    return not any(child.dep_ in _BARE_COMPLEMENTS for child in token.children)


def _remnant_index(doc):
    """The last bare do. An earlier "do" with its own object is not the gap."""
    found = None
    for token in doc:
        if _bare_do(token):
            found = token.i
    if found is not None:
        return found
    content = [token for token in doc if not token.is_punct]
    return content[-1].i if content else 0


def _licensing_aux(verb):
    """The modal or do-support on this verb, or "finite" when the verb itself is tensed."""
    auxiliaries = [child for child in verb.children if child.dep_ in {"aux", "auxpass"}]
    modals = [child for child in auxiliaries if child.tag_ == "MD"]
    if modals:
        return modals[-1].lemma_
    if any(child.lemma_ == "do" for child in auxiliaries):
        return "do"
    if verb.tag_ in {"VBZ", "VBP", "VBD"}:
        return "finite"
    return ""


def _remnant_aux_lemma(doc):
    """The auxiliary left in the gap. A modal that still has its verb is not the gap."""
    if any(_bare_do(token) for token in doc):
        return "do"
    found = ""
    for token in doc:
        if token.tag_ == "MD":
            if token.dep_ in {"aux", "auxpass"} and token.head.pos_ == "VERB" and token.head.i > token.i:
                continue
            found = token.lemma_
        elif token.lemma_ == "do" and token.dep_ in {"ROOT", "advcl", "conj", "ccomp", "relcl", "parataxis"}:
            found = "do"
    return found


def _aux_match(candidate_aux, remnant_aux):
    """Do-support matches a tensed verb. A modal matches only the same modal."""
    if not remnant_aux:
        return 0.0
    if remnant_aux == "do":
        return float(candidate_aux in {"do", "finite"})
    return float(candidate_aux == remnant_aux)


def _finite_clause(verb):
    """A tensed predicate, or one licensed by a modal, do, be, or have."""
    if verb.tag_ in {"VBZ", "VBP", "VBD"}:
        return True
    return any(child.dep_ in {"aux", "auxpass"} and child.lemma_ in _FINITE_AUX for child in verb.children)


def antecedent_candidates(doc):
    """Copies of material before the remnant. The remnant itself is not a candidate."""
    remnant = _remnant_index(doc)
    verbs = [token for token in doc if token.pos_ == "VERB" and token.i < remnant and not (
        token.lemma_ == "do" and not any(child.pos_ == "VERB" for child in token.children))]
    found, seen = [], set()
    for verb in verbs:
        ordered, nominals = _phrase(verb)
        same = int(verb.sent.start <= remnant)
        options = []
        verbal = any(token.pos_ == "VERB" and token is not verb for token in ordered)
        if nominals or (_finite_clause(verb) and verbal):
            options.append(("verb_and_nominal", ordered))
        options.append(("verb_only", _content_tokens([verb])))
        for child in verb.children:
            if child.dep_ in {"dobj", "obj", "attr"} and child.i < remnant:
                options.append(("nominal_only", _content_tokens(list(child.subtree))))
        for kind, tokens in options:
            tokens = [token for token in tokens if token.i < remnant]
            text = " ".join(token.text for token in tokens).strip()
            lemmas = " ".join(token.lemma_ for token in tokens).strip()
            key = (kind, normalize(text), normalize(lemmas))
            if not text or key in seen:
                continue
            seen.add(key)
            found.append(dict(text=text, lemmas=lemmas, kind=kind, nearest=0,
                              same_sentence=same, tokens=len(tokens), end=tokens[-1].i,
                              finite=_finite_clause(verb), matrix=verb.dep_ == "ROOT",
                              aux=_licensing_aux(verb), aux_match=0.0))
    for kind in {candidate["kind"] for candidate in found}:
        group = [candidate for candidate in found if candidate["kind"] == kind]
        best = max(candidate["end"] for candidate in group)
        for candidate in group:
            candidate["nearest"] = int(candidate["end"] == best)
    return found


def phrase_copies(doc):
    """Finite verb phrases before the gap, marked when their auxiliary matches the remnant."""
    candidates = [item for item in antecedent_candidates(doc) if item["kind"] == "verb_and_nominal"]
    finite = [item for item in candidates if item["finite"]]
    if finite:
        candidates = finite
        latest = max(item["end"] for item in candidates)
        for item in candidates:
            item["nearest"] = int(item["end"] == latest)
    remnant_aux = _remnant_aux_lemma(doc)
    for item in candidates:
        item["aux_match"] = _aux_match(item["aux"], remnant_aux)
    return candidates


def _features(candidate):
    """Kind and distance only. The remnant decides abstaining, not which copy ranks first."""
    return np.asarray([
        1.0,
        float(candidate["kind"] == "verb_and_nominal"),
        float(candidate["kind"] == "verb_only"),
        float(candidate["kind"] == "nominal_only"),
        float(candidate["nearest"]),
        float(candidate["same_sentence"]),
        float(candidate.get("aux_match", 0.0)),
    ], dtype=float)


def _verb_phrase_remnant(remnant):
    """Bare auxiliary, "too"/"either", or "does the same". Other gaps are not this copy."""
    return any(float(remnant[index]) > 0 for index in VERB_PHRASE_REMNANTS)


_NOMINAL = {"NOUN", "PROPN", "PRON"}
_PREP_ROLE = {"to": "destination", "on": "location", "in": "location", "at": "location"}
_OBJECT_DEPS = {"dobj", "obj", "attr"}


def _nominal_text(head, prep=None):
    tokens = [head]
    if prep is not None:
        tokens.append(prep)
    for child in head.children:
        if child.dep_ in {"det", "amod", "compound", "nummod", "poss"}:
            tokens.append(child)
    tokens.sort(key=lambda token: token.i)
    return " ".join(token.text for token in tokens)


def _stripping_sites(doc):
    """but/and + not + a nominal. A later clause does not erase the remnant."""
    sites = []
    for token in doc:
        if token.lower_ != "not":
            continue
        following = next((item for item in token.sent if item.i > token.i and not item.is_space), None)
        if following is not None and following.lower_ == "only":
            continue
        previous = [item for item in token.sent if item.i < token.i and not item.is_punct and not item.is_space]
        if not previous or previous[-1].lower_ not in {"but", "and"}:
            continue
        later = [item for item in detector.remnant_after_not(token) if not item.is_space]
        # In ``either X or Y, but not both``, bare ``both`` closes the
        # alternative set; it is not an argument whose missing predicate must
        # be reconstructed.  Keep the guard structural and narrow so nominal
        # remnants such as ``but not both parents`` remain ordinary stripping.
        remnant_words = [item for item in later if not item.is_punct]
        closes_alternative_set = (
            len(remnant_words) == 1
            and remnant_words[0].lower_ == "both"
            and any(
                item.lower_ == "or" and item.i < token.i
                for item in token.sent
            )
        )
        if closes_alternative_set:
            continue
        if not any(item.pos_ in _NOMINAL for item in later):
            continue
        sites.append((token, previous[-1], later))
    return sites


def _governed_verb(coordinator, negation):
    verb = coordinator.head
    seen = set()
    while verb.i not in seen:
        seen.add(verb.i)
        if verb.pos_ == "VERB" and verb.i < negation.i:
            return verb
        if verb.head.i == verb.i:
            return None
        verb = verb.head
    return None


def _remnant_shape(later, negation):
    first = later[0]
    if first.pos_ == "ADP":
        if first.lower_ not in _PREP_ROLE:
            return None
        heads = [child for child in first.children
                 if child.dep_ == "pobj" and child.i > negation.i and child.pos_ in _NOMINAL]
        if len(heads) != 1:
            return None
        return dict(prep=first, head=heads[0], role=_PREP_ROLE[first.lower_])
    heads = [item for item in later if item.pos_ in _NOMINAL and item.dep_ not in {"compound", "poss"}]
    if len(heads) != 1:
        return None
    return dict(prep=None, head=heads[0], role=None)


def _role_partners(verb, negation, remnant):
    if remnant["prep"] is not None:
        found = []
        for child in verb.children:
            if child.i >= negation.i or child.lower_ != remnant["prep"].lower_:
                continue
            if child.dep_ not in {"prep", "dative"}:
                continue
            objects = [item for item in child.children if item.dep_ == "pobj" and item.pos_ in _NOMINAL]
            if len(objects) == 1:
                found.append((remnant["role"], objects[0]))
        return found
    found = []
    for child in verb.children:
        if child.i >= negation.i or child.pos_ not in _NOMINAL:
            continue
        if child.dep_ in _OBJECT_DEPS:
            found.append(("object", child))
        elif child.dep_ == "dative":
            found.append(("destination", child))
    if found:
        return found
    return [("subject", child) for child in verb.children
            if child.i < negation.i and child.dep_ in {"nsubj", "nsubjpass"} and child.pos_ in _NOMINAL]


def _stripping_readings(negation, coordinator, later):
    verb = _governed_verb(coordinator, negation)
    if verb is None or not _finite_clause(verb):
        return []
    remnant = _remnant_shape(later, negation)
    if remnant is None:
        return []
    partners = _role_partners(verb, negation, remnant)
    if not partners:
        return []
    closest = {}
    for role, partner in partners:
        current = closest.get(role)
        if current is None or partner.i > current.i:
            closest[role] = partner
    subjects = [child for child in verb.children
                if child.i < negation.i and child.dep_ in {"nsubj", "nsubjpass"} and child.pos_ in _NOMINAL]
    shared = subjects[0] if len(subjects) == 1 else None
    proposals = []
    for role, partner in closest.items():
        subject = remnant["head"] if role == "subject" else shared
        prep = remnant["prep"]
        partner_prep = partner.head if partner.head.pos_ == "ADP" else None
        proposals.append(dict(
            text=verb.text,
            kind="stripping",
            status="unresolved",
            antecedent_copy=True,
            polarity="negative",
            role=role,
            verb=verb.text,
            verb_lemma=verb.lemma_,
            verb_index=verb.i,
            not_index=negation.i,
            remnant=_nominal_text(remnant["head"], prep),
            remnant_index=remnant["head"].i,
            prep_index=None if prep is None else prep.i,
            subject="" if subject is None else _nominal_text(subject),
            subject_index=None if subject is None else subject.i,
            partner=_nominal_text(partner, partner_prep),
            partner_index=partner.i,
        ))
    return proposals


def stripping_copies(doc):
    """Copies for one stripping remnant.

    None means this sentence is not but/and + not + a nominal. An empty list
    means that remnant has no left dependent in a matching role, so no copy
    is offered.
    """
    sites = _stripping_sites(doc)
    if len(sites) != 1:
        return None
    negation, coordinator, later = sites[0]
    return _stripping_readings(negation, coordinator, later)


_SENTENCE_VERB_PHRASE = ("bare_auxiliary_clause", "too_either", "do_the_same")


def scenario_stripping(doc, decide):
    """Stripping copies for each sentence that has this remnant.

    A later sentence does not change the decision. A verb-phrase remnant in
    the same sentence is left to that gate. Indices refer to the whole document.
    """
    copies = []
    for sent in doc.sents:
        sentence = sent.text.strip()
        if not sentence:
            continue
        decision = decide(sentence)
        features = decision["features"]
        if not decision["present"]:
            continue
        if any(float(features.get(name, 0)) > 0 for name in _SENTENCE_VERB_PHRASE):
            continue
        sites = [site for site in _stripping_sites(doc) if site[0] in sent]
        if len(sites) != 1:
            continue
        copies.extend(_stripping_readings(*sites[0]))
    return copies


def _stripping_result(decision, copies):
    proposals = [dict(item) for item in copies]
    remnants = {item["not_index"] for item in proposals}
    choice = None
    if len(remnants) == 1 and len(proposals) > 1:
        choice = dict(
            kind="interpretation", selection_rule="at_most_one", exhaustive=False,
            candidate_texts=[f"{item['role']}:{item['remnant']}" for item in proposals])
    return dict(present=True, proposals=proposals, choice=choice, detector_score=decision["score"],
                gate="stripping_not_nominal", stripping=proposals)


def _abstain_features(remnant):
    cues = [float(remnant[index] > 0) for index in range(8)]
    return np.asarray([1.0, *cues], dtype=float)


def _contained(candidate, span):
    """The copy may be the gap, or the gap plus words the remnant already showed."""
    left, right = candidate.split(), span.split()
    if not left or len(left) > len(right):
        return False
    width = len(left)
    return any(right[start:start + width] == left for start in range(len(right) - width + 1))


def _gold_index(candidates, spans):
    """The longest copy that sits inside the spelled-out gap is the training target."""
    targets = [normalize(span) for span in spans]
    best, best_length = 0, 0
    for index, candidate in enumerate(candidates, start=1):
        forms = {normalize(candidate["text"]), normalize(candidate["lemmas"])}
        for form in forms:
            if not form:
                continue
            if form in targets or any(_contained(form, target) for target in targets):
                length = len(form.split())
                if length > best_length:
                    best, best_length = index, length
    return best


def build_examples(rows, nlp, gate=None):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["group_id"]].append(row)
    examples = []
    order = []
    for group_id, items in grouped.items():
        gapped = next((item["text"] for item in items if item["label"] == 1), None)
        full = next((item["text"] for item in items if item["label"] == 0), None)
        if not gapped or not full:
            continue
        order.append((group_id, gapped, full))
    docs = list(nlp.pipe((item[1] for item in order), batch_size=32))
    decision = gate or detector.load_detector()
    for (group_id, gapped, full), doc in zip(order, docs):
        remnant = detector.featurize_doc(doc)
        if not decision.decide_features(remnant)["present"]:
            continue
        candidates = antecedent_candidates(doc)
        if not candidates:
            continue
        features = np.vstack([_features(candidate) for candidate in candidates])
        examples.append(dict(
            group_id=group_id,
            features=features,
            remnant=remnant,
            gold=_gold_index(candidates, missing_spans(gapped, full)),
            texts=[candidate["text"] for candidate in candidates],
        ))
    return examples


def _choice(weights, features, remnant):
    """0 is abstain. A candidate wins a tie with abstaining."""
    candidate_scores = features @ weights[:CANDIDATE_WIDTH]
    abstain = float(_abstain_features(remnant) @ weights[CANDIDATE_WIDTH:])
    scores = np.concatenate([[abstain], candidate_scores])
    if scores[0] > scores[1:].max():
        return 0, scores
    return int(1 + np.argmax(scores[1:])), scores


def _objective(weights, examples, margin):
    """Gold in the offered set, minus a penalty for every extra copy."""
    if not examples:
        return 0.0
    hits = extras = 0
    for example in examples:
        chosen = _within_margin(weights, example["features"], margin, example["remnant"])
        if example["gold"] == 0:
            hits += int(not chosen)
            extras += len(chosen)
        else:
            hits += int(example["gold"] in chosen)
            extras += max(0, len(chosen) - 1)
    total = len(examples)
    return hits / total - 0.08 * extras / total


def fit_policy(examples, seed=0, population=240, generations=60, aux_prior=0.0):
    """Cross-entropy search. The nearest copy should outrank both silence and the other copies."""
    width = len(FEATURE_NAMES)
    rng = np.random.default_rng(seed)
    mean = np.zeros(width)
    mean[4] = 1.5
    mean[1] = 0.4
    mean[6] = aux_prior
    std = np.ones(width)
    best_weights = mean.copy()
    best_accuracy = -1.0
    for _generation in range(generations):
        population_weights = rng.normal(mean, std, size=(population, width))
        scores = np.asarray([_objective(weights, examples, margin=0.25) for weights in population_weights])
        elite = population_weights[np.argsort(scores)[-max(2, population // 5):]]
        mean = elite.mean(axis=0)
        std = np.maximum(elite.std(axis=0), 0.05)
        winner = int(scores.argmax())
        if scores[winner] > best_accuracy:
            best_accuracy = float(scores[winner])
            best_weights = population_weights[winner].copy()
    return best_weights, best_accuracy


def choose_margin(weights, examples):
    """Development chooses how close a second copy must be. Larger sets pay a penalty."""
    best_margin, best_value = 0.0, -1.0
    for margin in (0.0, 0.05, 0.1, 0.25, 0.5, 1.0):
        if value_of(weights, examples, margin) > best_value:
            best_value = value_of(weights, examples, margin)
            best_margin = margin
    return best_margin


def value_of(weights, examples, margin):
    return _objective(weights, examples, margin)


def _within_margin(weights, features, margin, remnant):
    _choice_index, scores = _choice(weights, features, remnant)
    if scores[0] > scores[1:].max():
        return []
    best = float(scores[1:].max())
    return [index for index, score in enumerate(scores[1:], start=1) if best - float(score) <= margin]


def _metrics(weights, margin, examples):
    exact = recovered = false_proposals = 0
    recoverable = empty = proposed = 0
    for example in examples:
        chosen = _within_margin(weights, example["features"], margin, example["remnant"])
        proposed += len(chosen)
        if example["gold"] == 0:
            empty += 1
            false_proposals += int(bool(chosen))
        else:
            recoverable += 1
            recovered += int(example["gold"] in chosen)
            exact += int(len(chosen) == 1 and chosen[0] == example["gold"])
    total = max(len(examples), 1)
    return dict(
        rows=len(examples),
        objective=_objective(weights, examples, margin),
        exact_when_recoverable=exact / max(recoverable, 1),
        gold_in_set_when_recoverable=recovered / max(recoverable, 1),
        false_proposal_when_empty=false_proposals / max(empty, 1),
        mean_proposals=proposed / total,
    )


class EllipsisProposer:
    def __init__(self, weights, margin, feature_names=FEATURE_NAMES):
        self.weights = np.asarray(weights, dtype=float)
        self.margin = float(margin)
        self.feature_names = tuple(feature_names)
        if self.weights.shape != (len(self.feature_names),):
            raise ValueError("Proposer weights do not match the frozen feature list.")

    def propose(self, text, nlp=None, gate=None, *, doc=None):
        """Return indices into the exact input text's parse, never cleaned text.

        Exporters may supply their original Doc. Corpus normalization belongs in
        corpus preparation, not at this source-evidence boundary.
        """
        parser = nlp or detector.get_nlp()
        decider = gate or detector.load_detector()
        if doc is None:
            doc = parser(text)
        elif doc.text != text:
            raise ValueError('Proposal document must match the exact original text.')
        stripping = scenario_stripping(doc, lambda sentence: decider.decide(sentence, parser))
        decision = decider.decide(text, parser)
        remnant = detector.featurize_doc(doc)
        if decision["present"] and _verb_phrase_remnant(remnant):
            result = self._verb_phrase(decision, doc, remnant)
            result["stripping"] = stripping
            return result
        if stripping:
            return _stripping_result(decision, stripping)
        if not decision["present"]:
            return dict(present=False, proposals=[], choice=None, detector_score=decision["score"],
                        stripping=[])
        single = stripping_copies(doc)
        if single is not None:
            return _stripping_result(decision, single)
        return dict(present=True, proposals=[], choice=None, detector_score=decision["score"],
                    gate="not_a_verb_phrase_remnant", stripping=[])

    def _verb_phrase(self, decision, doc, remnant):
        candidates = phrase_copies(doc)
        if not candidates:
            return dict(present=True, proposals=[], choice=None, detector_score=decision["score"],
                        gate="verb_phrase")
        features = np.vstack([_features(candidate) for candidate in candidates])
        chosen = _within_margin(self.weights, features, self.margin, remnant)
        _index, scores = _choice(self.weights, features, remnant)
        proposals = []
        for index in chosen:
            candidate = candidates[index - 1]
            proposals.append(dict(
                text=candidate["text"], kind=candidate["kind"], status="unresolved",
                score=float(scores[index]), antecedent_copy=True))
        choice = None
        if len(proposals) > 1:
            choice = dict(kind="interpretation", selection_rule="at_most_one", exhaustive=False,
                          candidate_texts=[item["text"] for item in proposals])
        return dict(present=True, proposals=proposals, choice=choice, detector_score=decision["score"],
                    gate="verb_phrase")

    def to_json(self, metrics):
        return dict(
            schema_version="ellipsis-proposer-1",
            citation="Cavar, Mompelat, and Abdo (2024), The Typology of Ellipsis. English THEC.",
            feature_names=list(self.feature_names),
            weights=self.weights.tolist(),
            margin=self.margin,
            independent_of="parsing_game four-action relation CEM",
            rewrites_sentence=False,
            world_state_commitment="not_authorized",
            runs_only_after_detector=True,
            phrase_gate=["bare_auxiliary_clause", "too_either", "do_the_same"],
            copy_kind="verb_and_nominal",
            probe_not_used_for_fitting=True,
            metrics=metrics,
        )


def load_proposer(path=None):
    payload = json.loads(Path(path or MODEL_PATH).read_text(encoding="utf-8"))
    if payload.get("schema_version") != "ellipsis-proposer-1":
        raise ValueError("Unsupported ellipsis proposer schema.")
    return EllipsisProposer(payload["weights"], payload["margin"], payload["feature_names"])


def synthetic_examples(rows, nlp):
    """Ranking examples from the synthetic file. The gold phrase must still be a copy."""
    examples = []
    docs = list(nlp.pipe((row["text"] for row in rows), batch_size=32))
    for row, doc in zip(rows, docs):
        candidates = phrase_copies(doc)
        if not candidates:
            continue
        target = normalize(row["gold"])
        gold = 0
        for index, candidate in enumerate(candidates, start=1):
            forms = {normalize(candidate["text"]), normalize(candidate["lemmas"])}
            if target in forms:
                gold = index
                break
        if not gold:
            continue
        examples.append(dict(
            split=row["synthetic_split"],
            features=np.vstack([_features(candidate) for candidate in candidates]),
            remnant=detector.featurize_doc(doc),
            gold=gold,
        ))
    return examples


def fit_synthetic_and_save(rows, path=None):
    """Fit the auxiliary match on synthetic training rows. Synthetic dev chooses the margin."""
    nlp = detector.get_nlp()
    splits = {"train": [], "dev": []}
    for example in synthetic_examples(rows, nlp):
        splits[example["split"]].append(example)
    weights, train_search = fit_policy(splits["train"], aux_prior=2.5)
    margin = choose_margin(weights, splits["dev"])
    fitted = EllipsisProposer(weights, margin)
    metrics = {f"synthetic_{name}": _metrics(weights, margin, splits[name]) for name in ("train", "dev")}
    metrics["synthetic_train"]["search_accuracy"] = train_search
    metrics["fitted_on"] = "synthetic training rows from Hoosier train phrases"
    destination = Path(path or MODEL_PATH)
    destination.write_text(json.dumps(fitted.to_json(metrics), indent=2) + "\n", encoding="utf-8")
    return fitted, metrics


def train_and_save(root=None, path=None):
    nlp = detector.get_nlp()
    rows = load_examples(root)
    splits = {name: build_examples(rows_for_split(rows, name), nlp) for name in ("train", "dev", "test")}
    weights, train_search = fit_policy(splits["train"])
    margin = choose_margin(weights, splits["dev"])
    proposer = EllipsisProposer(weights, margin)
    metrics = {name: _metrics(weights, margin, splits[name]) for name in ("train", "dev", "test")}
    metrics["train"]["search_accuracy"] = train_search
    destination = Path(path or MODEL_PATH)
    destination.write_text(json.dumps(proposer.to_json(metrics), indent=2) + "\n", encoding="utf-8")
    return proposer, metrics


def main():
    _proposer, metrics = train_and_save()
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
