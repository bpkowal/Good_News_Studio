"""Ellipsis presence detector. Independent of the four-action relation CEM.

Stage one only. A positive decision means the sentence still has a gap.
It does not name the missing words, rewrite the sentence, or authorize a
world-state commitment. The proposer that would offer those words is a
later policy.

Features were fixed from the ellipsis types in the Hoosier corpus, then checked
against the training split only. A verb with no complement was dropped: in
newswire it fired on ordinary verbs more often than on null-complement
anaphora. Weights are fit on the training groups. The threshold is chosen on
the development groups, and it stays above the score of a sentence with no cue.
"""

import json
from pathlib import Path

import numpy as np

from ellipsis_corpus import SPLIT_SALT, load_examples, rows_for_split

FEATURE_NAMES = (
    "bare_auxiliary_clause",
    "too_either",
    "do_the_same",
    "wh_remnant",
    "than_nominal",
    "not_nominal",
    "coordinate_gap",
    "verbless_fragment",
)
MODEL_PATH = Path(__file__).resolve().parent / "resources" / "ellipsis_detector_v1.json"
_NLP = None


def get_nlp():
    global _NLP
    if _NLP is None:
        import spacy
        _NLP = spacy.load("en_core_web_sm")
    return _NLP


def _bare_auxiliary_clause(doc):
    """A do/modal/copula standing as the clause predicate, with no complement.

    "does not pull" keeps pull as the clause verb, so it does not count.
    "does not" is itself the clause verb. Copulas with a complement do not count.
    """
    complements = {"attr", "acomp", "dobj", "obj", "xcomp", "ccomp", "prep", "relcl"}
    count = 0
    for token in doc:
        if token.dep_ not in {"ROOT", "advcl", "conj", "ccomp", "relcl", "parataxis"}:
            continue
        if any(child.pos_ == "VERB" and child.dep_ != "aux" for child in token.children):
            continue
        if any(child.dep_ in complements for child in token.children):
            continue
        if token.lemma_ == "do" or token.pos_ == "AUX":
            count += 1
    return count


def _too_either(doc):
    count = 0
    for token in doc:
        if token.lemma_ not in {"too", "either"}:
            continue
        if token.lemma_ == "either" and any(later.lower_ == "or" and later.i > token.i for later in token.sent):
            continue
        sent_tokens = list(token.sent)
        position = sent_tokens.index(token)
        if position >= len(sent_tokens) - 2 or token.head.pos_ == "AUX" or token.head.lemma_ == "do":
            count += 1
    return count


def _do_the_same(doc):
    if any(token.lemma_ == "same" for token in doc) and any(token.lemma_ == "do" for token in doc):
        return 1
    return 0


def _wh_remnant(doc):
    """A clause that ends on a wh-word, as in "I don't know who."

    "Who left?" and "which book" do not end on the wh-word.
    """
    count = 0
    endings = {"who", "what", "where", "when", "why", "how"}
    for sent in doc.sents:
        content = [token for token in sent if not token.is_punct]
        if not content:
            continue
        last = content[-1]
        if last.lower_ in endings or last.tag_ in {"WP", "WRB"}:
            count += 1
    return count


def _than_nominal(doc):
    count = 0
    for token in doc:
        if token.lower_ != "than":
            continue
        later = [item for item in token.sent if item.i > token.i]
        if any(item.pos_ in {"NOUN", "PROPN", "PRON"} for item in later) and not any(
                item.pos_ in {"VERB", "AUX"} for item in later):
            count += 1
    return count


_CLAUSE_MARKS = {"if", "when", "because", "although", "while", "unless", "so", "before", "after"}
_NOMINAL_POS = {"NOUN", "PROPN", "PRON"}


def remnant_after_not(negation):
    """The nominal after but/and not, stopping where the next clause starts.

    A comma, a clause mark, or a following verb ends the remnant. The nominal
    does not have to be the last word in the sentence.
    """
    taken = []
    seen_nominal = False
    for item in negation.sent:
        if item.i <= negation.i:
            continue
        if item.is_space:
            continue
        if item.is_punct:
            if seen_nominal and item.text == ",":
                break
            continue
        opens_clause = item.pos_ in {"VERB", "AUX", "SCONJ"} or item.lower_ in _CLAUSE_MARKS
        if opens_clause:
            if not seen_nominal:
                return []
            break
        if seen_nominal and item.lower_ in {"and", "or", "but"} and item.dep_ == "cc":
            break
        taken.append(item)
        if item.pos_ in _NOMINAL_POS and item.dep_ not in {"compound", "poss"}:
            seen_nominal = True
    if not seen_nominal:
        return []
    heads = [item for item in taken
             if item.pos_ in _NOMINAL_POS and item.dep_ not in {"compound", "poss"}]
    if any(head.dep_ in {"nsubj", "nsubjpass"} and head.head.i > negation.i
           and head.head.pos_ in {"VERB", "AUX"} for head in heads):
        return []
    return taken


def _not_nominal(doc):
    """Stripping: "but not the diamonds," including when another clause follows."""
    count = 0
    for token in doc:
        if token.lower_ != "not":
            continue
        previous = [item for item in token.sent if item.i < token.i and not item.is_punct and not item.is_space]
        if not previous or previous[-1].lower_ not in {"but", "and"}:
            continue
        later = remnant_after_not(token)
        if any(item.pos_ in _NOMINAL_POS for item in later):
            count += 1
    return count


def _coordinate_gap(doc):
    count = 0
    for token in doc:
        if token.lower_ not in {"and", "or", "but"}:
            continue
        later = [item for item in token.sent if item.i > token.i]
        nominals = [item for item in later if item.pos_ in {"NOUN", "PROPN", "PRON"}]
        verbs = [item for item in later if item.pos_ in {"VERB", "AUX"}]
        if len(nominals) >= 2 and not verbs:
            count += 1
    return count


def _verbless_fragment(doc):
    count = 0
    for sent in doc.sents:
        has_nominal = any(token.pos_ in {"NOUN", "PROPN", "PRON", "NUM"} for token in sent)
        has_verb = any(token.pos_ in {"VERB", "AUX"} for token in sent)
        if has_nominal and not has_verb:
            count += 1
    return count


_EXTRACTORS = (
    _bare_auxiliary_clause,
    _too_either,
    _do_the_same,
    _wh_remnant,
    _than_nominal,
    _not_nominal,
    _coordinate_gap,
    _verbless_fragment,
)


def featurize_doc(doc):
    return np.asarray([float(function(doc)) for function in _EXTRACTORS], dtype=float)


def featurize(text, nlp=None):
    parser = nlp or get_nlp()
    normalized = text.replace("\u2019", "'").replace("\u2018", "'")
    return featurize_doc(parser(normalized))


def _sigmoid(values):
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30, 30)))


def _metrics(labels, predicted):
    labels = np.asarray(labels)
    predicted = np.asarray(predicted)
    true_positive = int(np.sum((predicted == 1) & (labels == 1)))
    false_positive = int(np.sum((predicted == 1) & (labels == 0)))
    false_negative = int(np.sum((predicted == 0) & (labels == 1)))
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return dict(precision=precision, recall=recall, f1=f1,
                true_positive=true_positive, false_positive=false_positive,
                false_negative=false_negative, rows=int(len(labels)))


def choose_threshold(scores, labels, zero_score):
    """Best balanced accuracy among thresholds that reject a featureless sentence.

    A sentence with no ellipsis cue stays absent. Ties keep the higher threshold.
    """
    labels = np.asarray(labels)
    best = None
    for threshold in sorted(set(float(value) for value in np.round(scores, 6))):
        if threshold <= zero_score:
            continue
        predicted = (scores >= threshold).astype(int)
        positive = max(int((labels == 1).sum()), 1)
        negative = max(int((labels == 0).sum()), 1)
        true_positive = int(np.sum((predicted == 1) & (labels == 1)))
        true_negative = int(np.sum((predicted == 0) & (labels == 0)))
        balanced = 0.5 * (true_positive / positive + true_negative / negative)
        f1 = _metrics(labels, predicted)["f1"]
        key = (balanced, f1, threshold)
        if best is None or key > best[0]:
            best = (key, threshold)
    if best is None:
        return float(min(1.0, zero_score + 1e-3))
    return best[1]


def fit_linear(features, labels, l2=0.02, steps=2500, learning_rate=0.15):
    labels = np.asarray(labels, dtype=float)
    positive = max(labels.sum(), 1.0)
    negative = max(len(labels) - positive, 1.0)
    weights_row = np.where(labels == 1, len(labels) / (2 * positive), len(labels) / (2 * negative))
    weight = np.zeros(features.shape[1])
    bias = 0.0
    for _ in range(steps):
        predicted = _sigmoid(features @ weight + bias)
        error = (predicted - labels) * weights_row
        weight -= learning_rate * (features.T @ error / len(labels) + l2 * weight)
        bias -= learning_rate * float(error.mean())
    return weight, float(bias)


class EllipsisDetector:
    def __init__(self, weight, bias, threshold, feature_names=FEATURE_NAMES):
        self.weight = np.asarray(weight, dtype=float)
        self.bias = float(bias)
        self.threshold = float(threshold)
        self.feature_names = tuple(feature_names)
        if self.weight.shape != (len(self.feature_names),):
            raise ValueError("Detector weights do not match the frozen feature list.")

    def score_features(self, features):
        return float(_sigmoid(np.asarray(features, dtype=float) @ self.weight + self.bias))

    def decide_features(self, features):
        score = self.score_features(features)
        return dict(present=score >= self.threshold, score=score, threshold=self.threshold,
                    features={name: float(value) for name, value in zip(self.feature_names, features)})

    def decide(self, text, nlp=None):
        return self.decide_features(featurize(text, nlp))

    def to_json(self, metrics):
        return dict(
            schema_version="ellipsis-detector-1",
            citation="Cavar, Mompelat, and Abdo (2024), The Typology of Ellipsis. English THEC.",
            feature_names=list(self.feature_names),
            weights=self.weight.tolist(),
            bias=self.bias,
            threshold=self.threshold,
            split_salt=SPLIT_SALT,
            independent_of="parsing_game four-action relation CEM",
            proposes_missing_words=False,
            probe_not_used_for_fitting=True,
            metrics=metrics,
        )


def load_detector(path=None):
    payload = json.loads(Path(path or MODEL_PATH).read_text(encoding="utf-8"))
    if payload.get("schema_version") != "ellipsis-detector-1":
        raise ValueError("Unsupported ellipsis detector schema.")
    return EllipsisDetector(payload["weights"], payload["bias"], payload["threshold"], payload["feature_names"])


def _matrix(rows, nlp):
    docs = list(nlp.pipe((row["text"] for row in rows), batch_size=64))
    features = np.vstack([featurize_doc(doc) for doc in docs]) if docs else np.zeros((0, len(FEATURE_NAMES)))
    labels = np.asarray([row["label"] for row in rows], dtype=int)
    return features, labels


def train_and_save(root=None, path=None):
    nlp = get_nlp()
    rows = load_examples(root)
    splits = {name: rows_for_split(rows, name) for name in ("train", "dev", "test")}
    train_x, train_y = _matrix(splits["train"], nlp)
    weight, bias = fit_linear(train_x, train_y)
    dev_x, dev_y = _matrix(splits["dev"], nlp)
    dev_scores = _sigmoid(dev_x @ weight + bias)
    threshold = choose_threshold(dev_scores, dev_y, float(_sigmoid(bias)))
    detector = EllipsisDetector(weight, bias, threshold)
    metrics = {}
    for name in ("train", "dev", "test"):
        features, labels = (train_x, train_y) if name == "train" else _matrix(splits[name], nlp)
        if name == "dev":
            features, labels = dev_x, dev_y
        scores = _sigmoid(features @ detector.weight + detector.bias)
        metrics[name] = _metrics(labels, (scores >= detector.threshold).astype(int))
        metrics[name]["groups"] = len({row["group_id"] for row in splits[name]})
    destination = Path(path or MODEL_PATH)
    destination.write_text(json.dumps(detector.to_json(metrics), indent=2) + "\n", encoding="utf-8")
    return detector, metrics


def main():
    detector, metrics = train_and_save()
    print(json.dumps(dict(threshold=detector.threshold, weights=dict(zip(FEATURE_NAMES, detector.weight.tolist())),
                          bias=detector.bias, metrics=metrics), indent=2))


if __name__ == "__main__":
    main()
