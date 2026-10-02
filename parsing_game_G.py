import gymnasium as gym
from gymnasium import spaces
import numpy as np
import spacy


# ============================================================
# LOAD SPACY
# ============================================================

try:
    nlp = spacy.load("en_core_web_sm")

except OSError:

    raise RuntimeError(
        "\nspaCy is installed, but en_core_web_sm is missing.\n"
        "Run:\n\n"
        "python -m spacy download en_core_web_sm\n"
    )


# ============================================================
# ACTIONS
# ============================================================

FIRST_CAUSES_SECOND = 0
SECOND_CAUSES_FIRST = 1
NO_CAUSAL_RELATION = 2


ACTION_NAMES = {
    FIRST_CAUSES_SECOND: "FIRST causes SECOND",
    SECOND_CAUSES_FIRST: "SECOND causes FIRST",
    NO_CAUSAL_RELATION: "NO causal relationship",
}


# ============================================================
# TRAINING DATA
# ============================================================

TRAIN_EXAMPLES = [

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
        "correct_action": NO_CAUSAL_RELATION,
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


# ============================================================
# SEMANTIC KNOWLEDGE
#
# Notice the change:
#
# These are now LEMMAS, not every inflected surface form.
#
# causes / caused / causing -> cause
# triggers / triggered      -> trigger
# etc.
# ============================================================

KNOWN_RELATION_LEMMAS = {
    "cause",
    "trigger",
    "produce",
}


SOURCE_MARKERS = {
    "by",
    "from",
}


NEGATION_WORDS = {
    "not",
    "never",
    "no",
}


NO_RELATION_WORDS = {
    "unrelated",
    "independent",
}


# ============================================================
# FALLBACK SEMANTIC CLASSIFIER
#
# These patterns operate on spaCy LEMMAS.
#
# Therefore:
#
# leads to   -> lead to
# led to     -> lead to
#
# results from -> result from
# resulted from -> result from
# ============================================================

CANDIDATE_CAUSAL_PATTERNS = {

    ("lead", "to"),

    ("result", "in"),

    ("result", "from"),

    ("stem", "from"),

    ("give", "rise", "to"),
}


CANDIDATE_NONCAUSAL_PATTERNS = {

    ("correlate", "with"),

    ("associate", "with"),
}


# ============================================================
# INFORMATION CHANNELS
# ============================================================

FEATURE_NAMES = [

    "relation_present",
    "relation_between_entities",

    "marker_before_entity1",
    "marker_between_entity1_relation",
    "marker_between_relation_entity2",
    "marker_after_entity2",

    "aux_between_entity1_relation",

    "negation_between_entity1_relation",
    "negation_adjacent_relation",
    "negation_closeness_relation",

    "negated_relation_structure",
    "negated_reverse_structure",

    "no_relation_signal",

    "distance_entity1_relation",
    "distance_relation_entity2",

    "bias",
]


# ============================================================
# SPACY HELPERS
# ============================================================

def find_entity_token(
    doc,
    entity_text
):

    entity_text = entity_text.lower()


    for token in doc:

        if token.text.lower() == entity_text:

            return token


    raise ValueError(
        f"Could not find entity "
        f"'{entity_text}' in: {doc.text}"
    )


def find_pattern(
    sequence,
    pattern
):

    pattern = list(pattern)

    pattern_length = len(pattern)


    for start in range(
        len(sequence)
        -
        pattern_length
        +
        1
    ):

        candidate = sequence[
            start:
            start + pattern_length
        ]


        if candidate == pattern:

            return start


    return None


# ============================================================
# BUILD A RELATION PHRASE FROM A SPACY PREDICATE
#
# Example:
#
# Rain leads to flooding
#
# spaCy gives us the VERB "leads".
# We also collect an attached preposition "to".
#
# candidate lemmas:
#
# ["lead", "to"]
# ============================================================

def build_candidate_relation(
    predicate_token
):

    relation_tokens = [
        predicate_token
    ]


    # Attached particles/prepositions:
    #
    # result FROM
    # lead TO
    # etc.

    for child in predicate_token.children:

        if child.dep_ in {
            "prep",
            "prt",
        }:

            relation_tokens.append(
                child
            )


        # Special support for:
        #
        # give rise to
        #
        # "rise" may be an object/complement.

        if (
            child.lemma_.lower() == "rise"
            and
            child.dep_ in {
                "obj",
                "dobj",
                "attr",
                "oprd",
            }
        ):

            relation_tokens.append(
                child
            )


            for grandchild in child.children:

                if grandchild.dep_ in {
                    "prep",
                    "prt",
                }:

                    relation_tokens.append(
                        grandchild
                    )


    relation_tokens = sorted(
        relation_tokens,
        key=lambda token: token.i
    )


    relation_text = [
        token.text.lower()
        for token in relation_tokens
    ]


    relation_lemmas = [
        token.lemma_.lower()
        for token in relation_tokens
    ]


    return (
        relation_tokens,
        relation_text,
        relation_lemmas,
    )


# ============================================================
# SEMANTIC CLASSIFICATION OF UNKNOWN PREDICATE
# ============================================================

def classify_candidate_relation(
    candidate_lemmas
):

    # --------------------------------------------------------
    # Known causal semantic patterns
    # --------------------------------------------------------

    for pattern in CANDIDATE_CAUSAL_PATTERNS:

        match_index = find_pattern(
            candidate_lemmas,
            pattern
        )

        if match_index is not None:

            return {
                "semantic_class":
                    "causal",

                "matched_pattern":
                    pattern,

                "match_index":
                    match_index,
            }


    # --------------------------------------------------------
    # Known noncausal semantic patterns
    # --------------------------------------------------------

    for pattern in CANDIDATE_NONCAUSAL_PATTERNS:

        match_index = find_pattern(
            candidate_lemmas,
            pattern
        )

        if match_index is not None:

            return {
                "semantic_class":
                    "noncausal",

                "matched_pattern":
                    pattern,

                "match_index":
                    match_index,
            }


    # --------------------------------------------------------
    # No known semantic pattern
    # --------------------------------------------------------

    return {
        "semantic_class":
            "unknown",

        "matched_pattern":
            None,

        "match_index":
            None,
    }
# ============================================================
# RELATION DETECTION
#
# ESCALATION:
#
# 1. Known relation lemma
# 2. spaCy detects unknown predicate-like VERB
# 3. semantic fallback classifies candidate
# ============================================================

def detect_relation(
    doc,
    entity1_token,
    entity2_token
):

    entity1_index = (
        entity1_token.i
    )

    entity2_index = (
        entity2_token.i
    )


    # ========================================================
    # STAGE 1
    #
    # Familiar semantic relation.
    #
    # spaCy lemmatization means:
    #
    # cause
    # causes
    # caused
    #
    # all become:
    #
    # cause
    # ========================================================

    known_candidates = [

        token

        for token in doc

        if (
            token.lemma_.lower()
            in KNOWN_RELATION_LEMMAS
        )

        and

        token.pos_ == "VERB"
    ]


    # Prefer one between the target entities.

    between_known = [

        token

        for token in known_candidates

        if (
            entity1_index
            <
            token.i
            <
            entity2_index
        )
    ]


    if between_known:

        relation_token = (
            between_known[0]
        )


    elif known_candidates:

        relation_token = (
            known_candidates[0]
        )


    else:

        relation_token = None


    if relation_token is not None:

        return {

            "stage":
                "shortlist",

            "status":
                "resolved",

            "semantic_class":
                "known",

            "relation_index":
                relation_token.i,

            "relation_text":
                relation_token.text,

            "relation_lemma":
                relation_token.lemma_,

            "relation_pos":
                relation_token.pos_,

            "relation_dep":
                relation_token.dep_,

            "relation_head":
                relation_token.head.text,

            "candidate_span":
                None,

            "candidate_lemmas":
                None,

            "matched_pattern":
                None,
        }

    # ========================================================
    # STAGE 2A
    #
    # Search the ENTIRE intervening lemma span for a semantic
    # relation pattern BEFORE trusting spaCy's POS tagging.
    #
    # Example:
    #
    # Flooding results from rain.
    #
    # Even if spaCy incorrectly tags "results",
    # the intervening lemmas may still be:
    #
    # ["result", "from"]
    #
    # and our semantic matcher can recover the relation.
    # ========================================================

    intervening_tokens = [

        token

        for token in doc

        if (
            entity1_index
            <
            token.i
            <
            entity2_index
        )

        and

        not token.is_punct
    ]


    intervening_text = [
        token.text.lower()
        for token in intervening_tokens
    ]


    intervening_lemmas = [
        token.lemma_.lower()
        for token in intervening_tokens
    ]


    span_classification = (
        classify_candidate_relation(
            intervening_lemmas
        )
    )


    span_semantic_class = (
        span_classification[
            "semantic_class"
        ]
    )


    # --------------------------------------------------------
    # Known causal pattern found in the span.
    #
    # No VERB tag is required.
    # --------------------------------------------------------

    if span_semantic_class == "causal":

        match_index = (
            span_classification[
                "match_index"
            ]
        )


        matched_token = (
            intervening_tokens[
                match_index
            ]
        )


        return {

            "stage":
                "semantic_span",

            "status":
                "resolved",

            "semantic_class":
                "causal",

            "relation_index":
                matched_token.i,

            "relation_text":
                matched_token.text,

            "relation_lemma":
                matched_token.lemma_,

            "relation_pos":
                matched_token.pos_,

            "relation_dep":
                matched_token.dep_,

            "relation_head":
                matched_token.head.text,

            "candidate_span":
                intervening_text,

            "candidate_lemmas":
                intervening_lemmas,

            "matched_pattern":
                span_classification[
                    "matched_pattern"
                ],
        }


    # --------------------------------------------------------
    # Known NONCAUSAL relation found.
    #
    # Again, POS status does not matter.
    # --------------------------------------------------------

    if span_semantic_class == "noncausal":

        match_index = (
            span_classification[
                "match_index"
            ]
        )


        matched_token = (
            intervening_tokens[
                match_index
            ]
        )


        return {

            "stage":
                "semantic_span",

            "status":
                "resolved_noncausal",

            "semantic_class":
                "noncausal",

            "relation_index":
                None,

            "relation_text":
                matched_token.text,

            "relation_lemma":
                matched_token.lemma_,

            "relation_pos":
                matched_token.pos_,

            "relation_dep":
                matched_token.dep_,

            "relation_head":
                matched_token.head.text,

            "candidate_span":
                intervening_text,

            "candidate_lemmas":
                intervening_lemmas,

            "matched_pattern":
                span_classification[
                    "matched_pattern"
                ],
        }


    # ========================================================
    # STAGE 2
    #
    # Ask spaCy for an intervening predicate-like region.
    #
    # IMPORTANT:
    #
    # VERB does NOT mean causal.
    #
    # It merely becomes a candidate for semantic analysis.
    # ========================================================

    predicate_candidates = [

        token

        for token in doc

        if (
            entity1_index
            <
            token.i
            <
            entity2_index
        )

        and

        token.pos_ == "VERB"
    ]


    if not predicate_candidates:

        return {

            "stage":
                "none",

            "status":
                "no_candidate",

            "semantic_class":
                "none",

            "relation_index":
                None,

            "relation_text":
                None,

            "relation_lemma":
                None,

            "relation_pos":
                None,

            "relation_dep":
                None,

            "relation_head":
                None,

            "candidate_span":
                None,

            "candidate_lemmas":
                None,

            "matched_pattern":
                None,
        }


    # For now choose the first intervening VERB.
    #
    # Later this is an obvious place to improve using
    # dependency paths between E1 and E2.

    predicate_token = (
        predicate_candidates[0]
    )


    (
        relation_tokens,
        candidate_span,
        candidate_lemmas,

    ) = build_candidate_relation(
        predicate_token
    )


    # ========================================================
    # STAGE 3
    #
    # Classify candidate semantics.
    # ========================================================

    classification = (
        classify_candidate_relation(
            candidate_lemmas
        )
    )


    semantic_class = (
        classification[
            "semantic_class"
        ]
    )


    if semantic_class == "causal":

        return {

            "stage":
                "spacy_candidate",

            "status":
                "resolved",

            "semantic_class":
                "causal",

            "relation_index":
                predicate_token.i,

            "relation_text":
                predicate_token.text,

            "relation_lemma":
                predicate_token.lemma_,

            "relation_pos":
                predicate_token.pos_,

            "relation_dep":
                predicate_token.dep_,

            "relation_head":
                predicate_token.head.text,

            "candidate_span":
                candidate_span,

            "candidate_lemmas":
                candidate_lemmas,

            "matched_pattern":
                classification[
                    "matched_pattern"
                ],
        }


    if semantic_class == "noncausal":

        return {

            "stage":
                "spacy_candidate",

            "status":
                "resolved_noncausal",

            "semantic_class":
                "noncausal",

            "relation_index":
                None,

            "relation_text":
                predicate_token.text,

            "relation_lemma":
                predicate_token.lemma_,

            "relation_pos":
                predicate_token.pos_,

            "relation_dep":
                predicate_token.dep_,

            "relation_head":
                predicate_token.head.text,

            "candidate_span":
                candidate_span,

            "candidate_lemmas":
                candidate_lemmas,

            "matched_pattern":
                classification[
                    "matched_pattern"
                ],
        }


    # --------------------------------------------------------
    # Predicate exists.
    #
    # spaCy successfully found it.
    #
    # But our semantic system does not know what it means.
    # --------------------------------------------------------

    return {

        "stage":
            "spacy_candidate",

        "status":
            "unresolved",

        "semantic_class":
            "unknown",

        "relation_index":
            None,

        "relation_text":
            predicate_token.text,

        "relation_lemma":
            predicate_token.lemma_,

        "relation_pos":
            predicate_token.pos_,

        "relation_dep":
            predicate_token.dep_,

        "relation_head":
            predicate_token.head.text,

        "candidate_span":
            candidate_span,

        "candidate_lemmas":
            candidate_lemmas,

        "matched_pattern":
            None,
    }


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_channels(
    example
):

    sentence = (
        example["sentence"]
    )


    doc = nlp(
        sentence
    )


    entity1_token = find_entity_token(
        doc,
        example["entity1"]
    )


    entity2_token = find_entity_token(
        doc,
        example["entity2"]
    )


    entity1_index = (
        entity1_token.i
    )

    entity2_index = (
        entity2_token.i
    )


    if (
        entity1_index
        >=
        entity2_index
    ):

        raise ValueError(
            f"Entity order problem in: "
            f"{sentence}"
        )


    n_tokens = len(
        doc
    )


    # --------------------------------------------------------
    # RELATION DETECTION
    # --------------------------------------------------------

    relation_analysis = detect_relation(
        doc,
        entity1_token,
        entity2_token
    )


    relation_index = (
        relation_analysis[
            "relation_index"
        ]
    )


    relation_present = float(
        relation_index is not None
    )


    relation_between_entities = 0.0


    if relation_index is not None:

        relation_between_entities = float(
            entity1_index
            <
            relation_index
            <
            entity2_index
        )


    # --------------------------------------------------------
    # SOURCE MARKERS
    # --------------------------------------------------------

    marker_indices = [

        token.i

        for token in doc

        if token.text.lower()
        in SOURCE_MARKERS
    ]


    marker_before_entity1 = float(
        any(
            marker < entity1_index
            for marker in marker_indices
        )
    )


    marker_between_entity1_relation = 0.0

    marker_between_relation_entity2 = 0.0


    if relation_index is not None:

        marker_between_entity1_relation = float(
            any(
                entity1_index
                <
                marker
                <
                relation_index

                for marker
                in marker_indices
            )
        )


        marker_between_relation_entity2 = float(
            any(
                relation_index
                <
                marker
                <
                entity2_index

                for marker
                in marker_indices
            )
        )


    marker_after_entity2 = float(
        any(
            marker > entity2_index
            for marker in marker_indices
        )
    )


    # --------------------------------------------------------
    # SPACY AUXILIARY DETECTION
    #
    # No hand-written list required.
    # --------------------------------------------------------

    auxiliary_indices = [

        token.i

        for token in doc

        if token.pos_ == "AUX"
    ]


    aux_between_entity1_relation = 0.0


    if relation_index is not None:

        aux_between_entity1_relation = float(
            any(
                entity1_index
                <
                auxiliary
                <
                relation_index

                for auxiliary
                in auxiliary_indices
            )
        )


    # --------------------------------------------------------
    # NEGATION
    #
    # spaCy dependency label "neg" is now another source
    # of evidence in addition to explicit negation words.
    # --------------------------------------------------------

    negation_indices = [

        token.i

        for token in doc

        if (
            token.dep_ == "neg"
            or
            token.text.lower()
            in NEGATION_WORDS
        )
    ]


    negation_between_entity1_relation = 0.0

    negation_adjacent_relation = 0.0

    negation_closeness_relation = 0.0


    if (
        relation_index is not None
        and
        negation_indices
    ):

        negation_between_entity1_relation = float(
            any(
                entity1_index
                <
                negation
                <
                relation_index

                for negation
                in negation_indices
            )
        )


        distances = [

            abs(
                negation
                -
                relation_index
            )

            for negation
            in negation_indices
        ]


        minimum_distance = min(
            distances
        )


        negation_adjacent_relation = float(
            minimum_distance == 1
        )


        if minimum_distance > 0:

            negation_closeness_relation = (
                1.0
                /
                minimum_distance
            )


    # --------------------------------------------------------
    # STRUCTURAL NEGATION INTERACTION
    # --------------------------------------------------------

    negated_relation_structure = (
        relation_between_entities
        *
        negation_between_entity1_relation
        *
        negation_closeness_relation
    )

    # --------------------------------------------------------
    # INTERACTION:
    #
    # Negation applied to a structurally reversed relation.
    #
    # Example:
    #
    # Flooding was not caused by rain.
    #
    # NEGATION scopes over the relation
    # AND
    # the source marker sits between RELATION and ENTITY 2.
    # --------------------------------------------------------

    negated_reverse_structure = (
        negated_relation_structure
        *
        marker_between_relation_entity2
    )

    # --------------------------------------------------------
    # EXPLICIT NO-RELATION LANGUAGE
    # --------------------------------------------------------

    no_relation_signal = float(
        any(
            token.lemma_.lower()
            in NO_RELATION_WORDS

            for token in doc
        )
    )


    # --------------------------------------------------------
    # DISTANCES
    #
    # Use lexical token positions only.
    # Punctuation should not change normalized structural distance.
    # --------------------------------------------------------

    lexical_tokens = [
        token
        for token in doc
        if not token.is_punct
    ]

    lexical_position = {
        token.i: position
        for position, token in enumerate(
            lexical_tokens
        )
    }

    n_lexical_tokens = len(
        lexical_tokens
    )

    distance_entity1_relation = 0.0
    distance_relation_entity2 = 0.0

    if (
        relation_index is not None
        and
        n_lexical_tokens > 1
    ):

        distance_entity1_relation = (
            abs(
                lexical_position[
                    relation_index
                ]
                -
                lexical_position[
                    entity1_index
                ]
            )
            /
            (n_lexical_tokens - 1)
        )

        distance_relation_entity2 = (
            abs(
                lexical_position[
                    entity2_index
                ]
                -
                lexical_position[
                    relation_index
                ]
            )
            /
            (n_lexical_tokens - 1)
        )

    bias = 1.0

    return np.array(
        [
            relation_present,
            relation_between_entities,

            marker_before_entity1,
            marker_between_entity1_relation,
            marker_between_relation_entity2,
            marker_after_entity2,

            aux_between_entity1_relation,

            negation_between_entity1_relation,
            negation_adjacent_relation,
            negation_closeness_relation,

            negated_relation_structure,
            negated_reverse_structure,

            no_relation_signal,

            distance_entity1_relation,
            distance_relation_entity2,

            bias,
        ],
        dtype=np.float32,
    )
# ============================================================
# ENVIRONMENT
# ============================================================

class TinyParsingEnv(
    gym.Env
):

    def __init__(
        self,
        examples
    ):

        super().__init__()

        self.examples = (
            examples
        )


        self.action_space = (
            spaces.Discrete(3)
        )


        self.observation_space = (
            spaces.Box(
                low=0.0,
                high=1.0,
                shape=(
                    len(
                        FEATURE_NAMES
                    ),
                ),
                dtype=np.float32,
            )
        )


        self.current_example = (
            None
        )


    def reset(
        self,
        seed=None,
        options=None
    ):

        super().reset(
            seed=seed
        )


        if (
            options is not None
            and
            "index" in options
        ):

            index = (
                options["index"]
            )


        else:

            index = int(
                self.np_random.integers(
                    len(
                        self.examples
                    )
                )
            )


        self.current_example = (
            self.examples[
                index
            ]
        )


        observation = (
            extract_channels(
                self.current_example
            )
        )


        info = {

            "sentence":
                self.current_example[
                    "sentence"
                ],

            "correct_action":
                self.current_example[
                    "correct_action"
                ],
        }


        return (
            observation,
            info
        )


    def step(
        self,
        action
    ):

        correct_action = (
            self.current_example[
                "correct_action"
            ]
        )


        reward = (
            1.0
            if action
            ==
            correct_action
            else 0.0
        )


        terminated = True
        truncated = False


        observation = (
            extract_channels(
                self.current_example
            )
        )


        info = {
            "correct_action":
                correct_action
        }


        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )


# ============================================================
# POLICY
# ============================================================

def choose_action(
    observation,
    weights
):

    scores = (
        weights
        @
        observation
    )


    return np.argmax(
        scores
    )


def evaluate_policy(
    env,
    weights
):

    correct = 0


    for index in range(
        len(
            env.examples
        )
    ):

        observation, info = (
            env.reset(
                options={
                    "index":
                        index
                }
            )
        )


        prediction = (
            choose_action(
                observation,
                weights
            )
        )


        if (
            prediction
            ==
            info[
                "correct_action"
            ]
        ):

            correct += 1


    return (
        correct
        /
        len(
            env.examples
        )
    )


# ============================================================
# COLLISION TESTS
# ============================================================

def find_representation_collisions(
    examples
):

    groups = {}


    for example in examples:

        observation = (
            extract_channels(
                example
            )
        )


        key = tuple(
            np.round(
                observation,
                6
            )
        )


        groups.setdefault(
            key,
            []
        ).append(
            example
        )


    collisions_found = (
        False
    )


    for group in groups.values():

        labels = {
            example[
                "correct_action"
            ]

            for example
            in group
        }


        if len(labels) > 1:

            collisions_found = (
                True
            )


            print(
                "\nREPRESENTATION COLLISION"
            )

            print(
                "-" * 70
            )


            for example in group:

                print(
                    ACTION_NAMES[
                        example[
                            "correct_action"
                        ]
                    ],
                    ":",
                    example[
                        "sentence"
                    ]
                )


    if not collisions_found:

        print(
            "No contradictory exact "
            "representation collisions found."
        )


def find_near_collisions(
    examples,
    threshold=0.35
):

    observations = [

        extract_channels(
            example
        )

        for example
        in examples
    ]


    print(
        "\nNEAR-COLLISION CHECK"
    )

    print(
        "=" * 70
    )


    found = False


    for i in range(
        len(examples)
    ):

        for j in range(
            i + 1,
            len(examples)
        ):

            if (
                examples[i][
                    "correct_action"
                ]
                ==
                examples[j][
                    "correct_action"
                ]
            ):

                continue


            distance = (
                np.linalg.norm(
                    observations[i]
                    -
                    observations[j]
                )
            )


            if distance <= threshold:

                found = True


                print(
                    f"\nRepresentation distance: "
                    f"{distance:.3f}"
                )


                print(
                    ACTION_NAMES[
                        examples[i][
                            "correct_action"
                        ]
                    ],
                    ":",
                    examples[i][
                        "sentence"
                    ]
                )


                print(
                    ACTION_NAMES[
                        examples[j][
                            "correct_action"
                        ]
                    ],
                    ":",
                    examples[j][
                        "sentence"
                    ]
                )


    if not found:

        print(
            "No opposite-label near-collisions "
            f"within distance {threshold}."
        )


# ============================================================
# REPRESENTATION AUDIT
# ============================================================

ALL_EXAMPLES = (
    TRAIN_EXAMPLES
    +
    TEST_EXAMPLES
    +
    NEGATION_GENERALIZATION_EXAMPLES
    +
    ROBUSTNESS_EXAMPLES
)


print(
    "\nCHECKING ALL REPRESENTATIONS TOGETHER"
)

print(
    "=" * 70
)


find_representation_collisions(
    ALL_EXAMPLES
)


find_near_collisions(
    ALL_EXAMPLES,
    threshold=0.35
)


# ============================================================
# CEM TRAINING
# ============================================================

train_env = TinyParsingEnv(
    TRAIN_EXAMPLES
)


number_of_actions = 3

number_of_features = len(
    FEATURE_NAMES
)


rng = (
    np.random.default_rng(
        42
    )
)


population_size = 100
elite_fraction = 0.20
generations = 40


number_of_elites = int(
    population_size
    *
    elite_fraction
)


mean_weights = np.zeros(
    (
        number_of_actions,
        number_of_features
    )
)


std_weights = np.ones(
    (
        number_of_actions,
        number_of_features
    )
)


best_weights = None

best_accuracy = -np.inf


for generation in range(
    generations
):

    population = rng.normal(
        loc=mean_weights,
        scale=std_weights,
        size=(
            population_size,
            number_of_actions,
            number_of_features
        )
    )


    scores = np.array(
        [
            evaluate_policy(
                train_env,
                candidate
            )

            for candidate
            in population
        ]
    )


    elite_indices = (
        np.argsort(
            scores
        )[
            -number_of_elites:
        ]
    )


    elite_weights = (
        population[
            elite_indices
        ]
    )


    mean_weights = np.mean(
        elite_weights,
        axis=0
    )


    std_weights = np.std(
        elite_weights,
        axis=0
    )


    std_weights = np.maximum(
        std_weights,
        0.05
    )


    best_index = np.argmax(
        scores
    )


    generation_best = (
        scores[
            best_index
        ]
    )


    if (
        generation_best
        >
        best_accuracy
    ):

        best_accuracy = (
            generation_best
        )


        best_weights = (
            population[
                best_index
            ].copy()
        )


    print(
        f"Generation "
        f"{generation + 1:2d}"
        f" | mean "
        f"{np.mean(scores):.2f}"
        f" | elite "
        f"{np.mean(scores[elite_indices]):.2f}"
        f" | best "
        f"{generation_best:.2f}"
    )


train_env.close()


# ============================================================
# TEST SUITE
# ============================================================

def run_test_suite(
    title,
    examples,
    best_weights,
    show_channels=True
):

    env = TinyParsingEnv(
        examples
    )


    print(
        f"\n\n{title}"
    )

    print(
        "=" * 70
    )


    correct = 0

    margins = []


    for index, example in enumerate(
        examples
    ):

        observation, info = (
            env.reset(
                options={
                    "index":
                        index
                }
            )
        )


        scores = (
            best_weights
            @
            observation
        )


        prediction = (
            np.argmax(
                scores
            )
        )


        sorted_scores = (
            np.sort(
                scores
            )
        )


        margin = (
            sorted_scores[-1]
            -
            sorted_scores[-2]
        )


        margins.append(
            margin
        )


        if (
            prediction
            ==
            example[
                "correct_action"
            ]
        ):

            correct += 1


        print(
            f"\n"
            f"{example['sentence']}"
        )


        if "family" in example:

            print(
                "Family:",
                example[
                    "family"
                ]
            )


        # ====================================================
        # SPACY + RELATION DIAGNOSTICS
        # ====================================================

        doc = nlp(
            example[
                "sentence"
            ]
        )


        entity1_token = (
            find_entity_token(
                doc,
                example[
                    "entity1"
                ]
            )
        )


        entity2_token = (
            find_entity_token(
                doc,
                example[
                    "entity2"
                ]
            )
        )


        relation_analysis = (
            detect_relation(
                doc,
                entity1_token,
                entity2_token
            )
        )


        print(
            "\nRelation detection:"
        )


        print(
            "  Stage:",
            relation_analysis[
                "stage"
            ]
        )


        print(
            "  Status:",
            relation_analysis[
                "status"
            ]
        )


        print(
            "  Semantic class:",
            relation_analysis[
                "semantic_class"
            ]
        )


        print(
            "  Candidate span:",
            relation_analysis[
                "candidate_span"
            ]
        )


        print(
            "  Candidate lemmas:",
            relation_analysis[
                "candidate_lemmas"
            ]
        )


        print(
            "  Matched pattern:",
            relation_analysis[
                "matched_pattern"
            ]
        )


        print(
            "  Relation token:",
            relation_analysis[
                "relation_text"
            ]
        )


        print(
            "  Relation lemma:",
            relation_analysis[
                "relation_lemma"
            ]
        )


        print(
            "  Relation POS:",
            relation_analysis[
                "relation_pos"
            ]
        )


        print(
            "  Relation dependency:",
            relation_analysis[
                "relation_dep"
            ]
        )


        print(
            "  Relation head:",
            relation_analysis[
                "relation_head"
            ]
        )


        # ----------------------------------------------------
        # ENTITY DEPENDENCY STRUCTURE
        # ----------------------------------------------------

        print(
            "\nspaCy entity structure:"
        )


        print(
            f"  Entity 1: "
            f"{entity1_token.text}"
            f" | lemma={entity1_token.lemma_}"
            f" | POS={entity1_token.pos_}"
            f" | dep={entity1_token.dep_}"
            f" | head={entity1_token.head.text}"
        )


        print(
            f"  Entity 2: "
            f"{entity2_token.text}"
            f" | lemma={entity2_token.lemma_}"
            f" | POS={entity2_token.pos_}"
            f" | dep={entity2_token.dep_}"
            f" | head={entity2_token.head.text}"
        )


        # ====================================================
        # STRUCTURAL CHANNELS
        # ====================================================

        if show_channels:

            print(
                "\nStructural channels:"
            )


            for feature, value in zip(
                FEATURE_NAMES,
                observation
            ):

                if value != 0:

                    print(
                        f"  "
                        f"{feature:34s}"
                        f"{value:.2f}"
                    )


        # ====================================================
        # POLICY SCORES
        # ====================================================

        print(
            "\nScores:"
        )


        for action in range(3):

            print(
                f"  "
                f"{ACTION_NAMES[action]:25s}"
                f"{scores[action]: .3f}"
            )


        print(
            f"Margin: "
            f"{margin:.3f}"
        )


        print(
            "Prediction:",
            ACTION_NAMES[
                prediction
            ]
        )


        print(
            "Correct:   ",
            ACTION_NAMES[
                example[
                    "correct_action"
                ]
            ]
        )


    accuracy = (
        correct
        /
        len(examples)
    )


    print(
        f"\nAccuracy: "
        f"{accuracy:.3f}"
    )


    print(
        f"Mean decision margin: "
        f"{np.mean(margins):.3f}"
    )


    print(
        f"Minimum decision margin: "
        f"{np.min(margins):.3f}"
    )


    env.close()


    return accuracy


# ============================================================
# RUN TESTS
# ============================================================

holdout_accuracy = run_test_suite(
    "ORDINARY HOLDOUT RESULTS",
    TEST_EXAMPLES,
    best_weights,
)


negation_accuracy = run_test_suite(
    "NEGATION GENERALIZATION RESULTS",
    NEGATION_GENERALIZATION_EXAMPLES,
    best_weights,
)


robustness_accuracy = run_test_suite(
    "ROBUSTNESS RESULTS",
    ROBUSTNESS_EXAMPLES,
    best_weights,
)


# ============================================================
# SUMMARY
# ============================================================

print(
    "\n\nSUMMARY"
)

print(
    "=" * 70
)


print(
    f"Training best accuracy: "
    f"{best_accuracy:.3f}"
)


print(
    f"Ordinary holdout accuracy: "
    f"{holdout_accuracy:.3f}"
)


print(
    f"Negation generalization accuracy: "
    f"{negation_accuracy:.3f}"
)


print(
    f"Robustness accuracy: "
    f"{robustness_accuracy:.3f}"
)