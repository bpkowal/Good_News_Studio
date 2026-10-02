import gymnasium as gym
from gymnasium import spaces
import numpy as np
import re


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

    # --------------------------------------------------------
    # FIRST -> SECOND
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # SECOND -> FIRST
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # NO CAUSAL RELATION
    # --------------------------------------------------------

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
# ORDINARY HOLDOUT DATA
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
# ADVERSARIAL / ROBUSTNESS DATA
#
# These are intentionally designed to attack our assumptions.
# ============================================================

ROBUSTNESS_EXAMPLES = [

    # --------------------------------------------------------
    # ABSTENTION BEHAVIOR
    # --------------------------------------------------------
    {
    "family": "unresolved_relation",
    "sentence": "Rain accompanies flooding.",
    "entity1": "rain",
    "entity2": "flooding",
    "correct_action": NO_CAUSAL_RELATION,
    },


    # --------------------------------------------------------
    # NEGATION SCOPE
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # DISTANCE / IRRELEVANT INSERTIONS
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # "BY" HAS DIFFERENT STRUCTURAL ROLES
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # OUT-OF-VOCABULARY CAUSAL EXPRESSIONS
    #
    # These intentionally expose weakness in RELATION_WORDS.
    # --------------------------------------------------------

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
NEGATION_GENERALIZATION_EXAMPLES = [

    # --------------------------
    # ACTIVE POSITIVE / NEGATIVE
    # --------------------------

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


    # --------------------------
    # PASSIVE POSITIVE / NEGATIVE
    # --------------------------

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


    # --------------------------
    # NEGATION THAT SHOULD NOT
    # NEGATE THE RELATION
    # --------------------------

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
# WORD CLASSES
# ============================================================

RELATION_WORDS = {
    "cause",
    "causes",
    "caused",
    "trigger",
    "triggers",
    "triggered",
    "produce",
    "produces",
    "produced",
}


SOURCE_MARKERS = {
    "by",
    "from",
}


AUXILIARIES = {
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
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
# RELATION ESCALATION
#
# Stage 1:
#     RELATION_WORDS handles familiar relations cheaply.
#
# Stage 2:
#     If no familiar relation is found, preserve the
#     intervening predicate-like span as a candidate.
#
# Stage 3:
#     Try to classify that candidate semantically.
#
# This semantic classifier is intentionally small and
# replaceable. Later it could be replaced by a trained
# classifier, dependency parser, embedding model, or LLM.
# ============================================================


CANDIDATE_CAUSAL_PATTERNS = {

    # forward-looking constructions
    ("lead", "to"),
    ("leads", "to"),
    ("led", "to"),

    ("result", "in"),
    ("results", "in"),
    ("resulted", "in"),

    ("give", "rise", "to"),
    ("gives", "rise", "to"),
    ("gave", "rise", "to"),

    # reverse-looking constructions
    ("result", "from"),
    ("results", "from"),
    ("resulted", "from"),

    ("stem", "from"),
    ("stems", "from"),
    ("stemmed", "from"),
}


# These are relations, but should NOT establish a causal edge.
CANDIDATE_NONCAUSAL_PATTERNS = {

    ("correlates", "with"),
    ("correlated", "with"),

    ("associated", "with"),
}


# Words that by themselves are not enough to make an
# intervening region a useful predicate candidate.
RELATION_GLUE_WORDS = {

    "a",
    "an",
    "the",

    "to",
    "from",
    "by",
    "with",
    "of",

    "is",
    "are",
    "was",
    "were",

    "not",
    "no",

    "and",
    "or",
    "but",
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

    "no_relation_signal",

    "distance_entity1_relation",
    "distance_relation_entity2",

    "bias",
]


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize(sentence):

    return re.findall(
        r"[a-z]+",
        sentence.lower()
    )

def find_pattern(
    tokens,
    pattern
):

    pattern = list(pattern)

    pattern_length = len(
        pattern
    )


    for start in range(
        len(tokens)
        -
        pattern_length
        +
        1
    ):

        candidate = tokens[
            start:
            start + pattern_length
        ]


        if candidate == pattern:

            return start


    return None

def classify_candidate_relation(
    candidate_tokens
):

    # --------------------------------------------------------
    # Try causal patterns
    # --------------------------------------------------------

    for pattern in CANDIDATE_CAUSAL_PATTERNS:

        match_index = find_pattern(
            candidate_tokens,
            pattern
        )


        if match_index is not None:

            return {
                "semantic_class": "causal",
                "match_index": match_index,
                "matched_pattern": pattern,
            }


    # --------------------------------------------------------
    # Try explicitly noncausal patterns
    # --------------------------------------------------------

    for pattern in CANDIDATE_NONCAUSAL_PATTERNS:

        match_index = find_pattern(
            candidate_tokens,
            pattern
        )


        if match_index is not None:

            return {
                "semantic_class": "noncausal",
                "match_index": match_index,
                "matched_pattern": pattern,
            }


    # --------------------------------------------------------
    # We found a possible predicate region,
    # but we don't know what it means.
    # --------------------------------------------------------

    return {
        "semantic_class": "unknown",
        "match_index": None,
        "matched_pattern": None,
    }

def detect_relation(
    tokens,
    entity1_index,
    entity2_index
):

    # ========================================================
    # STAGE 1
    #
    # Cheap familiar relation detection.
    # ========================================================

    known_relation_indices = [
        i
        for i, token in enumerate(tokens)
        if token in RELATION_WORDS
    ]


    if known_relation_indices:

        return {
            "stage": "shortlist",
            "status": "resolved",
            "relation_index":
                known_relation_indices[0],
            "candidate_span": None,
            "semantic_class": "known",
            "matched_pattern": None,
        }


    # ========================================================
    # STAGE 2
    #
    # Preserve an unknown intervening region.
    #
    # We are NOT saying it is causal.
    # ========================================================

    candidate_start = (
        entity1_index
        +
        1
    )

    candidate_end = (
        entity2_index
    )


    candidate_tokens = tokens[
        candidate_start:
        candidate_end
    ]


    # Does this span contain anything more informative
    # than glue/function words?

    content_tokens = [
        token
        for token in candidate_tokens
        if token not in RELATION_GLUE_WORDS
    ]


    if not content_tokens:

        return {
            "stage": "none",
            "status": "no_candidate",
            "relation_index": None,
            "candidate_span": candidate_tokens,
            "semantic_class": "none",
            "matched_pattern": None,
        }


    # ========================================================
    # STAGE 3
    #
    # Ask what the candidate relation appears to mean.
    # ========================================================

    classification = (
        classify_candidate_relation(
            candidate_tokens
        )
    )


    semantic_class = (
        classification[
            "semantic_class"
        ]
    )


    # --------------------------------------------------------
    # Candidate is recognized as causal.
    #
    # Promote it to an actual relation candidate.
    # --------------------------------------------------------

    if semantic_class == "causal":

        relative_index = (
            classification[
                "match_index"
            ]
        )


        relation_index = (
            candidate_start
            +
            relative_index
        )


        return {
            "stage": "candidate_span",
            "status": "resolved",
            "relation_index":
                relation_index,
            "candidate_span":
                candidate_tokens,
            "semantic_class":
                semantic_class,
            "matched_pattern":
                classification[
                    "matched_pattern"
                ],
        }


    # --------------------------------------------------------
    # Candidate expresses a relation but not a causal one.
    # --------------------------------------------------------

    if semantic_class == "noncausal":

        return {
            "stage": "candidate_span",
            "status": "resolved_noncausal",
            "relation_index": None,
            "candidate_span":
                candidate_tokens,
            "semantic_class":
                semantic_class,
            "matched_pattern":
                classification[
                    "matched_pattern"
                ],
        }


    # --------------------------------------------------------
    # Something predicate-like exists, but we cannot
    # responsibly decide what relation it expresses.
    #
    # Preserve uncertainty.
    # --------------------------------------------------------

    return {
        "stage": "candidate_span",
        "status": "unresolved",
        "relation_index": None,
        "candidate_span":
            candidate_tokens,
        "semantic_class": "unknown",
        "matched_pattern": None,
    }


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_channels(example):

    sentence = example["sentence"]

    entity1 = example["entity1"].lower()
    entity2 = example["entity2"].lower()

    tokens = tokenize(sentence)

    n_tokens = len(tokens)


    # --------------------------------------------------------
    # ENTITY LOCATIONS
    # --------------------------------------------------------

    entity1_index = tokens.index(entity1)
    entity2_index = tokens.index(entity2)


    if entity1_index >= entity2_index:

        raise ValueError(
            f"Entity order problem in: {sentence}"
        )


# --------------------------------------------------------
# RELATION DETECTION WITH ESCALATION
# --------------------------------------------------------

    relation_analysis = detect_relation(
        tokens,
        entity1_index,
        entity2_index   
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
    # SOURCE MARKER LOCATIONS
    # --------------------------------------------------------

    marker_indices = [
        i
        for i, token in enumerate(tokens)
        if token in SOURCE_MARKERS
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

                for marker in marker_indices
            )
        )


        marker_between_relation_entity2 = float(
            any(
                relation_index
                <
                marker
                <
                entity2_index

                for marker in marker_indices
            )
        )


    marker_after_entity2 = float(
        any(
            marker > entity2_index
            for marker in marker_indices
        )
    )


    # --------------------------------------------------------
    # AUXILIARY LOCATION
    # --------------------------------------------------------

    auxiliary_indices = [
        i
        for i, token in enumerate(tokens)
        if token in AUXILIARIES
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

                for auxiliary in auxiliary_indices
            )
        )


    # --------------------------------------------------------
    # NEGATION TOPOLOGY
    # --------------------------------------------------------

    negation_indices = [
        i
        for i, token in enumerate(tokens)
        if token in NEGATION_WORDS
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

                for negation in negation_indices
            )
        )


        distances = [
            abs(
                negation
                -
                relation_index
            )

            for negation in negation_indices
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
    # EXPLICIT NO-RELATION SIGNAL
    # --------------------------------------------------------

    no_relation_signal = float(
        any(
            token in NO_RELATION_WORDS
            for token in tokens
        )
    )
    

# --------------------------------------------------------
# STRUCTURAL INTERACTION:
#
# Does negation appear to scope over the relation?
#
# This is NOT a new linguistic label supplied by us.
# It is constructed from already-observed relationships.
# --------------------------------------------------------

    negated_relation_structure = (
        relation_between_entities
        *
        negation_between_entity1_relation
        *
        negation_closeness_relation
        )
    
    # --------------------------------------------------------
    # DISTANCES
    # --------------------------------------------------------

    distance_entity1_relation = 0.0
    distance_relation_entity2 = 0.0


    if (
        relation_index is not None
        and
        n_tokens > 1
    ):

        distance_entity1_relation = (
            abs(
                relation_index
                -
                entity1_index
            )
            /
            (n_tokens - 1)
        )


        distance_relation_entity2 = (
            abs(
                entity2_index
                -
                relation_index
            )
            /
            (n_tokens - 1)
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

class TinyParsingEnv(gym.Env):

    def __init__(self, examples):

        super().__init__()

        self.examples = examples

        self.action_space = spaces.Discrete(3)

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(len(FEATURE_NAMES),),
            dtype=np.float32,
        )

        self.current_example = None


    def reset(
        self,
        seed=None,
        options=None
    ):

        super().reset(seed=seed)


        if (
            options is not None
            and
            "index" in options
        ):

            index = options["index"]

        else:

            index = int(
                self.np_random.integers(
                    len(self.examples)
                )
            )


        self.current_example = (
            self.examples[index]
        )


        observation = extract_channels(
            self.current_example
        )


        info = {
            "sentence":
                self.current_example["sentence"],

            "correct_action":
                self.current_example["correct_action"],
        }


        return observation, info


    def step(self, action):

        correct_action = (
            self.current_example[
                "correct_action"
            ]
        )


        reward = (
            1.0
            if action == correct_action
            else 0.0
        )


        terminated = True
        truncated = False


        observation = extract_channels(
            self.current_example
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


# ============================================================
# POLICY EVALUATION
# ============================================================

def evaluate_policy(
    env,
    weights
):

    correct = 0


    for index in range(
        len(env.examples)
    ):

        observation, info = env.reset(
            options={
                "index": index
            }
        )


        prediction = choose_action(
            observation,
            weights
        )


        if (
            prediction
            ==
            info["correct_action"]
        ):

            correct += 1


    return (
        correct
        /
        len(env.examples)
    )


# ============================================================
# EXACT REPRESENTATION COLLISION CHECK
# ============================================================

def find_representation_collisions(
    examples
):

    groups = {}


    for example in examples:

        observation = extract_channels(
            example
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


    collisions_found = False


    for group in groups.values():

        labels = {
            example["correct_action"]
            for example in group
        }


        if len(labels) > 1:

            collisions_found = True

            print(
                "\nREPRESENTATION COLLISION"
            )

            print("-" * 70)


            for example in group:

                print(
                    ACTION_NAMES[
                        example[
                            "correct_action"
                        ]
                    ],
                    ":",
                    example["sentence"]
                )


    if not collisions_found:

        print(
            "No contradictory exact "
            "representation collisions found."
        )


# ============================================================
# NEAR-COLLISION CHECK
#
# Looks for different labels that occupy very similar places
# in representation space.
# ============================================================

def find_near_collisions(
    examples,
    threshold=0.35
):

    observations = [
        extract_channels(example)
        for example in examples
    ]


    print(
        "\nNEAR-COLLISION CHECK"
    )

    print("=" * 70)


    found = False


    for i in range(
        len(examples)
    ):

        for j in range(
            i + 1,
            len(examples)
        ):

            # Only interesting if labels differ

            if (
                examples[i]["correct_action"]
                ==
                examples[j]["correct_action"]
            ):

                continue


            distance = np.linalg.norm(
                observations[i]
                -
                observations[j]
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
                    examples[i]["sentence"]
                )


                print(
                    ACTION_NAMES[
                        examples[j][
                            "correct_action"
                        ]
                    ],
                    ":",
                    examples[j]["sentence"]
                )


    if not found:

        print(
            "No opposite-label near-collisions "
            f"within distance {threshold}."
        )


# ============================================================
# REPRESENTATION AUDIT
#
# IMPORTANT: CHECK ALL SPLITS TOGETHER
# ============================================================

ALL_EXAMPLES = (
    TRAIN_EXAMPLES
    +
    TEST_EXAMPLES
    +
    ROBUSTNESS_EXAMPLES
    +
    NEGATION_GENERALIZATION_EXAMPLES
)


print(
    "\nCHECKING ALL REPRESENTATIONS TOGETHER"
)

print("=" * 70)


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


rng = np.random.default_rng(42)


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


    scores = np.array([
        evaluate_policy(
            train_env,
            candidate
        )

        for candidate in population
    ])


    elite_indices = np.argsort(
        scores
    )[
        -number_of_elites:
    ]


    elite_weights = population[
        elite_indices
    ]


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


    generation_best = scores[
        best_index
    ]


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
        f"Generation {generation + 1:2d}"
        f" | mean {np.mean(scores):.2f}"
        f" | elite {np.mean(scores[elite_indices]):.2f}"
        f" | best {generation_best:.2f}"
    )


train_env.close()


# ============================================================
# GENERAL TEST FUNCTION
#
# Used for both normal holdout and robustness data.
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

    print("=" * 70)


    correct = 0

    margins = []


    for index, example in enumerate(
        examples
    ):

        observation, info = env.reset(
            options={
                "index": index
            }
        )


        scores = (
            best_weights
            @
            observation
        )


        prediction = np.argmax(
            scores
        )


        sorted_scores = np.sort(
            scores
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
            example["correct_action"]
        ):

            correct += 1


        # ====================================================
        # SENTENCE INFORMATION
        # ====================================================

        print(
            f"\n{example['sentence']}"
        )


        if "family" in example:

            print(
                "Family:",
                example["family"]
            )


        # ====================================================
        # RELATION DETECTION DIAGNOSTIC
        # ====================================================

        tokens = tokenize(
            example["sentence"]
        )


        entity1_index = tokens.index(
            example[
                "entity1"
            ].lower()
        )


        entity2_index = tokens.index(
            example[
                "entity2"
            ].lower()
        )


        relation_analysis = detect_relation(
            tokens,
            entity1_index,
            entity2_index
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
            "  Matched pattern:",
            relation_analysis[
                "matched_pattern"
            ]
        )


        print(
            "  Relation index:",
            relation_analysis[
                "relation_index"
            ]
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
                        f"  {feature:34s}"
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
            f"Margin: {margin:.3f}"
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


    # ========================================================
    # TEST-SUITE SUMMARY
    # ========================================================

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
# NORMAL HOLDOUT
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


# ============================================================
# ROBUSTNESS / ADVERSARIAL SUITE
# ============================================================

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

print("=" * 70)


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