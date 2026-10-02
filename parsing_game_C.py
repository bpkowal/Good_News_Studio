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
#
# For now we explicitly provide entity1 and entity2.
#
# This deliberately isolates:
#
#     STRUCTURAL INTERPRETATION
#
# from:
#
#     ENTITY EXTRACTION
#
# We will remove this assistance later.
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

    # "by" occurs BEFORE the entities
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

    # "by" occurs AFTER entity2: mechanism, not source
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

    # "not" exists but does NOT negate the causal relation
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

    # No auxiliary "was": tests topology rather than phrase template
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
    # NO RELATION
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
# HOLDOUT DATA
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
# WORD CLASSES
#
# These still provide candidate components,
# but their mere presence is no longer the main representation.
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
# INFORMATION CHANNELS
#
# Notice how these increasingly describe RELATIONS
# among components rather than isolated token properties.
# ============================================================

FEATURE_NAMES = [

    "relation_present",

    # Relation topology
    "relation_between_entities",

    # Source-marker topology
    "marker_before_entity1",
    "marker_between_entity1_relation",
    "marker_between_relation_entity2",
    "marker_after_entity2",

    # Auxiliary topology
    "aux_between_entity1_relation",

    # Negation topology
    "negation_between_entity1_relation",
    "negation_adjacent_relation",
    "negation_closeness_relation",

    # Explicit semantic signal
    "no_relation_signal",

    # Relative distances
    "distance_entity1_relation",
    "distance_relation_entity2",

    # Constant
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
    # Locate candidate entities
    # --------------------------------------------------------

    entity1_index = tokens.index(entity1)
    entity2_index = tokens.index(entity2)


    # Because our actions mean FIRST-mentioned and SECOND-mentioned,
    # these examples should preserve surface order.

    if entity1_index >= entity2_index:
        raise ValueError(
            f"Entity order problem in: {sentence}"
        )


    # --------------------------------------------------------
    # Locate relation cue
    # --------------------------------------------------------

    relation_indices = [
        i
        for i, token in enumerate(tokens)
        if token in RELATION_WORDS
    ]

    relation_index = (
        relation_indices[0]
        if relation_indices
        else None
    )

    relation_present = float(
        relation_index is not None
    )


    # --------------------------------------------------------
    # Relation BETWEEN the candidate entities?
    # --------------------------------------------------------

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
    # Locate source markers
    # --------------------------------------------------------

    marker_indices = [
        i
        for i, token in enumerate(tokens)
        if token in SOURCE_MARKERS
    ]


    # --------------------------------------------------------
    # MARKER TOPOLOGY
    # --------------------------------------------------------

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
    # AUXILIARY TOPOLOGY
    #
    # Flooding WAS CAUSED by rain
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
                aux
                <
                relation_index

                for aux in auxiliary_indices
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


        # Adjacent negation -> 1.0
        # two tokens away   -> 0.5
        # three away        -> 0.33 ...

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
    # DISTANCES
    #
    # Normalize by sentence length so values remain 0-1.
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


    observation = np.array(
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

            no_relation_signal,

            distance_entity1_relation,
            distance_relation_entity2,

            bias,
        ],
        dtype=np.float32,
    )


    return observation


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

    return np.argmax(scores)


# ============================================================
# EVALUATION
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
# REPRESENTATION COLLISION CHECK
#
# VERY IMPORTANT:
#
# If two sentences with DIFFERENT correct answers produce
# identical observations, then the representation has destroyed
# information needed to solve the task.
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


        if key not in groups:
            groups[key] = []


        groups[key].append(
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

            print("-" * 60)


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
            "\nNo contradictory representation "
            "collisions found."
        )


# ============================================================
# CHECK REPRESENTATION FIRST
# ============================================================

print(
    "\nCHECKING TRAINING REPRESENTATION"
)

find_representation_collisions(
    TRAIN_EXAMPLES
)


print(
    "\nCHECKING HOLDOUT REPRESENTATION"
)

find_representation_collisions(
    TEST_EXAMPLES
)


# ============================================================
# CEM
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
# HOLDOUT TEST
# ============================================================

test_env = TinyParsingEnv(
    TEST_EXAMPLES
)


print(
    "\n\nHOLDOUT RESULTS"
)

print("=" * 70)


correct = 0


for index, example in enumerate(
    TEST_EXAMPLES
):

    observation, info = test_env.reset(
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


    if (
        prediction
        ==
        example["correct_action"]
    ):

        correct += 1


    print(
        f"\n{example['sentence']}"
    )


    print(
        "Structural channels:"
    )


    for feature, value in zip(
        FEATURE_NAMES,
        observation
    ):

        if value != 0:

            print(
                f"  {feature:32s}"
                f"{value:.2f}"
            )


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


holdout_accuracy = (
    correct
    /
    len(TEST_EXAMPLES)
)


print(
    f"\nHoldout accuracy: "
    f"{holdout_accuracy:.2f}"
)


test_env.close()