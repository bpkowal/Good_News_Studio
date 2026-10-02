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
# CONTRASTIVE TRAINING SET
# ============================================================

TRAIN_EXAMPLES = [

    # --------------------------------------------------------
    # FIRST -> SECOND
    # --------------------------------------------------------

    {
        "sentence": "Rain causes flooding.",
        "correct_action": FIRST_CAUSES_SECOND,
    },

    {
        "sentence": "Smoke triggers the alarm.",
        "correct_action": FIRST_CAUSES_SECOND,
    },

    {
        "sentence": "Heat produces expansion.",
        "correct_action": FIRST_CAUSES_SECOND,
    },

    {
        "sentence": "Hunger motivates eating.",
        "correct_action": FIRST_CAUSES_SECOND,
    },

    # "by" exists, but BEFORE the causal relation.
    {
        "sentence": "By noon rain caused flooding.",
        "correct_action": FIRST_CAUSES_SECOND,
    },

    {
        "sentence": "By chance smoke triggered the alarm.",
        "correct_action": FIRST_CAUSES_SECOND,
    },

    # "from" exists, but it is not marking reversed causality.
    {
        "sentence": "From experience heat produces expansion.",
        "correct_action": FIRST_CAUSES_SECOND,
    },

    # Contains "not", but "not only" does NOT negate causality.
    {
        "sentence": "Not only rain but snow causes flooding.",
        "correct_action": FIRST_CAUSES_SECOND,
    },


    # --------------------------------------------------------
    # SECOND -> FIRST
    # --------------------------------------------------------

    {
        "sentence": "Flooding was caused by rain.",
        "correct_action": SECOND_CAUSES_FIRST,
    },

    {
        "sentence": "The alarm was triggered by smoke.",
        "correct_action": SECOND_CAUSES_FIRST,
    },

    {
        "sentence": "Expansion was produced by heat.",
        "correct_action": SECOND_CAUSES_FIRST,
    },

    {
        "sentence": "Eating was motivated by hunger.",
        "correct_action": SECOND_CAUSES_FIRST,
    },

    {
        "sentence": "Panic was caused by flooding.",
        "correct_action": SECOND_CAUSES_FIRST,
    },

    {
        "sentence": "Fear was triggered by noise.",
        "correct_action": SECOND_CAUSES_FIRST,
    },


    # --------------------------------------------------------
    # NO CAUSAL RELATION
    # --------------------------------------------------------

    {
        "sentence": "Rain does not cause flooding.",
        "correct_action": NO_CAUSAL_RELATION,
    },

    {
        "sentence": "Smoke never triggers the alarm.",
        "correct_action": NO_CAUSAL_RELATION,
    },

    {
        "sentence": "Cats and bicycles are unrelated.",
        "correct_action": NO_CAUSAL_RELATION,
    },

    {
        "sentence": "Music and rainfall are independent.",
        "correct_action": NO_CAUSAL_RELATION,
    },

    # "from" occurs but means nothing causal here.
    {
        "sentence": "Music is independent from rainfall.",
        "correct_action": NO_CAUSAL_RELATION,
    },

    {
        "sentence": "Rain and gravity are not causal.",
        "correct_action": NO_CAUSAL_RELATION,
    },

]


# ============================================================
# HOLDOUT CONTRAST SET
#
# CEM NEVER TRAINS ON THESE
# ============================================================

TEST_EXAMPLES = [

    {
        "sentence": "Stress causes insomnia.",
        "correct_action": FIRST_CAUSES_SECOND,
    },

    {
        "sentence": "Insomnia was caused by stress.",
        "correct_action": SECOND_CAUSES_FIRST,
    },

    {
        "sentence": "Stress does not cause insomnia.",
        "correct_action": NO_CAUSAL_RELATION,
    },

    # "by" should NOT cause reversal here.
    {
        "sentence": "By midnight stress caused insomnia.",
        "correct_action": FIRST_CAUSES_SECOND,
    },

    # "not" should NOT mean negation here.
    {
        "sentence": "Not only stress but noise causes insomnia.",
        "correct_action": FIRST_CAUSES_SECOND,
    },

    {
        "sentence": "Fear was triggered by thunder.",
        "correct_action": SECOND_CAUSES_FIRST,
    },

    {
        "sentence": "Thunder never triggers fear.",
        "correct_action": NO_CAUSAL_RELATION,
    },

    {
        "sentence": "Music and insomnia are unrelated.",
        "correct_action": NO_CAUSAL_RELATION,
    },
]


# ============================================================
# INFORMATION CHANNELS
# ============================================================

FEATURE_NAMES = [
    "relation_cue",
    "source_marker",
    "source_before_relation",
    "source_after_relation",
    "aux_before_relation",
    "negation",
    "negation_near_relation",
    "not_only",
    "no_relation_signal",
    "relation_position",
    "bias",
]


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
    "motivate",
    "motivates",
    "motivated",
    "increase",
    "increases",
    "increased",
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


AUXILIARIES = {
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
}


NO_RELATION_WORDS = {
    "unrelated",
    "independent",
}


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_channels(sentence):

    tokens = re.findall(
        r"[a-z]+",
        sentence.lower()
    )

    n_tokens = len(tokens)


    # --------------------------------------------------------
    # Find relation cue
    # --------------------------------------------------------

    relation_indices = [
        i
        for i, token in enumerate(tokens)
        if token in RELATION_WORDS
    ]

    relation_cue = float(
        len(relation_indices) > 0
    )

    relation_index = (
        relation_indices[0]
        if relation_indices
        else None
    )


    # --------------------------------------------------------
    # Find source markers: "by", "from"
    # --------------------------------------------------------

    source_indices = [
        i
        for i, token in enumerate(tokens)
        if token in SOURCE_MARKERS
    ]

    source_marker = float(
        len(source_indices) > 0
    )


    # --------------------------------------------------------
    # Did the source marker occur BEFORE the relation?
    #
    # "BY noon rain CAUSED flooding"
    # --------------------------------------------------------

    source_before_relation = 0.0

    if relation_index is not None:

        if any(
            source_index < relation_index
            for source_index in source_indices
        ):
            source_before_relation = 1.0


    # --------------------------------------------------------
    # Did the source marker occur AFTER the relation?
    #
    # "Flooding was CAUSED BY rain"
    # --------------------------------------------------------

    source_after_relation = 0.0

    if relation_index is not None:

        if any(
            source_index > relation_index
            for source_index in source_indices
        ):
            source_after_relation = 1.0


    # --------------------------------------------------------
    # Auxiliary before relation
    #
    # "was caused"
    # "was triggered"
    #
    # We are deliberately NOT calling this "passive voice".
    # It is simply an observable positional relationship.
    # --------------------------------------------------------

    aux_before_relation = 0.0

    if relation_index is not None:

        preceding_tokens = tokens[
            max(0, relation_index - 2):
            relation_index
        ]

        if any(
            token in AUXILIARIES
            for token in preceding_tokens
        ):
            aux_before_relation = 1.0


    # --------------------------------------------------------
    # Any negation token anywhere?
    # --------------------------------------------------------

    negation_indices = [
        i
        for i, token in enumerate(tokens)
        if token in NEGATION_WORDS
    ]

    negation = float(
        len(negation_indices) > 0
    )


    # --------------------------------------------------------
    # Is negation close enough to plausibly affect
    # the causal relation?
    #
    # "does NOT CAUSE"
    # "NEVER TRIGGERS"
    # --------------------------------------------------------

    negation_near_relation = 0.0

    if relation_index is not None:

        for neg_index in negation_indices:

            distance = (
                relation_index
                -
                neg_index
            )

            if 0 < distance <= 2:
                negation_near_relation = 1.0


    # --------------------------------------------------------
    # Special contrast:
    #
    # "NOT ONLY rain ..."
    #
    # Contains "not", but doesn't negate the relation.
    # --------------------------------------------------------

    not_only = 0.0

    for i in range(
        len(tokens) - 1
    ):

        if (
            tokens[i] == "not"
            and
            tokens[i + 1] == "only"
        ):
            not_only = 1.0


    # --------------------------------------------------------
    # Explicit no-relation signals
    # --------------------------------------------------------

    no_relation_signal = float(
        any(
            token in NO_RELATION_WORDS
            for token in tokens
        )
    )


    # Detect phrases like:
    #
    # "not causal"
    # "no causal"
    # --------------------------------------------------------

    for i in range(
        len(tokens) - 1
    ):

        if (
            tokens[i] in {"not", "no"}
            and
            tokens[i + 1] == "causal"
        ):
            no_relation_signal = 1.0


    # --------------------------------------------------------
    # Relation position
    # --------------------------------------------------------

    if (
        relation_index is not None
        and
        n_tokens > 1
    ):

        relation_position = (
            relation_index
            /
            (n_tokens - 1)
        )

    else:

        relation_position = 0.0


    # --------------------------------------------------------
    # Bias
    # --------------------------------------------------------

    bias = 1.0


    return np.array(
        [
            relation_cue,
            source_marker,
            source_before_relation,
            source_after_relation,
            aux_before_relation,
            negation,
            negation_near_relation,
            not_only,
            no_relation_signal,
            relation_position,
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
            self.current_example[
                "sentence"
            ]
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
            self.current_example[
                "sentence"
            ]
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
# EVALUATE POLICY
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
# TRAINING ENVIRONMENT
# ============================================================

train_env = TinyParsingEnv(
    TRAIN_EXAMPLES
)


number_of_actions = 3

number_of_features = len(
    FEATURE_NAMES
)


# ============================================================
# CEM SETTINGS
# ============================================================

rng = np.random.default_rng(42)

population_size = 80

elite_fraction = 0.20

generations = 40

number_of_elites = int(
    population_size
    *
    elite_fraction
)


# ============================================================
# INITIAL SEARCH DISTRIBUTION
# ============================================================

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


# ============================================================
# CEM
# ============================================================

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
# LEARNED WEIGHTS
# ============================================================

print("\nLEARNED POLICY")
print("=" * 70)


for action in range(
    number_of_actions
):

    print(
        f"\n{ACTION_NAMES[action]}"
    )

    for i, feature in enumerate(
        FEATURE_NAMES
    ):

        print(
            f"{feature:26s}"
            f"{best_weights[action, i]: .3f}"
        )


# ============================================================
# HOLDOUT TEST
# ============================================================

test_env = TinyParsingEnv(
    TEST_EXAMPLES
)


print("\nHOLDOUT CONTRAST CASES")
print("=" * 70)


correct = 0


for index, example in enumerate(
    TEST_EXAMPLES
):

    observation, info = (
        test_env.reset(
            options={
                "index": index
            }
        )
    )


    scores = (
        best_weights
        @
        observation
    )


    prediction = np.argmax(
        scores
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
        dict(
            zip(
                FEATURE_NAMES,
                np.round(
                    observation,
                    2
                )
            )
        )
    )

    print(
        "Scores:"
    )

    for action in range(3):

        print(
            f"  "
            f"{ACTION_NAMES[action]:25s}"
            f"{scores[action]: .3f}"
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