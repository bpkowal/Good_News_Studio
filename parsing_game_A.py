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
    NO_CAUSAL_RELATION: "NO causal relationship"
}


# ============================================================
# TRAINING EXAMPLES
# ============================================================

TRAIN_EXAMPLES = [

    # First entity causes second entity

    {
        "sentence": "Exercise increases mood.",
        "correct_action": FIRST_CAUSES_SECOND
    },

    {
        "sentence": "Rain causes flooding.",
        "correct_action": FIRST_CAUSES_SECOND
    },

    {
        "sentence": "Hunger motivates eating.",
        "correct_action": FIRST_CAUSES_SECOND
    },

    {
        "sentence": "Heat produces expansion.",
        "correct_action": FIRST_CAUSES_SECOND
    },


    # Second entity causes first entity

    {
        "sentence": "Flooding was caused by rain.",
        "correct_action": SECOND_CAUSES_FIRST
    },

    {
        "sentence": "The alarm was triggered by smoke.",
        "correct_action": SECOND_CAUSES_FIRST
    },

    {
        "sentence": "Expansion was produced by heat.",
        "correct_action": SECOND_CAUSES_FIRST
    },

    {
        "sentence": "Eating was motivated by hunger.",
        "correct_action": SECOND_CAUSES_FIRST
    },


    # No causal relationship

    {
        "sentence": "Cats and bicycles are unrelated.",
        "correct_action": NO_CAUSAL_RELATION
    },

    {
        "sentence": "Music and rainfall are independent.",
        "correct_action": NO_CAUSAL_RELATION
    },

    {
        "sentence": "Coffee does not cause gravity.",
        "correct_action": NO_CAUSAL_RELATION
    },

    {
        "sentence": "Exercise and moonlight have no causal relationship.",
        "correct_action": NO_CAUSAL_RELATION
    }
]


# ============================================================
# HOLDOUT EXAMPLES
#
# CEM will NOT train on these.
# ============================================================

TEST_EXAMPLES = [

    {
        "sentence": "Stress causes insomnia.",
        "correct_action": FIRST_CAUSES_SECOND
    },

    {
        "sentence": "Insomnia was caused by stress.",
        "correct_action": SECOND_CAUSES_FIRST
    },

    {
        "sentence": "Stress and triangles are unrelated.",
        "correct_action": NO_CAUSAL_RELATION
    }
]


# ============================================================
# INFORMATION CHANNELS
# ============================================================

FEATURE_NAMES = [

    "relation_cue",
    "source_marker",
    "source_after_relation",
    "no_relation_signal",
    "negation",
    "relation_position",
    "bias"
]


# Words that suggest some relationship or change.
#
# These are NOT assumed to determine direction by themselves.

RELATION_WORDS = {
    "cause",
    "causes",
    "caused",

    "increase",
    "increases",
    "increased",

    "trigger",
    "triggers",
    "triggered",

    "motivate",
    "motivates",
    "motivated",

    "produce",
    "produces",
    "produced"
}


SOURCE_MARKERS = {
    "by",
    "from"
}


NEGATION_WORDS = {
    "not",
    "no",
    "never"
}


NO_RELATION_WORDS = {
    "unrelated",
    "independent"
}


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_channels(sentence):

    # Convert sentence into simple lowercase word tokens.

    tokens = re.findall(
        r"[a-z]+",
        sentence.lower()
    )

    number_of_tokens = len(tokens)


    # --------------------------------------------------------
    # CHANNEL 1:
    # Is there some explicit relationship/change cue?
    # --------------------------------------------------------

    relation_indices = [
        i
        for i, token in enumerate(tokens)
        if token in RELATION_WORDS
    ]

    relation_cue = float(
        len(relation_indices) > 0
    )


    # --------------------------------------------------------
    # CHANNEL 2:
    # Is there a source marker such as "by"?
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
    # CHANNEL 3:
    # Does a source marker occur AFTER the relation cue?
    #
    # Example:
    #
    # "triggered BY smoke"
    # --------------------------------------------------------

    source_after_relation = 0.0

    if relation_indices and source_indices:

        first_relation = relation_indices[0]
        first_source = source_indices[0]

        if first_source > first_relation:
            source_after_relation = 1.0


    # --------------------------------------------------------
    # CHANNEL 4:
    # Is there explicit evidence saying no relationship?
    # --------------------------------------------------------

    no_relation_signal = float(
        any(
            token in NO_RELATION_WORDS
            for token in tokens
        )
    )


    # --------------------------------------------------------
    # CHANNEL 5:
    # Is there negation?
    #
    # "does NOT cause"
    # "NO causal relationship"
    # --------------------------------------------------------

    negation = float(
        any(
            token in NEGATION_WORDS
            for token in tokens
        )
    )


    # --------------------------------------------------------
    # CHANNEL 6:
    # Where does the relation cue occur?
    #
    # 0.0 = near beginning
    # 1.0 = near end
    # --------------------------------------------------------

    if relation_indices and number_of_tokens > 1:

        relation_position = (
            relation_indices[0]
            /
            (number_of_tokens - 1)
        )

    else:

        relation_position = 0.0


    # --------------------------------------------------------
    # CHANNEL 7:
    # Constant bias feature.
    #
    # Lets actions have a default tendency independent
    # of the other features.
    # --------------------------------------------------------

    bias = 1.0


    observation = np.array(
        [
            relation_cue,
            source_marker,
            source_after_relation,
            no_relation_signal,
            negation,
            relation_position,
            bias
        ],
        dtype=np.float32
    )

    return observation


# ============================================================
# GYMNASIUM ENVIRONMENT
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
            dtype=np.float32
        )

        self.current_example = None
        self.current_index = None


    def reset(
        self,
        seed=None,
        options=None
    ):

        super().reset(seed=seed)

        # Allows us either to choose a specific sentence
        # or randomly choose one.

        if options is not None and "index" in options:

            self.current_index = options["index"]

        else:

            self.current_index = int(
                self.np_random.integers(
                    0,
                    len(self.examples)
                )
            )


        self.current_example = self.examples[
            self.current_index
        ]

        observation = extract_channels(
            self.current_example["sentence"]
        )

        info = {
            "sentence":
                self.current_example["sentence"],

            "correct_action":
                self.current_example["correct_action"]
        }

        return observation, info


    def step(self, action):

        correct_action = (
            self.current_example[
                "correct_action"
            ]
        )

        if action == correct_action:
            reward = 1.0
        else:
            reward = 0.0


        # One parsing decision completes an episode.

        terminated = True
        truncated = False


        observation = extract_channels(
            self.current_example["sentence"]
        )


        info = {
            "sentence":
                self.current_example["sentence"],

            "correct_action":
                correct_action
        }


        return (
            observation,
            reward,
            terminated,
            truncated,
            info
        )


# ============================================================
# POLICY
# ============================================================

def choose_action(
    observation,
    weights
):

    # weights has shape:
    #
    # 3 actions × 7 information channels
    #
    # Each possible action gets its own score.

    scores = weights @ observation

    action = np.argmax(scores)

    return action


# ============================================================
# POLICY EVALUATION
# ============================================================

def evaluate_policy(
    env,
    weights
):

    total_reward = 0


    for index in range(
        len(env.examples)
    ):

        observation, info = env.reset(
            options={
                "index": index
            }
        )

        action = choose_action(
            observation,
            weights
        )

        (
            observation,
            reward,
            terminated,
            truncated,
            info
        ) = env.step(action)


        total_reward += reward


    accuracy = (
        total_reward
        /
        len(env.examples)
    )


    return accuracy


# ============================================================
# CREATE TRAINING ENVIRONMENT
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

population_size = 100

elite_fraction = 0.20

generations = 30

number_of_elites = int(
    population_size
    *
    elite_fraction
)


# ============================================================
# INITIAL SEARCH DISTRIBUTION
#
# Shape:
#
# actions × features
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
# CEM TRAINING
# ============================================================

for generation in range(
    generations
):

    # --------------------------------------------------------
    # Generate many candidate policies
    # --------------------------------------------------------

    population = rng.normal(
        loc=mean_weights,
        scale=std_weights,
        size=(
            population_size,
            number_of_actions,
            number_of_features
        )
    )


    scores = []


    # --------------------------------------------------------
    # Evaluate every candidate policy
    # --------------------------------------------------------

    for candidate_weights in population:

        accuracy = evaluate_policy(
            train_env,
            candidate_weights
        )

        scores.append(
            accuracy
        )


    scores = np.array(scores)


    # --------------------------------------------------------
    # Find elite policies
    # --------------------------------------------------------

    elite_indices = np.argsort(
        scores
    )[
        -number_of_elites:
    ]


    elite_weights = population[
        elite_indices
    ]


    # --------------------------------------------------------
    # Update search distribution
    # --------------------------------------------------------

    mean_weights = np.mean(
        elite_weights,
        axis=0
    )


    std_weights = np.std(
        elite_weights,
        axis=0
    )


    # Prevent CEM from becoming completely certain
    # too early.

    std_weights = np.maximum(
        std_weights,
        0.05
    )


    # --------------------------------------------------------
    # Track best candidate
    # --------------------------------------------------------

    best_index = np.argmax(
        scores
    )

    generation_best = scores[
        best_index
    ]


    if generation_best > best_accuracy:

        best_accuracy = generation_best

        best_weights = population[
            best_index
        ].copy()


    print(
        f"Generation "
        f"{generation + 1:2d} "
        f"| best accuracy: "
        f"{generation_best:.2f}"
    )


train_env.close()


# ============================================================
# SHOW LEARNED WEIGHTS
# ============================================================

print("\n\nLEARNED POLICY")
print("=" * 60)


for action in range(
    number_of_actions
):

    print()
    print(
        ACTION_NAMES[action]
    )

    for feature_index, feature_name in enumerate(
        FEATURE_NAMES
    ):

        weight = best_weights[
            action,
            feature_index
        ]

        print(
            f"  {feature_name:24s}"
            f"{weight: .3f}"
        )


# ============================================================
# TEST ON TRAINING SENTENCES
# ============================================================

print("\n\nTRAINING EXAMPLES")
print("=" * 60)


train_env = TinyParsingEnv(
    TRAIN_EXAMPLES
)


for index, example in enumerate(
    TRAIN_EXAMPLES
):

    observation, info = train_env.reset(
        options={
            "index": index
        }
    )

    prediction = choose_action(
        observation,
        best_weights
    )

    print()
    print(example["sentence"])

    print(
        "Features:",
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
        "Predicted:",
        ACTION_NAMES[prediction]
    )

    print(
        "Correct:  ",
        ACTION_NAMES[
            example["correct_action"]
        ]
    )


train_env.close()


# ============================================================
# HOLDOUT TEST
# ============================================================

print("\n\nUNSEEN SENTENCES")
print("=" * 60)


test_env = TinyParsingEnv(
    TEST_EXAMPLES
)


test_accuracy = evaluate_policy(
    test_env,
    best_weights
)


for index, example in enumerate(
    TEST_EXAMPLES
):

    observation, info = test_env.reset(
        options={
            "index": index
        }
    )

    prediction = choose_action(
        observation,
        best_weights
    )

    print()
    print(example["sentence"])

    print(
        "Channels:",
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
        "Prediction:",
        ACTION_NAMES[prediction]
    )

    print(
        "Correct:   ",
        ACTION_NAMES[
            example["correct_action"]
        ]
    )


print(
    f"\nHoldout accuracy: "
    f"{test_accuracy:.2f}"
)

test_env.close()