import gymnasium as gym
from gymnasium import spaces
import numpy as np


class TinyParsingEnv(gym.Env):

    def __init__(self):

        # --------------------------------------------------
        # SENTENCES AND CORRECT PARSES
        # --------------------------------------------------

        self.examples = [
            {
                "sentence": "Exercise increases mood.",
                "correct_action": 0
            },
            {
                "sentence": "Rain causes flooding.",
                "correct_action": 0
            },
            {
                "sentence": "Hunger motivates eating.",
                "correct_action": 0
            },
            {
                "sentence": "The alarm was triggered by smoke.",
                "correct_action": 1
            },
            {
                "sentence": "Cats and bicycles are unrelated.",
                "correct_action": 2
            }
        ]

        # --------------------------------------------------
        # ACTION SPACE
        # --------------------------------------------------

        # 0 = first thing causes second thing
        # 1 = second thing causes first thing
        # 2 = no causal relationship

        self.action_space = spaces.Discrete(3)

        # --------------------------------------------------
        # OBSERVATION SPACE
        # --------------------------------------------------

        # The observation is simply:
        #
        # 0 = first sentence
        # 1 = second sentence
        # 2 = third sentence
        # etc.
        #
        # Later we will replace this with actual linguistic features.

        self.observation_space = spaces.Discrete(len(self.examples))

        self.current_example = None
        self.current_index = None


    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        # Pick a random sentence.

        self.current_index = self.np_random.integers(
            0,
            len(self.examples)
        )

        self.current_example = self.examples[self.current_index]

        observation = self.current_index

        info = {
            "sentence": self.current_example["sentence"]
        }

        return observation, info


    def step(self, action):

        correct_action = self.current_example["correct_action"]

        # --------------------------------------------------
        # REWARD
        # --------------------------------------------------

        if action == correct_action:
            reward = 1.0
        else:
            reward = 0.0

        # One decision completes the episode.
        terminated = True
        truncated = False

        observation = self.current_index

        info = {
            "sentence": self.current_example["sentence"],
            "correct_action": correct_action
        }

        return observation, reward, terminated, truncated, info


# --------------------------------------------------
# HUMAN-READABLE ACTION NAMES
# --------------------------------------------------

action_names = {
    0: "FIRST causes SECOND",
    1: "SECOND causes FIRST",
    2: "NO causal relationship"
}


# --------------------------------------------------
# CREATE ENVIRONMENT
# --------------------------------------------------

env = TinyParsingEnv()


# --------------------------------------------------
# PLAY TEN RANDOM ROUNDS
# --------------------------------------------------

total_reward = 0

for episode in range(10):

    observation, info = env.reset()

    print("\nSentence:")
    print(info["sentence"])

    # For now, our agent knows nothing.
    # It chooses randomly.

    action = env.action_space.sample()

    print("Agent chooses:")
    print(action_names[action])

    observation, reward, terminated, truncated, info = env.step(action)

    print("Reward:", reward)

    if reward == 0:
        print(
            "Correct answer:",
            action_names[info["correct_action"]]
        )

    total_reward += reward


print("\nTotal reward:", total_reward)
print("Accuracy:", total_reward / 10)

env.close()