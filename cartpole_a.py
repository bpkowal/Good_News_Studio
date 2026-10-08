import gymnasium as gym
import numpy as np


def evaluate_policy(env, weights, episodes=10):
    rewards = []

    for _ in range(episodes):
        reward = run_episode(env, weights)
        rewards.append(reward)

    return np.mean(rewards)

def choose_action(observation, weights):
    score = np.dot(weights, observation)

    if score > 0:
        return 1
    else:
        return 0


def run_episode(env, weights):
    observation, info = env.reset()
    total_reward = 0

    while True:
        action = choose_action(observation, weights)

        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        if terminated or truncated:
            return total_reward


# TRAINING ENVIRONMENT — invisible and fast
env = gym.make("CartPole-v1")

weight_cart_position = 0.0
weight_cart_velocity = 0.0
weight_pole_angle = 0.0
weight_pole_angular_velocity = 0.0

weights = np.array([
    weight_cart_position,
    weight_cart_velocity,
    weight_pole_angle,
    weight_pole_angular_velocity
])

best_reward = 0

for trial in range(200):

    candidate_weights = weights + np.random.normal(
        0,
        0.5,
        size=len(weights)
    )

    reward = evaluate_policy(env, candidate_weights)

    if reward > best_reward:
        best_reward = reward
        weights = candidate_weights

        print("New best:", best_reward)
        print("Weights:", weights)


env.close()


# VISUALIZE ONLY THE BEST POLICY
print("\nTraining complete.")
print("Best reward:", best_reward)

demo_env = gym.make("CartPole-v1", render_mode="human")

demo_reward = run_episode(demo_env, weights)

print("Demo survived:", demo_reward, "steps")

demo_env.close()