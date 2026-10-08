import gymnasium as gym
import numpy as np


# --------------------------------------------------
# POLICY
# --------------------------------------------------

def choose_action(observation, weights):
    score = np.dot(weights, observation)

    if score > 0:
        return 1   # push right
    else:
        return 0   # push left


# --------------------------------------------------
# RUN ONE EPISODE
# --------------------------------------------------

def run_episode(env, weights):
    observation, info = env.reset()

    total_reward = 0

    while True:
        action = choose_action(observation, weights)

        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        if terminated or truncated:
            return total_reward


# --------------------------------------------------
# EVALUATE ONE POLICY OVER SEVERAL EPISODES
# --------------------------------------------------

def evaluate_policy(env, weights, episodes=5):
    rewards = []

    for _ in range(episodes):
        reward = run_episode(env, weights)
        rewards.append(reward)

    return np.mean(rewards)


# --------------------------------------------------
# ENVIRONMENT
# --------------------------------------------------

env = gym.make("CartPole-v1")

rng = np.random.default_rng(42)


# --------------------------------------------------
# CEM SETTINGS
# --------------------------------------------------

population_size = 40
elite_fraction = 0.20
generations = 20
evaluation_episodes = 5

number_of_elites = int(population_size * elite_fraction)


# --------------------------------------------------
# INITIAL DISTRIBUTION OVER THE FOUR WEIGHTS
# --------------------------------------------------

mean_weights = np.array([
    0.0,   # cart position
    0.0,   # cart velocity
    0.0,   # pole angle
    0.0    # pole angular velocity
])

std_weights = np.array([
    1.0,   # uncertainty about cart position weight
    1.0,   # uncertainty about cart velocity weight
    1.0,   # uncertainty about pole angle weight
    1.0    # uncertainty about pole angular velocity weight
])


best_weights = None
best_reward = -np.inf


# --------------------------------------------------
# TRAINING
# --------------------------------------------------

for generation in range(generations):

    # Generate a population of candidate weight vectors
    population = rng.normal(
        loc=mean_weights,
        scale=std_weights,
        size=(population_size, 4)
    )

    scores = []

    # Test every candidate policy
    for candidate_weights in population:

        score = evaluate_policy(
            env,
            candidate_weights,
            episodes=evaluation_episodes
        )

        scores.append(score)

    scores = np.array(scores)


    # Rank candidates from best to worst
    elite_indices = np.argsort(scores)[-number_of_elites:]

    elite_weights = population[elite_indices]


    # Update the search distribution
    mean_weights = np.mean(elite_weights, axis=0)
    std_weights = np.std(elite_weights, axis=0)

    # Prevent exploration from collapsing completely
    std_weights = np.maximum(std_weights, 0.05)


    # Track the best individual policy found
    generation_best_index = np.argmax(scores)
    generation_best_reward = scores[generation_best_index]

    if generation_best_reward > best_reward:
        best_reward = generation_best_reward
        best_weights = population[generation_best_index].copy()


    print(f"\nGeneration {generation + 1}")
    print(f"Best mean reward: {generation_best_reward:.1f}")
    print(f"Elite mean reward: {np.mean(scores[elite_indices]):.1f}")

    print("Mean weights:")
    print(f"  cart position:        {mean_weights[0]: .3f}")
    print(f"  cart velocity:        {mean_weights[1]: .3f}")
    print(f"  pole angle:           {mean_weights[2]: .3f}")
    print(f"  pole angular velocity:{mean_weights[3]: .3f}")

    print("Search uncertainty:")
    print(f"  cart position:        {std_weights[0]: .3f}")
    print(f"  cart velocity:        {std_weights[1]: .3f}")
    print(f"  pole angle:           {std_weights[2]: .3f}")
    print(f"  pole angular velocity:{std_weights[3]: .3f}")


env.close()


# --------------------------------------------------
# DEMONSTRATE BEST POLICY
# --------------------------------------------------

demo_env = gym.make(
    "CartPole-v1",
    render_mode="human"
)

demo_reward = run_episode(
    demo_env,
    best_weights
)

print("\nBest learned weights:")
print(f"cart position:         {best_weights[0]: .3f}")
print(f"cart velocity:         {best_weights[1]: .3f}")
print(f"pole angle:            {best_weights[2]: .3f}")
print(f"pole angular velocity: {best_weights[3]: .3f}")

print(f"\nDemo survived {demo_reward} steps")

demo_env.close()