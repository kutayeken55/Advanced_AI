import matplotlib.pyplot as plt
import gymnasium as gym
from collections import defaultdict
import numpy as np

def eps_greedy(state, q_vals, epsilon=0.1):
    x = np.random.uniform(0, 1)

    if x <= epsilon:
        # take random action
        return env.action_space.sample()
    else:
        # greedy action
        return (int)(np.argmax(q_vals[state]))


# creating the environment
env = gym.make("Blackjack-v1")


# Q-table initialized as a defaultdict
# Each state maps to an array of two Q-values: [Q_hit, Q_stick]
q_values = defaultdict(lambda: np.zeros(env.action_space.n))

num_episodes = 100000
gamma = 0.95
alpha = 0.1
cumulative_rewards = []

for i in range(num_episodes):
    # initialize environment
    state, info = env.reset()
    finished = False
    episode_reward = 0

    # repeat (for each step of episode)
    while not finished:
        # choose a from s using policy derived from Q (eps-greedy)
        action = eps_greedy(state, q_values)

        # Take action a, observe r, s'
        next_state, reward, terminated, truncated, info = env.step(action)

        # update q_vals
        current_q_val = q_values[state][action]
        best_next_action = np.argmax(q_values[next_state])
        q_values[state][action] = current_q_val + alpha * (reward + gamma * q_values[next_state][best_next_action] - current_q_val)

        # update environment
        episode_reward += reward
        finished = terminated or truncated
        state = next_state

    cumulative_rewards.append(episode_reward)

print("----- Training is completed -----")
# Testing the trained agent
wins = 0

for i in range(num_episodes):
    state, info = env.reset()
    finished = False

    while not finished:
        action = np.argmax(q_values[state])
        next_state, reward, terminated, truncated, info = env.step(action)
        finished = terminated or truncated
        state = next_state
        
    if reward == 1:
        wins += 1


print("----- Win ratio for the trained agent is: %", (wins/num_episodes) * 100, "-----")

# Testing the random agent
random_wins = 0
random_cumulative_reward = []

for i in range(num_episodes):
    state, info = env.reset()
    rnd_episode_reward = 0
    finished = False

    while not finished:
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)
        finished = terminated or truncated
        state = next_state

        rnd_episode_reward += reward
    
    random_cumulative_reward.append(rnd_episode_reward)
        
    if reward == 1:
        random_wins += 1

print("----- Win ratio for the random agent is: %", (random_wins/num_episodes) * 100, "-----")

# Plotting

print("----- Plotting... -----")
window_size = 200
cumulative_rewards = np.convolve(cumulative_rewards, np.ones(window_size) / window_size, mode='valid')
random_cumulative_reward = np.convolve(random_cumulative_reward, np.ones(window_size) / window_size, mode='valid')
# Create a single figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # 1 row, 2 columns

# Plot Random Agent Curve
axes[0].plot(range(len(random_cumulative_reward)), random_cumulative_reward)
axes[0].set_xlabel("Episodes")
axes[0].set_ylabel("Cumulative Reward")
axes[0].set_title("Random Agent Plot")

# Plot Q-Learning Agent Curve
axes[1].plot(range(len(cumulative_rewards)), cumulative_rewards)
axes[1].set_xlabel("Episodes")
axes[1].set_ylabel("Cumulative Reward")
axes[1].set_title("Q-Learning Agent Plot")

# Adjust layout and display
plt.tight_layout()
plt.show()
