import numpy as np
import random
import matplotlib.pyplot as plt

# Author: Kutay Eken
# Date: Jan 29, 2025

# PT1:
# - only plot the average rewards, not the optimal action
# - plot lines for epsilon 0, 0.01, and 0.1
# - use numpy.random.normal for pt1 (reward sampling)
# - number of bandit tasks should be at least 100
# - discuss assumptions and results
class Bandit():
    # intializes the multi-arm bandit
    def __init__(self, num_actions):
        self.action_ids = list(range(num_actions))
        self.current_act_vals = {key: 0 for key in self.action_ids}
        self.real_act_vals = np.random.normal(0, 1, num_actions)
        self.action_to_rewards = {key: [] for key in self.action_ids}

    # takes action given the epsilon value, updates current estimate rewards
    def take_action(self, epsilon):
        x = np.random.uniform(0, 1)

        if x <= epsilon:
            # take random action
            action = random.choice(self.action_ids)
        else:
            # greedy action
            action = max(self.current_act_vals, key=self.current_act_vals.get)
        
        new_reward = np.random.normal(self.real_act_vals[action], 1)

        self.action_to_rewards[action].append(new_reward)

        new_estimate_reward = sum(self.action_to_rewards[action]) / len(self.action_to_rewards[action])

        self.current_act_vals[action] = new_estimate_reward

        return new_reward

# creates 2000 10-armed bandits. Each agent plays 1000 times. Computes average reward for each time_step t. Plots the results to compare different epsilon values.
def run_experiment(epsilon_vals=[0, 0.01, 0.1], plays=1000):
    results = []
    for eps in epsilon_vals:
        eps_t_reward_avg = {key: 0 for key in range(plays)}
        for i in range(2000):
            bandit = Bandit(10)
            for t in range(plays):
                reward = bandit.take_action(eps)
                eps_t_reward_avg[t] += reward

        eps_t_reward_avg = {key: value / 2000 for key, value in eps_t_reward_avg.items()}
        results.append(eps_t_reward_avg)

    print("Experiment is finished. Now Plotting...")
    
    plt.figure(figsize=(8,5))

    for i, data in enumerate(results):
        epsil = epsilon_vals[i]
        x = list(data.keys())
        y = list(data.values())
        plt.plot(x, y, marker='.', label=f'ε = {epsil}')

    plt.xlabel("Plays")
    plt.ylabel("Avg. Reward")
    plt.title("Comparison of Different Epsilon Values in ε-Greedy Multi-Armed Bandits")
    plt.legend()
    plt.show()


run_experiment()


# PT2:
# - have 10 arms
# - Bernoulli bandit (+1 with prob. p, 0 with prob. 1-p)
# - use epsilon = 0, 0.01, 0.1 with UCB1??
# - average your results over 100 random bandit problems
# - To create each problem, sample a probability p for each arm uniformly in the range [0,1]
# - use numpy.random.binomial for pt2
# - Discuss the results

