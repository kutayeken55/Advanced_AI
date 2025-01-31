import numpy as np
import random
import matplotlib.pyplot as plt

# Author: Kutay Eken
# Date: Jan 29, 2025

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

class Bernoulli_Bandit():
    # intializes the multi-arm bandit
    def __init__(self, num_actions, greedy):
        self.action_ids = list(range(num_actions))
        self.action_to_probs = {key: np.random.uniform(0, 1) for key in self.action_ids}
        self.action_to_rewards = {key: [] for key in self.action_ids}
        self.action_to_counts = {key: 0 for key in self.action_ids}
        self.Q_t = {key: 0 for key in self.action_ids}
        
        if greedy == False:
            # pull each arm once
            for action_id in self.action_ids:
                init_reward = np.random.binomial(1, self.action_to_probs[action_id])
                self.action_to_rewards[action_id].append(init_reward)
                self.Q_t[action_id] = init_reward
                self.action_to_counts[action_id] += 1

    def calculate_c(self, action_id, t):
        return np.sqrt((np.log(t + 1)) / (self.action_to_counts[action_id] + 1e-5))

    def take_action_bern(self, time_step):
        ucb_values = np.array([self.Q_t[a] + self.calculate_c(a, time_step) for a in self.action_ids])
        action_taken = np.argmax(ucb_values)

        new_reward = np.random.binomial(1, self.action_to_probs[action_taken])

        self.action_to_counts[action_taken] += 1

        self.action_to_rewards[action_taken].append(new_reward)

        new_estimate_reward = sum(self.action_to_rewards[action_taken]) / self.action_to_counts[action_taken]

        self.Q_t[action_taken] = new_estimate_reward

        return new_reward

    def take_action_greedy(self, epsilon, time_step):
        x = np.random.uniform(0, 1)

        if x <= epsilon:
            # take random action
            action = random.choice(self.action_ids)
        else:
            # greedy action
            action = max(self.Q_t, key=self.Q_t.get)
        
        new_reward = np.random.binomial(1, self.action_to_probs[action])

        self.action_to_rewards[action].append(new_reward)

        self.action_to_counts[action] += 1

        new_estimate_reward = sum(self.action_to_rewards[action]) / self.action_to_counts[action]

        self.Q_t[action] = new_estimate_reward

        return new_reward

# creates 2000 10-armed bandits. Each agent plays 1000 times. Computes average reward for each time_step t. Plots the results to compare different epsilon values.
def run_experiment(epsilon_vals=[0, 0.01, 0.1], plays=1000):
    print("Started Epsilon Greedy Experiment")
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

    print("Finished Epsilon Greedy Experiment")
    
    return results

# Compares Bernoulli Bandit Performance Between UBC1 and Epsilon Greedy (0, 0.01, 0.1)
def run_experiment_bernoulli(epsilon_vals=[0, 0.01, 0.1], plays=1000):
    print("Started Bernoulli Bandit Experiment...")
    results = []

    # eps Greedy w/ Bernouilli Bandit
    for eps in epsilon_vals:
        eps_t_reward_avg = {key: 0 for key in range(plays)}
        for i in range(2000):
            bandit = Bernoulli_Bandit(10, True)
            for t in range(plays):
                reward = bandit.take_action_greedy(eps, t)
                eps_t_reward_avg[t] += reward

        eps_t_reward_avg = {key: value / 2000 for key, value in eps_t_reward_avg.items()}
        results.append(eps_t_reward_avg)


    # UBC1
    time_to_avg_rew = {key: 0 for key in range(plays)}
    for i in range(2000):
        bern_bandit = Bernoulli_Bandit(10, False)
        for t in range(plays):
            reward = bern_bandit.take_action_bern(t)
            time_to_avg_rew[t] += reward
    
    time_to_avg_rew = {key: value / 2000 for key, value in time_to_avg_rew.items()}
    print("Finished Bernoulli Bandit Experiment...")

    return results, time_to_avg_rew

# Plots Greedy Epsilon Comparison
def plot_eps(results, epsilon_vals=[0, 0.01, 0.1]):
    print("Plotting the Greedy Epsilon...")

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

# Plots Greedy Epsilon vs UBC1
def plot_all(results, ubc_results, epsilon_vals=[0, 0.01, 0.1]):
    print("Plotting Greedy Epsilon & UBC1 Comparison")

    plt.figure(figsize=(8,5))

    for i, data in enumerate(results):
        epsil = epsilon_vals[i]
        x = list(data.keys())
        y = list(data.values())
        plt.plot(x, y, marker='.', label=f'ε = {epsil}')
    
    ubc_x = list(ubc_results.keys())
    ubc_y = list(ubc_results.values())
    plt.plot(ubc_x, ubc_y, marker='.', label="UBC1")

    plt.xlabel("Plays")
    plt.ylabel("Avg. Reward")
    plt.title("Multi-Armed Bandit Comparison of ε-Greedy and UBC1 Algorithms")
    plt.legend()
    plt.show()

# PART 1
results = run_experiment()
plot_eps(results)

# PART 2
results, ubc_results = run_experiment_bernoulli()
plot_all(results, ubc_results)