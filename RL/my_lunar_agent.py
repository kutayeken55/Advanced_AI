import numpy as np
import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

env = gym.make("LunarLander-v3")

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

# plt.ion()

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 256)
        self.layer3 = nn.Linear(256,128)
        self.layer4 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)


# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 256
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 500
TAU = 1e-3
LR = 5e-4
MAX_STEPS = 500


# Get the number of state observations
state, info = env.reset()
n_observations = len(state)
# Get number of actions from gym action space
n_actions = env.action_space.n

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.MSELoss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

    return loss

if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 600
else:
    num_episodes = 120

TARGET_UPDATE = 20
losses = []
rewards = []
print("TRAINING...")
for i_episode in range(num_episodes):
    # Initialize the environment and get its state
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    episode_loss = 0
    total_reward = 0
    for t in count():
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        # Extract state variables from observation
        lander_x, lander_y, vel_x, vel_y, angle, angle_vel, left_leg, right_leg = observation

        # Manually detect a stable landing and end the episode
        if left_leg > 0.5 and right_leg > 0.5 and abs(vel_x) < 0.1 and abs(vel_y) < 0.1 and abs(angle) < 0.1:
            done = True  # Force episode termination

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state
        total_reward += reward.item()

        # Perform one step of the optimization (on the policy network)
        loss = optimize_model()
        if loss is not None:
            episode_loss += loss.item()

        if done:
            break
    
    losses.append(episode_loss)
    rewards.append(total_reward)
            
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Training Completed')

def select_action(state):
    """Selects the best action from the trained DQN model."""
    with torch.no_grad():
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        action = policy_net(state).argmax(dim=1).item()
    return action

def run_trained_agent(episodes=5):
    rewards = []
    for episode in range(episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        step_count = 0

        while not done:
            action = select_action(obs)  # Get action from trained model
            obs, reward, done, truncated, info = env.step(action)
            step_count += 1
            total_reward += reward
            if step_count >= 500:
                done = True
            # env.render()  # Show environment

        print(f"Episode {episode+1}: Total Reward: {total_reward}")
        rewards.append(total_reward)

    env.close()
    return rewards

def run_random_agent(episodes=5):
    rewards = []
    for episode in range(episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = env.action_space.sample()  # random action
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            env.render()  # Show environment

        print(f"Episode {episode+1}: Total Reward: {total_reward}")
        rewards.append(total_reward)

    env.close()
    return rewards

agent_rewards = run_trained_agent(episodes=50)
random_rewards = run_random_agent(episodes=50)

print("random avg: ", np.mean(random_rewards))
print("DQN avg: ", np.mean(agent_rewards))

window_size = 50
 
moving_average_reward = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
moving_averages_loss = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')

# # Create a single figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # 1 row, 2 columns
# # Plot Random Agent Curve
axes[0].plot(range(len(moving_averages_loss)), moving_averages_loss)
axes[0].set_xlabel("Episodes")
axes[0].set_ylabel("Losses")
axes[0].set_title("Loss vs Episodes")

# # Plot Q-Learning Agent Curve
axes[1].plot(range(len(moving_average_reward)), moving_average_reward)
axes[1].set_xlabel("Episodes")
axes[1].set_ylabel("Reward")
axes[1].set_title("Rewards vs Episodes")

# # Adjust layout and display
plt.tight_layout()
plt.show()


