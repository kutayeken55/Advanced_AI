import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete, Box
from core import MLPActorCritic

def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    # Build a feedforward neural network.
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def train(env_name='CartPole-v1', hidden_sizes=[32], lr=1e-2, 
          epochs=100, batch_size=5000, render=False):

    # make environment, check spaces, get obs / act dims
    env = gym.make(env_name)
    # assert isinstance(env.observation_space, Box), \
    #     "This example only works for envs with continuous state spaces."
    # assert isinstance(env.action_space, Discrete), \
    #     "This example only works for envs with discrete action spaces."

    obs_dim = env.observation_space.shape[0]
    # n_acts = env.action_space.n
    actor_critic = MLPActorCritic(env.observation_space, env.action_space, hidden_sizes)
    optimizer = Adam(actor_critic.parameters(), lr=lr)
      
    def compute_loss(obs, act, weights):
        dist = actor_critic.pi._distribution(obs)
        logp = actor_critic.pi._log_prob_from_distribution(dist, act)
        return -(logp * weights).mean()

    # for training policy
    def train_one_epoch():
        # make some empty lists for logging.
        batch_obs = []          # for observations
        batch_acts = []         # for actions
        batch_weights = []      # for R(tau) weighting in policy gradient
        batch_rets = []         # for measuring episode returns
        batch_lens = []         # for measuring episode lengths

        # reset episode-specific variables
        obs, info = env.reset()  # updated reset handling
        done = False            # signal from environment that episode is over
        ep_rews = []            # list for rewards accrued throughout ep

        # render first episode of each epoch
        finished_rendering_this_epoch = False

        # collect experience by acting in the environment with current policy
        while True:

            # rendering
            if (not finished_rendering_this_epoch) and render:
                env.render()

            # save obs
            batch_obs.append(obs.copy())
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
            dist = actor_critic.pi._distribution(obs_tensor)
            act_tensor = dist.sample()

            if isinstance(env.action_space, Discrete):
                act = int(act_tensor)
            else:
                act = act_tensor.detach().cpu().tolist()

            # act in the environment
            obs, rew, terminated, truncated, info = env.step(act)
            done = terminated or truncated

            # save action, reward
            batch_acts.append(act_tensor)
            ep_rews.append(rew)

            if done:
                # if episode is over, record info about episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # the weight for each logprob(a|s) is R(tau)
                batch_weights += [ep_ret] * ep_len

                # reset episode-specific variables
                obs, info = env.reset()
                done, ep_rews = False, []

                # won't render again this epoch
                finished_rendering_this_epoch = True

                # end experience loop if we have enough of it
                if len(batch_obs) > batch_size:
                    break
        
        obs_tensor = torch.as_tensor(batch_obs, dtype=torch.float32)
        act_tensor = torch.stack(batch_acts)
        weights = torch.as_tensor(batch_weights, dtype=torch.float32)

        # take a single policy gradient update step
        optimizer.zero_grad()
        loss = compute_loss(obs_tensor, act_tensor, weights)
        loss.backward()
        optimizer.step()

        return loss.item(), batch_rets, batch_lens

    # training loop
    for i in range(epochs):
        batch_loss, batch_rets, batch_lens = train_one_epoch()
        print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f'%
                (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v1')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-2)
    args = parser.parse_args()
    print('\nUsing simplest formulation of policy gradient.\n')
    train(env_name=args.env_name, render=args.render, lr=args.lr)