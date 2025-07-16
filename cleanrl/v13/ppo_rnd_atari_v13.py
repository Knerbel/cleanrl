import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from dataclasses import dataclass
import tyro
from typing import Optional
import time
from torch.utils.tensorboard import SummaryWriter
import time
import os


@dataclass
class Args:
    exp_name: str = "PPO_RND"
    env_id: str = "CartPole-v1"
    total_timesteps: int = 500000
    learning_rate: float = 2.5e-4
    num_steps: int = 128
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    rnd_reward_scale: float = 0.1
    seed: int = 1
    cuda: bool = True
    track: bool = False
    capture_video: bool = False
    track: bool = False
    """if toggled, this experiment will be tracked with TensorBoard"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""

    # Add to runtime calculations
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, act_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def get_action(self, x):
        logits = self.actor(x)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action)

    def get_value(self, x):
        return self.critic(x)


class RNDNetwork(nn.Module):
    def __init__(self, obs_dim, hidden_dim=64):
        super().__init__()
        self.target = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Initialize target network (frozen)
        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, obs):
        target_feature = self.target(obs)
        predict_feature = self.predictor(obs)
        return ((target_feature - predict_feature) ** 2).mean(dim=1)


def main():
    args = tyro.cli(Args)
    # Calculate runtime values
    args.batch_size = int(args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    # Setup tensorboard
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % (
            "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Initialize environment
    env = gym.make(args.env_id)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env.seed(args.seed)

    # Initialize networks
    policy = PolicyNetwork(obs_dim, act_dim).to(device)
    rnd = RNDNetwork(obs_dim).to(device)
    optimizer = optim.Adam([
        {'params': policy.parameters(), 'lr': args.learning_rate},
        {'params': rnd.predictor.parameters(), 'lr': args.learning_rate}
    ])

    # Storage
    obs = torch.zeros((args.num_steps, obs_dim)).to(device)
    actions = torch.zeros((args.num_steps,)).to(device)
    rewards = torch.zeros((args.num_steps,)).to(device)
    dones = torch.zeros((args.num_steps,)).to(device)
    values = torch.zeros((args.num_steps,)).to(device)
    log_probs = torch.zeros((args.num_steps,)).to(device)

    # Training loop
    obs_env = env.reset()
    episode_reward = 0
    total_steps = 0

    global_step = 0
    start_time = time.time()

    while total_steps < args.total_timesteps:
        for step in range(args.num_steps):
            global_step += 1

            obs[step] = torch.FloatTensor(obs_env)

            # Get action and value
            with torch.no_grad():
                action, log_prob = policy.get_action(obs[step])
                value = policy.get_value(obs[step])

            # Environment step
            obs_env, reward, done, epoch = env.step(action.item())

            # Calculate intrinsic reward
            with torch.no_grad():
                intrinsic_reward = rnd(obs[step].unsqueeze(0)).item()

            # Store data
            actions[step] = action
            log_probs[step] = log_prob
            values[step] = value
            rewards[step] = reward + args.rnd_reward_scale * intrinsic_reward
            dones[step] = done

            episode_reward += reward
            total_steps += 1

            if done:
                writer.add_scalar("charts/episodic_return",
                                  episode_reward, global_step)
                print(
                    f"global_step={global_step}, episodic_return={episode_reward:.2f}")
                episode_reward = 0
                obs_env = env.reset()

        # Calculate advantages
        with torch.no_grad():
            next_value = policy.get_value(
                torch.FloatTensor(obs_env).to(device))
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0

            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]

                delta = rewards[t] + args.gamma * \
                    nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * \
                    args.gae_lambda * nextnonterminal * lastgaelam

            returns = advantages + values

        # PPO Update
        batch_size = args.num_steps
        mini_batch_size = batch_size // args.num_minibatches

        for epoch in range(args.update_epochs):
            for start in range(0, batch_size, mini_batch_size):
                end = start + mini_batch_size
                mb_obs = obs[start:end]
                mb_actions = actions[start:end]
                mb_old_values = values[start:end]
                mb_advantages = advantages[start:end]
                mb_returns = returns[start:end]
                mb_old_log_probs = log_probs[start:end]

                # Get new values and log probs
                new_values = policy.get_value(mb_obs)
                logits = policy.actor(mb_obs)
                probs = Categorical(logits=logits)
                new_log_probs = probs.log_prob(mb_actions)
                entropy = probs.entropy().mean()

                # Policy loss
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - args.clip_coef,
                                    1.0 + args.clip_coef) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = ((new_values - mb_returns) ** 2).mean()

                # RND loss
                rnd_loss = rnd(mb_obs).mean()

                # Total loss
                loss = policy_loss + args.vf_coef * value_loss - \
                    args.ent_coef * entropy + rnd_loss

                # Update networks
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    policy.parameters(), args.max_grad_norm)
                nn.utils.clip_grad_norm_(
                    rnd.predictor.parameters(), args.max_grad_norm)
                optimizer.step()

                writer.add_scalar("losses/value_loss",
                                  value_loss.item(), global_step)
                writer.add_scalar("losses/policy_loss",
                                  policy_loss.item(), global_step)
                writer.add_scalar("losses/entropy",
                                  entropy.item(), global_step)
                writer.add_scalar("losses/rnd_loss",
                                  rnd_loss.item(), global_step)
                writer.add_scalar("losses/total_loss",
                                  loss.item(), global_step)

        writer.add_scalar("charts/learning_rate",
                          optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("charts/SPS",
                          int(global_step / (time.time() - start_time)), global_step)

    env.close()
    writer.close()


if __name__ == "__main__":
    main()
