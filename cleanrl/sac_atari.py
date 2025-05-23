import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from stable_baselines3.common.buffers import ReplayBuffer
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter


# Import your Fireboy and Watergirl environment to ensure it's registered
import cleanrl.fireboy_and_watergirl_sac
import cleanrl.fireboy_and_watergirl_ppo_v4


@dataclass
class Args:
    exp_name: str = "snake learning parallel"
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "FireboyAndWatergirl-ppo-v4"
    """the id of the environment"""
    total_timesteps: int = 2_000_000
    """total timesteps of the experiments"""
    buffer_size: int = int(256000/6)
    """the replay memory buffer size"""  # smaller than in original paper but evaluation is done only for 100k steps anyway
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """target smoothing coefficient (default: 1)"""
    batch_size: int = 64
    """the batch size of sample from the reply memory"""
    learning_starts: int = 2e4
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 3e-4
    """the learning rate of the Q network network optimizer"""
    update_frequency: int = 4
    """the frequency of training updates"""
    target_network_frequency: int = 8000
    """the frequency of updates for the target networks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    target_entropy_scale: float = 0.89
    """coefficient for scaling the autotune entropy target"""


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)

        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)

        env.action_space.seed(seed)
        return env

    return thunk

# def make_env(env_id, seed, idx, capture_video, run_name):
#     def thunk():
#         if capture_video and idx == 0:
#             env = gym.make(env_id, render_mode="rgb_array")
#             env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
#         else:
#             env = gym.make(env_id)
#         env = gym.wrappers.RecordEpisodeStatistics(env)

#         # Non-Atari environments do not need Atari-specific wrappers
#         env.action_space.seed(seed)
#         return env

#     return thunk


def layer_init(layer, bias_const=0.0):
    nn.init.kaiming_normal_(layer.weight)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# ALGO LOGIC: initialize agent here:
# NOTE: Sharing a CNN encoder between Actor and Critics is not recommended for SAC without stopping actor gradients
# See the SAC+AE paper https://arxiv.org/abs/1910.01741 for more info
# TL;DR The actor's gradients mess up the representation when using a joint encoder
class SoftQNetwork(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_shape = envs.single_observation_space.shape
        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.Flatten(),
        )

        with torch.inference_mode():
            output_dim = self.conv(torch.zeros(1, *obs_shape)).shape[1]

        self.fc1 = layer_init(nn.Linear(output_dim, 512))
        # Modify for MultiDiscrete: Create separate Q-values for each action dimension
        # Get dimensions from MultiDiscrete
        self.action_dims = envs.single_action_space.nvec
        self.fc_qs = nn.ModuleList([
            layer_init(nn.Linear(512, dim)) for dim in self.action_dims
        ])

    def forward(self, x):
        x = F.relu(self.conv(x / 255.0))
        x = F.relu(self.fc1(x))
        # Return Q-values for each action dimension
        return [fc_q(x) for fc_q in self.fc_qs]


class Actor(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_shape = envs.single_observation_space.shape
        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.Flatten(),
        )

        with torch.inference_mode():
            output_dim = self.conv(torch.zeros(1, *obs_shape)).shape[1]

        self.fc1 = layer_init(nn.Linear(output_dim, 512))
        # Modify for MultiDiscrete: Create separate logits for each action dimension
        self.action_dims = envs.single_action_space.nvec
        self.fc_logits = nn.ModuleList([
            layer_init(nn.Linear(512, dim)) for dim in self.action_dims
        ])

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = F.relu(self.fc1(x))
        return [fc_logit(x) for fc_logit in self.fc_logits]

    def get_action(self, x):
        logits = self(x / 255.0)
        # Create separate distributions for each action dimension
        policy_dists = [Categorical(logits=logit) for logit in logits]

        if len(x.shape) == 4:  # If batch of observations
            actions = torch.stack([dist.sample()
                                  for dist in policy_dists], dim=1)
            log_probs = torch.stack([F.log_softmax(logit, dim=1)
                                    for logit in logits], dim=1)
            action_probs = torch.stack(
                [F.softmax(logit, dim=1) for logit in logits], dim=1)
        else:  # If single observation
            actions = torch.tensor([dist.sample() for dist in policy_dists])
            log_probs = torch.stack(
                [F.log_softmax(logit, dim=0) for logit in logits])
            action_probs = torch.stack(
                [F.softmax(logit, dim=0) for logit in logits])

        return actions, log_probs, action_probs


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:

poetry run pip install "stable_baselines3==2.0.0a1" "gymnasium[atari,accept-rom-license]==0.28.1"  "ale-py==0.8.1" 
"""
        )
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{args.total_timesteps}__{args.buffer_size}__{args.gamma}__{args.tau}__{args.target_network_frequency}__{args.batch_size}__{args.learning_starts}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % (
            "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    # assert isinstance(envs.single_action_space,
    #                   gym.spaces.Discrete), "only discrete action space is supported"

    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    # TRY NOT TO MODIFY: eps=1e-4 increases numerical stability
    q_optimizer = optim.Adam(list(qf1.parameters()) +
                             list(qf2.parameters()), lr=args.q_lr, eps=1e-4)
    actor_optimizer = optim.Adam(
        list(actor.parameters()), lr=args.policy_lr, eps=1e-4)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -args.target_entropy_scale * \
            sum([torch.log(1 / torch.tensor(dim))
                for dim in envs.single_action_space.nvec])
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr, eps=1e-4)
    else:
        alpha = args.alpha

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample()
                               for _ in range(envs.num_envs)])

        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

            # if global_step % 1000 == 0:
            #     envs.envs[0].render()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(
            actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                # Skip the envs that are not done
                if "episode" not in info:
                    continue
                print(
                    f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return",
                                  info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length",
                                  info["episode"]["l"], global_step)
                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.update_frequency == 0:
                data = rb.sample(args.batch_size)
                # CRITIC training
                with torch.no_grad():
                    _, next_state_log_pi, next_state_action_probs = actor.get_action(
                        data.next_observations)
                    qf1_next_targets = qf1_target(data.next_observations)
                    qf2_next_targets = qf2_target(data.next_observations)

                    # Initialize next Q-value
                    min_qf_next_target = torch.zeros(
                        args.batch_size, device=device)

                    # Handle each action dimension separately
                    for dim in range(len(actor.action_dims)):
                        min_qf_next = torch.min(
                            qf1_next_targets[dim], qf2_next_targets[dim])
                        # Compute expected Q-values for each action
                        next_q_pi = (
                            next_state_action_probs[..., dim, :] * min_qf_next).sum(dim=1)
                        # Add entropy term
                        min_qf_next_target += next_q_pi - alpha * (next_state_action_probs[..., dim, :] *
                                                                   next_state_log_pi[..., dim, :]).sum(dim=1)

                    # Compute target Q-value
                    next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * \
                        args.gamma * min_qf_next_target

                # Compute Q-losses for each action dimension
                qf1_loss = 0
                qf2_loss = 0
                for dim in range(len(actor.action_dims)):
                    # Get tensor for this dimension
                    qf1_values = qf1(data.observations)[dim]
                    qf2_values = qf2(data.observations)[dim]
                    qf1_a_values = qf1_values.gather(
                        1, data.actions[..., dim].unsqueeze(-1)).squeeze(-1)
                    qf2_a_values = qf2_values.gather(
                        1, data.actions[..., dim].unsqueeze(-1)).squeeze(-1)
                    qf1_loss += F.mse_loss(qf1_a_values, next_q_value)
                    qf2_loss += F.mse_loss(qf2_a_values, next_q_value)

                # Total loss is the sum of losses for each action dimension
                qf_loss = qf1_loss + qf2_loss

                q_optimizer.zero_grad()
                qf_loss.backward()
                q_optimizer.step()

                # ACTOR training (also needs to be modified for multiple dimensions)
                _, log_pi, action_probs = actor.get_action(data.observations)
                actor_loss = 0

                with torch.no_grad():
                    qf1_pi = qf1(data.observations)
                    qf2_pi = qf2(data.observations)

                for dim in range(len(actor.action_dims)):
                    min_qf_pi = torch.min(qf1_pi[dim], qf2_pi[dim])
                    actor_loss += (action_probs[..., dim, :] *
                                   (alpha * log_pi[..., dim, :] - min_qf_pi)).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                if args.autotune:
                    # re-use action probabilities for temperature loss
                    alpha_loss = (action_probs.detach(
                    ) * (-log_alpha.exp() * (log_pi + target_entropy).detach())).mean()

                    a_optimizer.zero_grad()
                    alpha_loss.backward()
                    a_optimizer.step()
                    alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(
                        args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(
                        args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values",
                                  qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values",
                                  qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss",
                                  qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss",
                                  qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss",
                                  qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss",
                                  actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar(
                    "charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss",
                                      alpha_loss.item(), global_step)

    envs.close()
    writer.close()
