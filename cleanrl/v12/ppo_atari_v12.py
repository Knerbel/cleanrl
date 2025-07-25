# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_ataripy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

# Import your Fireboy and Watergirl environment to ensure it's registered
# import cleanrl.fireboy_and_watergirl_ppo
import cleanrl.v9.fireboy_and_watergirl_ppo_random_baseline
import cleanrl.v9.fireboy_and_watergirl_ppo_v9
import cleanrl.v9.fireboy_and_watergirl_ppo_v9_wo_observation_space
import cleanrl.v10.fireboy_and_watergirl_ppo_v10
import cleanrl.v11.fireboy_and_watergirl_ppo_v11
import cleanrl.v12.fireboy_and_watergirl_ppo_v12


@dataclass
class Args:
    exp_name: str = "level0_boxes"
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
    env_id: str = 'FireboyAndWatergirl-ppo-v12'
    """the id of the environment"""
    total_timesteps: int = 2000_000
    """total timesteps of the experiments"""
    learning_rate: float = 1 * 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 8
    """the number of parallel game environments"""
    num_steps: int = 128 * 4
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        # env = MaxAndSkipEnv(env, skip=4)
        # env = EpisodicLifeEnv(env)
        # if "FIRE" in env.unwrapped.get_action_meanings():
        #     env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (23, 32))
        # env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        # For RGB input, input channels = 3 * 4 = 12 (4 stacked RGB frames)
        self.network = nn.Sequential(
            # (12, 23, 34) -> (64, 11, 16)
            layer_init(nn.Conv2d(12, 64, 3, stride=2)),
            nn.ReLU(),
            # (64, 11, 16) -> (128, 5, 7)
            layer_init(nn.Conv2d(64, 128, 3, stride=2)),
            nn.ReLU(),
            # (128, 5, 7) -> (128, 3, 5)
            layer_init(nn.Conv2d(128, 128, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(128 * 3 * 5, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, 8), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def forward(self, x):
        # x shape: (batch, 4, 23, 34, 3)
        x = x.permute(0, 1, 4, 2, 3).reshape(
            x.shape[0], -1, x.shape[2], x.shape[3])
        return self.network(x / 255.0)

    def get_value(self, x: torch.Tensor):
        # x shape: (batch, 4, 23, 34, 3) from env, need to reshape to (batch, 12, 23, 34)
        x = x.permute(0, 1, 4, 2, 3).reshape(
            x.shape[0], -1, x.shape[2], x.shape[3])
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, x: torch.Tensor, action=None):
        x = x.permute(0, 1, 4, 2, 3).reshape(
            x.shape[0], -1, x.shape[2], x.shape[3])
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        logits1, logits2 = logits.split(4, dim=-1)
        dist1 = Categorical(logits=logits1)
        dist2 = Categorical(logits=logits2)
        if action is None:
            action1 = dist1.sample()
            action2 = dist2.sample()
            action = torch.stack([action1, action2], dim=-1)
        else:
            action1, action2 = action[..., 0], action[..., 1]
        logprob = dist1.log_prob(
            action[..., 0]) + dist2.log_prob(action[..., 1])
        entropy = dist1.entropy() + dist2.entropy()
        return action, logprob, entropy, self.critic(hidden)


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    # run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    run_name = f"{args.env_id}_{args.exp_name}_{args.seed}_{args.total_timesteps}_{args.learning_rate}_{args.num_envs}_{args.num_steps}_{args.anneal_lr}_{args.gamma}_{args.gae_lambda}_{args.num_minibatches}_{args.update_epochs}_{args.norm_adv}_{args.clip_coef}_{args.clip_vloss}_{args.ent_coef}_{args.vf_coef}_{args.max_grad_norm}_{args.target_kl}_{int(time.time())}"

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
        [make_env(args.env_id, i, args.capture_video, run_name)
         for i in range(args.num_envs)],
    )
    # assert isinstance(envs.single_action_space,
    #                   gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # dummy_input = torch.randn(1, 4, 23, 34, 3).to(device)
    # torch.onnx.export(
    #     agent,
    #     dummy_input,
    #     "ppo_agent.onnx",
    #     input_names=["input"],
    #     output_names=["output"],
    #     opset_version=11
    # )
    # print("Exported model to ppo_agent.onnx")

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) +
                      envs.single_observation_space.shape).to(device)
    # actions = torch.zeros((args.num_steps, args.num_envs) +
    #                       envs.single_action_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs, 2)).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs

            # Inside the training loop, before calling the agent
            if (global_step//args.num_envs) % 100 == 0:  # Save every 1000 steps
                # Convert the observation tensor to a NumPy array
                # Take the first environment's observation
                obs_image = next_obs[0].cpu().numpy()

                # Reshape and normalize the observation for visualization
                # Convert from (C, H, W) to (H, W, C)
                obs_image = obs_image[0]  # .transpose(1, 2, 0)
                # Normalize to [0, 1] for visualization
                obs_image = obs_image / 255.0

                # Plot and save the image
                plt.figure(figsize=(10, 3))

                plt.imshow(obs_image, cmap="gray")
                plt.axis("off")
                plt.savefig(
                    f"agent_input_step.png", bbox_inches="tight", pad_inches=0)
                plt.close()

            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(
                    next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(
                action.cpu().numpy())

            # Add logging for exploration metrics
            #
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(
                device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(
                            f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar(
                            "charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar(
                            "charts/episodic_length", info["episode"]["l"], global_step)
                        writer.add_scalar(
                            "charts/stars_collected", info["stars_collected"], global_step)
                        writer.add_scalar(
                            "charts/zero_reward", info["zero_reward"], global_step)
                        writer.add_scalar(
                            "charts/unique_positions", info["unique_positions"], global_step)
                        writer.add_scalar(
                            "charts/finished", info["finished"], global_step)
                        writer.add_scalar(
                            "charts/players_at_door", info["players_at_door"], global_step)
                        writer.add_scalar(
                            "charts/times_in_water", info["times_in_water"], global_step)
                        writer.add_scalar(
                            "charts/times_in_fire", info["times_in_fire"], global_step)
                        writer.add_scalar(
                            "charts/times_in_goo", info["times_in_goo"], global_step)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * \
                    nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * \
                    args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1, 2))  # Changed this line
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(b_obs.shape[0])  # Changed this line
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]  # Changed this line
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() >
                                   args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (
                        mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * \
                    torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * \
                        ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - \
            np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate",
                          optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl",
                          old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance",
                          explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step /
                          (time.time() - start_time)), global_step)
        print(
            f"Learning iteration {iteration}/{args.num_iterations} completed")

    envs.close()
    writer.close()
