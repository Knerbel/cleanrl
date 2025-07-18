import random
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from dataclasses import dataclass
import tyro
import time
from torch.utils.tensorboard import SummaryWriter
import time


import cleanrl.v13.fireboy_and_watergirl_ppo_v13


@dataclass
class Args:
    exp_name: str = "PPO_RND TEST"
    """the name of this experiment"""
    env_id: str = "FireboyAndWatergirl-ppo-v13"
    """the id of the environment"""
    total_timesteps: int = 2000_000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_steps: int = 128 * 4
    """the number of steps to run in each environment per policy rollout"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    rnd_reward_scale: float = 0.1
    seed: int = 1
    """seed of the experiment"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    num_envs: int = 8
    """the number of parallel game environments"""
    track: bool = False
    """if toggled, this experiment will be tracked with TensorBoard"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""

    # Add to runtime calculations
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

# ALGO LOGIC: initialize agent here:


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(3, 32, 3, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 3, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 3 * 5, 512)),
            nn.ReLU(),
        )
        self.actor_fb = layer_init(nn.Linear(512, 4), std=0.01)
        self.actor_wg = layer_init(nn.Linear(512, 4), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_action_and_value(self, x, action=None):
        # Handle input shape for policy
        if len(x.shape) == 3:  # Single observation (H, W, C)
            x = x.unsqueeze(0)  # Add batch dimension
        # Convert from [B, H, W, C] to [B, C, H, W]
        x = x.permute(0, 3, 1, 2).float() / 255.0

        hidden = self.network(x)
        logits_fb = self.actor_fb(hidden)
        logits_wg = self.actor_wg(hidden)

        dist_fb = Categorical(logits=logits_fb)
        dist_wg = Categorical(logits=logits_wg)

        if action is None:
            action_fb = dist_fb.sample()
            action_wg = dist_wg.sample()
            action = torch.stack([action_fb, action_wg], dim=1)
        else:
            action_fb, action_wg = action[:, 0], action[:, 1]

        logprob = dist_fb.log_prob(action_fb) + dist_wg.log_prob(action_wg)
        entropy = dist_fb.entropy() + dist_wg.entropy()
        return action, logprob, entropy, self.critic(hidden)

    def get_value(self, x):
        # Handle input shape for value function
        if len(x.shape) == 3:  # Single observation (H, W, C)
            x = x.unsqueeze(0)  # Add batch dimension
        # Convert from [B, H, W, C] to [B, C, H, W]
        x = x.permute(0, 3, 1, 2).float() / 255.0
        return self.critic(self.network(x))


class RNDModel(nn.Module):
    def __init__(self, envs):
        super().__init__()
        # Same architecture for both target and predictor
        self.predictor = nn.Sequential(
            layer_init(nn.Conv2d(3, 32, 3, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 3, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 3 * 5, 512))
        )

        # Target network
        self.target = nn.Sequential(
            layer_init(nn.Conv2d(3, 32, 3, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 3, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 3 * 5, 512))
        )

        # target network is not trainable
        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, x):
        # Handle input shape for RND
        if len(x.shape) == 3:  # Single observation (H, W, C)
            x = x.unsqueeze(0)  # Add batch dimension
        # Convert from [B, H, W, C] to [B, C, H, W]
        x = x.permute(0, 3, 1, 2).float() / 255.0

        target_feature = self.target(x)
        predict_feature = self.predictor(x)
        return ((target_feature - predict_feature) ** 2).mean(dim=1)


def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)

        # Add proper seeding
        env.action_space.seed(idx)
        env.observation_space.seed(idx)

        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ResizeObservation(env, (23, 32))
        # env = gym.wrappers.FrameStack(env, 4)
        return env
    return thunk


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # Calculate runtime values

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
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name)
         for i in range(args.num_envs)],
    )
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Initialize networks
    policy = Agent(envs).to(device)
    rnd = RNDModel(envs).to(device)
    optimizer = optim.Adam([
        {'params': policy.parameters(), 'lr': args.learning_rate},
        {'params': rnd.predictor.parameters(), 'lr': args.learning_rate}
    ])

    # Storage
    obs = torch.zeros((args.num_steps, args.num_envs) +
                      envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs, 2)).to(
        device)  # 2 for both characters
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # Training loop
    obs_env, _ = envs.reset()
    episode_reward = 0

    global_step = 0
    start_time = time.time()

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(args.num_steps):
            global_step += args.num_envs

            # Convert observation to tensor
            obs[step] = torch.FloatTensor(obs_env).to(device)

            # Get actions and values
            with torch.no_grad():
                action, logprob, _, value = policy.get_action_and_value(
                    obs[step])

            # Execute action in environment
            next_obs, reward, terminated, truncated, infos = envs.step(
                action.cpu().numpy())
            done = np.logical_or(terminated, truncated)

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

            # Calculate intrinsic reward
            with torch.no_grad():
                intrinsic_reward = rnd(obs[step])

            # Store data
            actions[step] = action
            logprobs[step] = logprob
            values[step] = value.flatten()
            rewards[step] = torch.tensor(reward).to(
                device) + args.rnd_reward_scale * intrinsic_reward
            dones[step] = torch.tensor(done).to(device)

            episode_reward += reward.mean()
            obs_env = next_obs

            if done.any():
                writer.add_scalar("charts/episodic_return",
                                  episode_reward, global_step)
                print(
                    f"global_step={global_step}, episodic_return={episode_reward:.2f}")
                episode_reward = 0

        # PPO update
        with torch.no_grad():
            next_value = policy.get_value(
                torch.FloatTensor(next_obs).to(device)).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0

            # GAE calculation
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - dones[t]
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]

                delta = rewards[t] + args.gamma * \
                    nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * \
                    args.gae_lambda * nextnonterminal * lastgaelam

            # Calculate returns
            returns = advantages + values

        # Flatten batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_actions = actions.reshape(-1, 2)
        b_logprobs = logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # PPO Training
        for epoch in range(args.update_epochs):
            # Generate random indices for minibatches
            indices = torch.randperm(args.batch_size)

            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = indices[start:end]

                # Get minibatch data
                _, newlogprob, entropy, newvalue = policy.get_action_and_value(
                    b_obs[mb_inds],
                    b_actions[mb_inds]
                )

                # Policy loss
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                # Policy loss
                pg_loss1 = -b_advantages[mb_inds] * ratio
                pg_loss2 = - \
                    b_advantages[mb_inds] * \
                    torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                v_loss = 0.5 * \
                    ((newvalue.flatten() - b_returns[mb_inds]) ** 2).mean()

                # RND loss
                rnd_loss = rnd(b_obs[mb_inds]).mean()

                # Total loss
                loss = pg_loss - args.ent_coef * entropy.mean() + v_loss * \
                    args.vf_coef + rnd_loss

                # Optimization step
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    policy.parameters(), args.max_grad_norm)
                nn.utils.clip_grad_norm_(
                    rnd.predictor.parameters(), args.max_grad_norm)
                optimizer.step()

        # Log metrics
        writer.add_scalar("charts/learning_rate",
                          optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        # writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        # writer.add_scalar("losses/explained_variance",
        #                   explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))

        writer.add_scalar("charts/SPS", int(global_step /
                          (time.time() - start_time)), global_step)
        print(
            f"Learning iteration {iteration}/{args.num_iterations} completed")

    envs.close()
    writer.close()
