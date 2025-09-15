# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqn_ataripy
from collections import deque
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
from torch.utils.tensorboard import SummaryWriter

from cleanrl_utils.buffers import ReplayBuffer

import cleanrl.v17.fireboy_and_watergirl_ppo_v17


@dataclass
class Args:
    exp_name: str = "DQN_atari_v17_stars"
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
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "FireboyAndWatergirl-ppo-v17"
    """the id of the environment"""
    total_timesteps: int = 350_000
    """total timesteps of the experiments"""
    learning_rate: float = 1e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 1000000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 1000
    """the timesteps it takes to update the target network"""
    batch_size: int = 32
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.01
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.20
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 80000
    """timestep to start learning"""
    train_frequency: int = 4
    """the frequency of training"""


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ResizeObservation(env, (18, 18))
        env = gym.wrappers.FrameStack(env, 4)
        env.action_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        num_tile_types = int(env.single_observation_space.high.max()) + 1
        embedding_dim = 8
        frames = env.single_observation_space.shape[0]
        self.embedding = nn.Embedding(num_tile_types, embedding_dim)
        self.network = nn.Sequential(
            nn.Conv2d(frames * embedding_dim, 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # dummy = torch.zeros(1, frames, height, width)
        # dummy = self.embedding(dummy.long())
        # dummy = dummy.permute(0, 1, 4, 2, 3).reshape(
        #     1, frames * embedding_dim, height, width)
        # out_dim = self.network(dummy).shape[1]
        # print(out_dim)

        # Feature extraction
        self.fc = nn.Sequential(
            nn.Linear(64, 512),
            nn.ReLU(),
        )
        # Q-Value for each action
        self.head1 = nn.Linear(512, env.single_action_space.nvec[0])
        self.head2 = nn.Linear(512, env.single_action_space.nvec[1])

    def forward(self, x):
        # x: (batch, 4, 18, 18)
        x = self.embedding(x.long())
        x = x.permute(0, 1, 4, 2, 3).reshape(
            x.shape[0], -1, x.shape[2], x.shape[3])
        x = self.network(x)
        x = self.fc(x)
        q1 = self.head1(x)
        q2 = self.head2(x)
        return q1, q2  # Each: (batch, 4)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    args = tyro.cli(Args)
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
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
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name)
         for i in range(args.num_envs)]
    )
    q_network = QNetwork(envs).to(device)

    # pretrained_model_path = "DQN_best_model_stars.pt"
    # q_network.load_state_dict(torch.load(
    #     pretrained_model_path, map_location=device))

    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        optimize_memory_usage=True
    )
    start_time = time.time()

    n = 40  # window size for averaging
    recent_returns = deque(maxlen=n)
    best_avg_return = -float('inf')
    best_return = -float('inf')

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(
            args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        # print(linear_schedule(
        #     args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, 340000))
        if random.random() < epsilon:
            actions = np.stack([
                np.random.randint(
                    envs.single_action_space.nvec[0], size=args.num_envs),
                np.random.randint(
                    envs.single_action_space.nvec[1], size=args.num_envs)
            ], axis=-1)
        else:
            q_value1, q_value2 = q_network(torch.as_tensor(
                obs, dtype=torch.long, device=device))
            actions1 = torch.argmax(q_value1, dim=1).cpu().numpy()
            actions2 = torch.argmax(q_value2, dim=1).cpu().numpy()
            actions = np.stack([actions1, actions2], axis=-1)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(
            actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(
                        f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar(
                        "charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                    writer.add_scalar("charts/episodic_return",
                                      info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length",
                                      info["episode"]["l"], global_step)
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
                    episode_return = info["episode"]["r"]

                    if episode_return > best_return:
                        best_return = episode_return
                        torch.save(q_network.state_dict(),
                                   f"DQN_best_model.pt")

                    recent_returns.append(episode_return)
                    if len(recent_returns) == n:
                        avg_return = sum(recent_returns) / n
                        if avg_return > best_avg_return:
                            best_avg_return = avg_return
                            torch.save(q_network.state_dict(),
                                       f"DQN_best_n_model.pt")

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                q_value1, q_value2 = q_network(data.observations)
                old_val1 = q_value1.gather(1, data.actions[:, [0]])
                old_val2 = q_value2.gather(1, data.actions[:, [1]])
                with torch.no_grad():
                    target_q1, target_q2 = target_network(
                        data.next_observations)
                    target_max1 = target_q1.max(dim=1)[0]
                    target_max2 = target_q2.max(dim=1)[0]
                    td_target = data.rewards.flatten() + args.gamma * (target_max1 +
                                                                       target_max2) * (1 - data.dones.flatten())
                loss = F.mse_loss(
                    td_target, old_val1.squeeze() + old_val2.squeeze())

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update target network
            if global_step % args.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data +
                        (1.0 - args.tau) * target_network_param.data
                    )

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(q_network.state_dict(), model_path)
        print(f"model saved to {model_path}")
        from cleanrl_utils.evals.dqn_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=QNetwork,
            device=device,
            epsilon=args.end_e,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(args, episodic_returns, repo_id, "DQN",
                        f"runs/{run_name}", f"videos/{run_name}-eval")

    envs.close()
    writer.close()
