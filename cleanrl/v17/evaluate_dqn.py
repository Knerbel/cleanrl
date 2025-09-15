import random
from typing import Callable

import gymnasium as gym
import numpy as np
import torch

from dqn_atari_v17 import QNetwork, make_env


def evaluate(
    model_path: str,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    seed: int = 1,
    device: torch.device = torch.device("cpu"),
    epsilon: float = 0.05,
    capture_video: bool = True,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id, seed, 0, capture_video, run_name)])
    model = QNetwork(envs).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    obs, _ = envs.reset(seed=seed)
    episodic_returns = []
    with torch.no_grad():
        while len(episodic_returns) < eval_episodes:
            if random.random() < epsilon:
                actions = np.array([envs.single_action_space.sample()
                                   for _ in range(envs.num_envs)])
            else:
                q_value1, q_value2 = model(torch.as_tensor(
                    obs, dtype=torch.long, device=device))

                actions1 = torch.argmax(q_value1, dim=1).cpu().numpy()
                actions2 = torch.argmax(q_value2, dim=1).cpu().numpy()
                actions = np.stack([actions1, actions2], axis=-1)

            next_obs, _, _, _, infos = envs.step(actions)
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if "episode" not in info:
                        continue
                    print(
                        f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
                    episodic_returns += [info["episode"]["r"]]
            obs = next_obs

    return episodic_returns


if __name__ == "__main__":

    evaluate(
        model_path="DQN_best_model.pt",
        env_id="FireboyAndWatergirl-ppo-v17",
        eval_episodes=10,
        run_name=f"eval",
        seed=1,
        device="cpu",
        capture_video=False,
    )
