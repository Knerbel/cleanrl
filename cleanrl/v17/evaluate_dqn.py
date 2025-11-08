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
    final_unique_positions = []
    final_finished = []
    final_players_at_door = []
    final_times_in_water = []
    final_times_in_fire = []
    final_times_in_goo = []
    final_stars_collected = []
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
                    # print(
                    #     f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
                    episodic_returns += [info["episode"]["r"]]
                    final_finished.append(info['finished'])
                    final_players_at_door.append(info['players_at_door'])
                    final_unique_positions.append(info['unique_positions'])
                    final_stars_collected.append(info['stars_collected'])
                    final_times_in_water.append(info['times_in_water'])
                    final_times_in_fire.append(info['times_in_fire'])
                    final_times_in_goo.append(info['times_in_goo'])
            obs = next_obs

    average_return = np.mean(episodic_returns)
    # Calculate statistics
    metrics = {
        "unique_positions": {
            "max": max(final_unique_positions) if final_unique_positions else 0,
            "mean": np.mean(final_unique_positions) if final_unique_positions else 0,
            "std": np.std(final_unique_positions) if final_unique_positions else 0
        },
        "finished": {
            "max": max(final_finished) if final_finished else 0,
            "mean": np.mean(final_finished) if final_finished else 0,
            "std": np.std(final_finished) if final_finished else 0
        },
        "players_at_door": {
            "max": max(final_players_at_door) if final_players_at_door else 0,
            "mean": np.mean(final_players_at_door) if final_players_at_door else 0,
            "std": np.std(final_players_at_door) if final_players_at_door else 0
        },
        "times_in_water": {
            "max": max(final_times_in_water) if final_times_in_water else 0,
            "mean": np.mean(final_times_in_water) if final_times_in_water else 0,
            "std": np.std(final_times_in_water) if final_times_in_water else 0
        },
        "times_in_fire": {
            "max": max(final_times_in_fire) if final_times_in_fire else 0,
            "mean": np.mean(final_times_in_fire) if final_times_in_fire else 0,
            "std": np.std(final_times_in_fire) if final_times_in_fire else 0
        },
        "times_in_goo": {
            "max": max(final_times_in_goo) if final_times_in_goo else 0,
            "mean": np.mean(final_times_in_goo) if final_times_in_goo else 0,
            "std": np.std(final_times_in_goo) if final_times_in_goo else 0
        },
        "stars_collected": {
            "max": max(final_stars_collected) if final_stars_collected else 0,
            "mean": np.mean(final_stars_collected) if final_stars_collected else 0,
            "std": np.std(final_stars_collected) if final_stars_collected else 0
        }
    }

    # Print statistics
    print("\n"+run_name+" evaluation Metrics:")
    print("-" * 50)
    print(f"{'Metric':<20} {'Max':>8} {'Mean':>8} {'Std':>8}")
    print("-" * 50)
    for metric_name, values in metrics.items():
        print(f"{metric_name.replace('_', ' ').title():<20} "
              f"{float(values['max']):8.2f} "
              f"{float(values['mean']):8.2f} "
              f"{float(values['std']):8.2f}")
    print("-" * 50)
    print(
        f"\nAverage episodic_return over {eval_episodes} episodes: {average_return:.2f}")

    return episodic_returns, metrics


if __name__ == "__main__":

    evaluate(
        model_path="DQN_best_n_model.pt",
        env_id="FireboyAndWatergirl-ppo-v17",
        eval_episodes=10,
        run_name=f"eval",
        seed=1,
        device="cpu",
        capture_video=False,
    )
