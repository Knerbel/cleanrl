import os
import random

import gymnasium as gym
import numpy as np
import torch

from cleanrl.v17.dqn_atari_v17 import QNetwork, make_env
from heatmaps.heatmap import create_heatmap


def evaluate(
    model_path: str,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    start_seed: int = 1,
    device: torch.device = torch.device("cpu"),
    epsilon: float = 0.05,
    capture_video: bool = True,
):
    episodic_returns = []
    final_unique_positions = []
    final_finished = []
    final_players_at_door = []
    final_times_in_water = []
    final_times_in_fire = []
    final_times_in_goo = []
    final_stars_collected = []

    # Run evaluation for each seed
    for seed in range(start_seed, start_seed + eval_episodes):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

        envs = gym.vector.SyncVectorEnv(
            [make_env(env_id, seed, capture_video, run_name)])
        model = QNetwork(envs).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        obs, _ = envs.reset(seed=seed)
        episode_return = None

        with torch.no_grad():
            while episode_return is None:
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
                        #     f"eval_episode={len(episodic_returns)}, seed={seed}, episodic_return={info['episode']['r']}")
                        episode_return = info["episode"]["r"]
                        final_finished.append(info['finished'])
                        final_players_at_door.append(info['players_at_door'])
                        final_unique_positions.append(info['unique_positions'])
                        final_stars_collected.append(info['stars_collected'])
                        final_times_in_water.append(info['times_in_water'])
                        final_times_in_fire.append(info['times_in_fire'])
                        final_times_in_goo.append(info['times_in_goo'])

                        episodic_returns.append(episode_return)
                obs = next_obs

        envs.close()

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


def clear_image_folder(folder_path: str):
    """Delete all image files in the specified folder."""
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            file_path = os.path.join(folder_path, filename)
            try:
                os.remove(file_path)
                # print(f"Deleted: {filename}")
            except Exception as e:
                print(f"Error deleting {filename}: {e}")


if __name__ == "__main__":
    final_runs = "C:\\Users\\knerb\\Documents\\GitHub\\Knerbel-cleanrl\\episode_images"
    training_runs = "C:\\Users\\knerb\\Documents\\Masterthesis\\final runs\\"

    evaluate(
        model_path=f"{training_runs}level8_exploration\\DQN\\DQN 1757925424\\DQN_best_model.pt",
        env_id="FireboyAndWatergirl-ppo-v17-exploration",
        eval_episodes=100,
        epsilon=1,
        run_name="Random_exploration_best"
    )
    create_heatmap(f"{final_runs}", name="Random_exploration")
    clear_image_folder(final_runs)
    # Exploration
    # evaluate(
    #     model_path=f"{training_runs}level8_exploration\\DQN\\DQN 1757925424\\DQN_best_model.pt",
    #     env_id="FireboyAndWatergirl-ppo-v17-exploration",
    #     eval_episodes=100,
    #     run_name="DQN_exploration_best"
    # )
    # create_heatmap(f"{final_runs}", name="DQN_exploration_best")
    # clear_image_folder(final_runs)

    # evaluate(
    #     model_path=f"{training_runs}level8_exploration\\DQN\\DQN 1757925424\\DQN_best_n_model.pt",
    #     env_id="FireboyAndWatergirl-ppo-v17-exploration",
    #     eval_episodes=100,
    #     run_name="DQN_exploration_best_n"
    # )
    # create_heatmap(f"{final_runs}",
    #                name="DQN_exploration_best_n_model")
    # clear_image_folder(final_runs)
#
    # Obstacles
    # evaluate(
    #    model_path=f"{training_runs}level8_obstacles\\DQN\\DQN 1757938758\\DQN_best_model.pt",
    #    env_id="FireboyAndWatergirl-ppo-v17-obstacles",
    #    eval_episodes=100,
    #    run_name="DQN_obstacles_best"
    # )
    # create_heatmap(f"{final_runs}", name="DQN_obstacles_best")
    # clear_image_folder(final_runs)
#
    # evaluate(
    #    model_path=f"{training_runs}level8_obstacles\\DQN\\DQN 1757938758\\DQN_best_n_model.pt",
    #    env_id="FireboyAndWatergirl-ppo-v17-obstacles",
    #    eval_episodes=100,
    #    run_name="DQN_obstacles_best_n"
    # )
    # create_heatmap(f"{final_runs}",
    #               name="DQN_obstacles_best_n_model")
    # clear_image_folder(final_runs)
#
    # Stars
    # evaluate(
    #    model_path=f"{training_runs}level8_stars\\DQN\\DQN 1757920887\\DQN_best_model.pt",
    #    env_id="FireboyAndWatergirl-ppo-v17-stars",
    #    eval_episodes=100,
    #    run_name="DQN_stars_best"
    # )
    # create_heatmap(f"{final_runs}", name="DQN_stars_best")
    # clear_image_folder(final_runs)
#
    # evaluate(
    #    model_path=f"{training_runs}level8_stars\\DQN\\DQN 1757920887\\DQN_best_n_model.pt",
    #    env_id="FireboyAndWatergirl-ppo-v17-stars",
    #    eval_episodes=100,
    #    run_name="DQN_stars_best_n"
    # )
    # create_heatmap(f"{final_runs}",
    #               name="DQN_stars_best_n_model")
    # clear_image_folder(final_runs)
#
    # Plates and gates
    # evaluate(
    #    model_path=f"{training_runs}level8_plates_and_gates\\DQN\\DQN 1757934218\\DQN_best_model.pt",
    #    env_id="FireboyAndWatergirl-ppo-v17-plates_and_gates",
    #    eval_episodes=100,
    #    run_name="DQN_plates_and_gates_best"
    # )
    # create_heatmap(f"{final_runs}",
    #               name="DQN_plates_and_gates_best")
    # clear_image_folder(final_runs)

    # evaluate(
    #     model_path=f"{training_runs}level8_plates_and_gates\\DQN\\DQN 1757934218\\DQN_best_n_model.pt",
    #     env_id="FireboyAndWatergirl-ppo-v17-plates_and_gates",
    #     eval_episodes=100,
    #     run_name="DQN_plates_and_gates_best_n"
    # )
    # create_heatmap(f"{final_runs}",
    #                name="DQN_plates_and_gates_best_n_model")
    # clear_image_folder(final_runs)

    # Combined
    # evaluate(
    #    model_path=f"{training_runs}level8_combined\\DQN\\DQN 1758085360\\DQN_best_model.pt",
    #    env_id="FireboyAndWatergirl-ppo-v17-combined",
    #    eval_episodes=100,
    #    run_name="DQN_combined_best"
    # )
    # create_heatmap(f"{final_runs}",
    #               name="DQN_combined_best")
    # clear_image_folder(final_runs)
#
    # evaluate(
    #    model_path=f"{training_runs}level8_combined\\DQN\\DQN 1758085360\\DQN_best_n_model.pt",
    #    env_id="FireboyAndWatergirl-ppo-v17-combined",
    #    eval_episodes=100,
    #    run_name="DQN_combined_best_n"
    # )
    # create_heatmap(f"{final_runs}",
    #               name="DQN_combined_best_n_model")
    # clear_image_folder(final_runs)
#
    # evaluate(
    #    model_path=f"{training_runs}level8_combined\\DQN\\DQN 1758090065 pretrained\\DQN_best_model.pt",
    #    env_id="FireboyAndWatergirl-ppo-v17-combined",
    #    eval_episodes=100,
    #    run_name="DQN_Pretrained_combined_best"
    # )
    # create_heatmap(f"{final_runs}",
    #               name="DQN_Pretrained_combined_best")
    # clear_image_folder(final_runs)
#
    # evaluate(
    #    model_path=f"{training_runs}level8_combined\\DQN\\DQN 1758090065 pretrained\\DQN_best_n_model.pt",
    #    env_id="FireboyAndWatergirl-ppo-v17-combined",
    #    eval_episodes=100,
    #    run_name="DQN_Pretrained_combined_best_n"
    # )
    # create_heatmap(f"{final_runs}",
    #               name="DQN_Pretrained_combined_best_n_model")
    # clear_image_folder(final_runs)
