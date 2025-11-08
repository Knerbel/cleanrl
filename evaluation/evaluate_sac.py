import os
import random

import gymnasium as gym
import numpy as np
import torch

from cleanrl.v17.sac_atari_v17_single_action_space import Actor, make_env
from heatmaps.heatmap import create_heatmap


def evaluate(
    model_path: str,
    env_id: str,
    eval_episodes: int,
    run_name: str = F"eval",
    start_seed: int = 1,
    device: torch.device = torch.device("cpu"),
    epsilon: float = 0.05,
    capture_video: bool = False,
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

        # Only load the actor
        model = Actor(envs).to(device)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['actor_state_dict'])
        model.eval()

        obs, _ = envs.reset()
        episode_return = None

        with torch.no_grad():
            while episode_return is None:
                if random.random() < epsilon:
                    actions = np.array([envs.single_action_space.sample()
                                        for _ in range(envs.num_envs)])
                else:
                    _, _, action_probs = model.get_action(
                        torch.Tensor(obs).to(device))
                    actions = torch.argmax(action_probs, dim=1).cpu().numpy()
                next_obs, _, _, _, infos = envs.step(actions)
                if "final_info" in infos:
                    for info in infos["final_info"]:
                        if "episode" not in info:
                            continue
                        # print(
                        #     f"eval_episode={len(episodic_returns)}, seed={seed}, episodic_return={info['episode']['r']}")
                        episode_return = info["episode"]["r"]
                        episodic_returns.append(episode_return)
                        final_finished.append(info['finished'])
                        final_players_at_door.append(info['players_at_door'])
                        final_unique_positions.append(info['unique_positions'])
                        final_stars_collected.append(info['stars_collected'])
                        final_times_in_water.append(info['times_in_water'])
                        final_times_in_fire.append(info['times_in_fire'])
                        final_times_in_goo.append(info['times_in_goo'])
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

    # Exploration
    evaluate(
        # model_path="SAC_single_best_model_exploration.pt",

        model_path=f"{training_runs}level8_exploration\\SAC_Single_Action_Space\\SAC 1758259390\\SAC_best_model.pt",
        env_id="FireboyAndWatergirl_sac-v17-exploration",
        eval_episodes=100,
        run_name="SAC_evaluation_exploration_best"
    )
    create_heatmap(f"{final_runs}", name="SAC_evaluation_exploration_best")
    clear_image_folder(final_runs)

    evaluate(
        model_path=f"{training_runs}level8_exploration\\SAC_Single_Action_Space\\SAC 1758259390\\SAC_best_n_model.pt",
        env_id="FireboyAndWatergirl_sac-v17-exploration",
        eval_episodes=100,
        run_name="SAC_evaluation_exploration_best_n_model"
    )
    create_heatmap(f"{final_runs}",
                   name="SAC_evaluation_exploration_best_n_model")
    clear_image_folder(final_runs)

    # # Obstacles
    evaluate(
        model_path=f"{training_runs}level8_obstacles\\SAC_single_action_space\\SAC other run 1758269897\\SAC_best_model.pt",
        env_id="FireboyAndWatergirl_sac-v17-obstacles",
        eval_episodes=100,
        run_name="SAC_evaluation_obstacles_best"
    )
    create_heatmap(f"{final_runs}", name="SAC_evaluation_obstacles_best")
    clear_image_folder(final_runs)

    evaluate(
        model_path=f"{training_runs}level8_obstacles\\SAC_single_action_space\\SAC other run 1758269897\\SAC_best_n_model.pt",
        env_id="FireboyAndWatergirl_sac-v17-obstacles",
        eval_episodes=100,
        run_name="SAC_evaluation_obstacles_best_n_model"
    )
    create_heatmap(f"{final_runs}",
                   name="SAC_evaluation_obstacles_best_n_model")
    clear_image_folder(final_runs)

    # # Stars
    evaluate(
        model_path=f"{training_runs}level8_stars\\SAC_single_action_space\\SAC_best_model.pt",
        env_id="FireboyAndWatergirl_sac-v17-stars",
        eval_episodes=100,
        run_name="SAC_evaluation_stars_best"
    )
    create_heatmap(f"{final_runs}", name="SAC_evaluation_stars_best")
    clear_image_folder(final_runs)

    evaluate(
        model_path=f"{training_runs}level8_stars\\SAC_single_action_space\\SAC_best_n_model.pt",
        env_id="FireboyAndWatergirl_sac-v17-stars",
        eval_episodes=100,
        run_name="SAC_evaluation_stars_best_n_model"
    )
    create_heatmap(f"{final_runs}",
                   name="SAC_evaluation_stars_best_n_model")
    clear_image_folder(final_runs)

    # Plates and gates
    evaluate(
        model_path=f"{training_runs}level8_plates_and_gates\\SAC_single_action_space\\SAC 1758519205\\SAC_best_model.pt",
        env_id="FireboyAndWatergirl_sac-v17-plates_and_gates",
        eval_episodes=100,
        run_name="SAC_evaluation_plates_and_gates_best"
    )
    create_heatmap(f"{final_runs}",
                   name="SAC_evaluation_plates_and_gates_best")
    clear_image_folder(final_runs)

    evaluate(
        model_path=f"{training_runs}level8_plates_and_gates\\SAC_single_action_space\\SAC 1758519205\\SAC_best_n_model.pt",
        env_id="FireboyAndWatergirl_sac-v17-plates_and_gates",
        eval_episodes=100,
        run_name="SAC_evaluation_plates_and_gates_best_n_model"
        #
    )
    create_heatmap(f"{final_runs}",
                   name="SAC_evaluation_plates_and_gates_best_n_model")
    clear_image_folder(final_runs)

    # Combined
    evaluate(
        model_path=f"{training_runs}level8_combined\\SAC_single_action_space\\SAC_best_model.pt",
        env_id="FireboyAndWatergirl_sac-v17-combined",
        eval_episodes=100,
        run_name="SAC_evaluation_combined_best"
    )
    create_heatmap(f"{final_runs}",
                   name="SAC_evaluation_combined_best")
    clear_image_folder(final_runs)

    evaluate(
        model_path=f"{training_runs}level8_combined\\SAC_single_action_space\\SAC_best_n_model.pt",
        env_id="FireboyAndWatergirl_sac-v17-combined",
        eval_episodes=100,
        run_name="SAC_evaluation_combined_best_n"
    )
    create_heatmap(f"{final_runs}",
                   name="SAC_evaluation_combined_best_n_model")
    clear_image_folder(final_runs)

    # Combined pretrained
    evaluate(
        model_path=f"{training_runs}level8_combined\\SAC_single_action_space\\1758864584 pretrained\\SAC_best_model.pt",
        env_id="FireboyAndWatergirl_sac-v17-combined",
        eval_episodes=100,
        run_name="SAC_evaluation_combined_best"
    )
    create_heatmap(f"{final_runs}",
                   name="SAC_evaluation_pretrained_combined_best")
    clear_image_folder(final_runs)

    # evaluate(
    #     model_path=f"{training_runs}level8_combined\\SAC_single_action_space\\1758864584 pretrained\\SAC_best_n_model.pt",
    #     env_id="FireboyAndWatergirl_sac-v17-combined",
    #     eval_episodes=100,
    #     run_name="SAC_evaluation_pretrained_combined_best_n"
    # )
    # create_heatmap(f"{final_runs}",
    #                name="SAC_evaluation_pretrained_combined_best_n_model")
    # clear_image_folder(final_runs)
