import random
import gymnasium as gym
import torch
import numpy as np

from ppo_atari_v17 import Agent, make_env


def evaluate(
    model_path: str,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    seed: int = 1,
    device: torch.device = torch.device("cpu"),
    capture_video: bool = False,
):

    # Set seeds for deterministic behavior
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id, 0, capture_video, run_name)])

    agent = Agent(envs).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()

    obs, _ = envs.reset()
    episodic_returns = []
    with torch.no_grad():
        while len(episodic_returns) < eval_episodes:
            actions, _, _, _ = agent.get_action_and_value(
                torch.Tensor(obs).to(device))
            next_obs, _, _, _, infos = envs.step(actions.cpu().numpy())
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if "episode" not in info:
                        continue
                    print(
                        f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
                    episodic_returns += [info["episode"]["r"]]
            obs = next_obs

    average_return = np.mean(episodic_returns)
    print(
        f"Average episodic_return over {eval_episodes} episodes: {average_return}")
    return episodic_returns


if __name__ == "__main__":
    foo = evaluate(
        model_path="PPO_best_model_exploration.pt",
        env_id="FireboyAndWatergirl-ppo-v17",
        eval_episodes=100,
        seed=1,
        run_name=f"eval"
    )

    print(foo)
