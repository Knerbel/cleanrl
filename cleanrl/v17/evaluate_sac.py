import random
from time import time
import gymnasium as gym
import torch
import numpy as np
import time

from sac_atari_v17_single_action_space import Actor, make_env


def evaluate(
    model_path: str,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    seed: int = 1,
    device: torch.device = torch.device("cpu"),
    epsilon: float = 1.0,
    capture_video: bool = False,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id, 0, capture_video, run_name)])

    # Only load the actor
    model = Actor(envs).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['actor_state_dict'])
    model.eval()

    obs, _ = envs.reset()
    episodic_returns = []
    with torch.no_grad():
        while len(episodic_returns) < eval_episodes:
            if random.random() < epsilon:
                actions = np.array([envs.single_action_space.sample()
                                   for _ in range(envs.num_envs)])
            else:
                # Get action probabilities and take most likely action
                # with torch.no_grad():
                _, _, action_probs = model.get_action(
                    torch.Tensor(obs).to(device))
                actions = torch.argmax(action_probs, dim=1).cpu().numpy()
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
    start_time = time.time()

    foo = evaluate(
        model_path="SAC_single_best_model_exploration.pt",
        env_id="FireboyAndWatergirl_sac-v17",
        eval_episodes=10,
        run_name=f"eval"
    )

    end_time = time.time()
    execution_time = end_time - start_time

    print(f"\nExecution time: {execution_time:.2f} seconds")
