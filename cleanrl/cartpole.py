# Simplified version of CartPole from Gymnasium
import math
import numpy as np
from gymnasium import Env, spaces

from gymnasium.envs.registration import register


class CartPoleEnv(Env):
    """
    Simplified CartPole environment.
    """
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self):
        super().__init__()
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masscart + self.masspole
        self.length = 0.5  # Half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # Time step

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Observation space: [cart position, cart velocity, pole angle, pole angular velocity]
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # Action space: 0 (push left) or 1 (push right)
        self.action_space = spaces.Discrete(2)

        self.state = None
        self.steps_beyond_done = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        return np.array(self.state, dtype=np.float32), {}

    def step(self, action):
        assert self.action_space.contains(action), f"{action} is invalid"
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # Equations of motion
        temp = (force + self.polemass_length *
                theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole *
                           costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        # Update state
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        self.state = (x, x_dot, theta, theta_dot)

        # Check if the episode is done
        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        reward = 1.0 if not done else 0.0

        return np.array(self.state, dtype=np.float32), reward, done, False, {}

    def render(self, mode="human"):
        # Rendering logic (e.g., using Pygame or Matplotlib)
        pass

    def close(self):
        pass


register(
    id="CustomCartPole-v0",  # Unique ID for your environment
    entry_point="cleanrl.cartpole:CartPoleEnv",  # Path to your CartPoleEnv class
)
