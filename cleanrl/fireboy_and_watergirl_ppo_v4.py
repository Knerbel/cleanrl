import random
import gymnasium as gym
from gymnasium import spaces
from matplotlib import pyplot as plt
import numpy as np
from gymnasium.envs.registration import register

from fireboy_and_watergirl.board import Board
from fireboy_and_watergirl.character import FireBoy, WaterGirl
from fireboy_and_watergirl.doors import FireDoor, WaterDoor
from fireboy_and_watergirl.game import Game
from fireboy_and_watergirl.gates import Gates
from fireboy_and_watergirl.stars import Stars


class FireboyAndWatergirlEnv(gym.Env):
    """
    Custom Environment for Fireboy and Watergirl that follows the Gymnasium API.
    """
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self):
        super(FireboyAndWatergirlEnv, self).__init__()

        # 4 actions for each character
        self.action_space = spaces.MultiDiscrete([4, 4])
        # Initialize game components
        self.available_levels = ["level1_empty1",
                                 "level1_empty2", "level1_empty3"]  # Add your level names
        self.level = random.choice(self.available_levels)
        self.episode_window = 32  # Number of episodes to calculate success rate
        self.episode_results = []  # Track success/failure of episodes

        self.game = Game()  # Instantiate the Game class
        self.board = None
        self.fire_boy = None
        self.water_girl = None
        self.gates = None
        self.doors = None

        # Initialize game state
        self.state = None
        self.done = False
        self._load_level()

        self.steps = 0
        self.max_steps = 128  # 400
        self.envs = 8

        self.level_height = 25 - 2  # Assuming 1-tile border on top and bottom
        self.level_width = 34 - 2   # Assuming 1-tile border on left and right
        self.num_channels = 3       # RGB channels
        # Define the flattened observation space for 3 stacked frames
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.level_height, self.level_width, self.num_channels),
            dtype=np.uint8
        )

    def get_action_meanings(self):
        return [
            "NOOP",
            "Left",
            "Right",
            "Up",
        ]

    def _load_level(self):
        """
        Load the level data from the file and dynamically set up the game components.
        """

        self.level = random.choice(self.available_levels)
        # Read the level data from the file
        with open('./fireboy_and_watergirl/data/'+self.level+'.txt', 'r') as file:
            level_data = [line.strip().split(',') for line in file.readlines()]

        # Initialize game components
        self.board = Board('./fireboy_and_watergirl/data/'+self.level+'.txt')
        self.gates: list[Gates] = []
        self.doors: list[FireDoor | WaterDoor] = []
        self.stars: list[Stars] = []
        self.fire_boy: FireBoy = None
        self.water_girl: WaterGirl = None

        for y, row in enumerate(level_data):
            for x, tile in enumerate(row):
                pos = (x * 16, y * 16)
                if tile == 'A':
                    self.doors.append(FireDoor(pos))
                elif tile == 'B':
                    self.doors.append(WaterDoor(pos))
                elif tile == 'f':
                    self.fire_boy = FireBoy(pos)
                elif tile == 'w':
                    self.water_girl = WaterGirl(pos)
                elif tile == 'a':
                    self.stars.append(Stars([*pos], "fire"))
                elif tile == 'b':
                    self.stars.append(Stars([*pos], "water"))

    def reset(self, seed=None, options=None):
        """
        Reset the environment to its initial state and return the initial observation.
        """
        super().reset(seed=seed)

        self._load_level()
        self.state = self._get_state()
        self.done = False
        self.steps = 0

        # Reset cumulative rewards
        if not hasattr(self, "cumulative_rewards"):
            self.cumulative_rewards = np.zeros(self.envs)
        self.cumulative_rewards[:] = 0

        return self.state, {}

    def step(self, action):
        """
        Apply the given action to the environment and return the results.
        """
        # Apply the action to the characters
        self._apply_action(action)
        # Update the game state
        self.state = self._get_state()

        # Compute reward
        reward = self._compute_reward()

        # Update cumulative rewards for this environment
        if not hasattr(self, "cumulative_rewards"):
            self.cumulative_rewards = np.zeros(self.envs)
        self.cumulative_rewards[self.game.index] += reward

        # Check if the game is done
        self.done = self._check_done()
        if self.done:
            # print('DONE!')
            self.draw_observation(self.state)
            # Track if episode was successful (all stars collected)

        self.steps += 1
        if self.steps == self.max_steps and self.game.index % self.envs == 0:
            self._get_state(draw=True)
        if self.steps >= self.max_steps:
            self.done = True

        if self.done:
            # Identify the best environment based on cumulative rewards
            best_env_index = np.argmax(self.cumulative_rewards)
            # Draw the observation of the best-performing environment
            if self.game.index == best_env_index:
                self.draw_observation(self.state)

            success = self._check_done() and self.steps < self.max_steps
            self.episode_results.append(1 if success else 0)

        # Optionally, provide additional info
        info = {}
        return self.state, reward, self.done, False, info

    def draw_observation(self, observation):
        """
        Draw the observation as an image and save it.
        """
        # Ensure the observation is in the correct shape (H, W, C)
        if len(observation.shape) == 3:
            rgb_image = observation
        else:
            raise ValueError("Observation must have 3 dimensions (H, W, C).")
        plt.figure(figsize=(8, 6))
        plt.imshow(rgb_image)
        plt.axis("off")
        plt.title(f"Best Environment - Episode ")
        plt.savefig(f"best_env_episode.png", bbox_inches="tight", pad_inches=0)
        plt.close()

    def render(self, mode="human"):
        """
        Render the environment to the screen or other output.
        Shows only the RGB observation used by the agent.
        """
        if mode == "human":
            rgb_image = self._get_state()
            if not hasattr(FireboyAndWatergirlEnv, 'plt_initialized'):
                print("Initializing matplotlib...")
                import matplotlib
                matplotlib.use('TkAgg')
                import matplotlib.pyplot as plt
                plt.ion()
                FireboyAndWatergirlEnv.matplotlib = matplotlib
                FireboyAndWatergirlEnv.plt = plt
                FireboyAndWatergirlEnv.plt_initialized = True
                FireboyAndWatergirlEnv.render_fig = plt.figure(
                    figsize=(8, 6), num="Fireboy & Watergirl")
                FireboyAndWatergirlEnv.render_ax = FireboyAndWatergirlEnv.render_fig.add_subplot(
                    111)
                FireboyAndWatergirlEnv.render_img = FireboyAndWatergirlEnv.render_ax.imshow(
                    rgb_image)
                FireboyAndWatergirlEnv.render_ax.axis('off')
                FireboyAndWatergirlEnv.render_fig.tight_layout()
                # plt.show(block=False)

            # Update the existing static figure
            FireboyAndWatergirlEnv.render_img.set_data(rgb_image)

            # Update statistics
            reward = self._compute_reward()
            num_stars_collected = sum(s.is_collected for s in self.stars)
            FireboyAndWatergirlEnv.render_fig.suptitle(
                f"Step: {self.steps} | Stars: {num_stars_collected}/{len(self.stars)} | Reward: {reward}",
                color="red"
            )

            # Process events but handle errors silently
            try:
                FireboyAndWatergirlEnv.render_fig.canvas.draw_idle()
                FireboyAndWatergirlEnv.render_fig.canvas.flush_events()
                FireboyAndWatergirlEnv.plt.pause(0.001)
            except Exception as e:
                if not ("closed" in str(e).lower() or "destroy" in str(e).lower()):
                    print(f"Warning: {e}")
            return rgb_image
        elif mode == "rgb_array":
            return self._get_state()

    def close(self):
        pass

    def _get_state(self, draw=False):
        level_data = self.board.get_level_data()
        # Ignore the outer border (typically 1 tile) since it's always constant
        border_size = 1  # Assuming outer wall is 1 tile thick
        # Initialize empty RGB grid for the inner area only
        inner_height = len(level_data) - (2 * border_size)
        inner_width = len(level_data[0]) - (2 * border_size)
        rgb_image = np.zeros((inner_height, inner_width, 3), dtype=np.uint8)
        # Fill the RGB image based on the inner level data
        color_mapping = {
            ' ': [255, 255, 255],
            'S': [50, 50, 50],
            'L': [255, 50, 0],
            'W': [0, 100, 255],
            'G': [50, 200, 50],
            'f': [255, 0, 0],
            'w': [0, 0, 255],
            'A': [200, 100, 0],
            'B': [0, 150, 200],
            'a': [255, 200, 0],
            'b': [0, 200, 255],
            'P': [150, 150, 150],
            'D': [200, 200, 100],
        }

        # Vectorized tile drawing
        tile_array = np.array([[level_data[y][x] for x in range(border_size, len(level_data[0]) - border_size)]
                               for y in range(border_size, len(level_data) - border_size)])
        for tile_char, color in color_mapping.items():
            mask = tile_array == tile_char
            rgb_image[mask] = color

        # Remove static player tiles (they are drawn dynamically)
        for tile_char in ['f', 'w']:
            mask = tile_array == tile_char
            rgb_image[mask] = color_mapping[' ']

        # Draw stars (vectorized)
        for star in self.stars:
            s_x, s_y = np.array(star.get_position()) // 16 - border_size
            if 0 <= s_y < rgb_image.shape[0] and 0 <= s_x < rgb_image.shape[1]:
                if not star.is_collected:
                    rgb_image[s_y, s_x] = color_mapping['a' if star._player ==
                                                        "fire" else 'b']
                else:
                    rgb_image[s_y, s_x] = color_mapping[' ']

        # Draw dynamic player positions
        if self.fire_boy:
            fb_x, fb_y = np.array(self.fire_boy.get_position()) // 16
            fb_x = int(fb_x)
            fb_y = int(fb_y)
            fb_x -= 1
            if 0 <= fb_y < rgb_image.shape[0] and 0 <= fb_x < rgb_image.shape[1]:
                rgb_image[fb_y, fb_x] = color_mapping['f']
        if self.water_girl:
            wg_x, wg_y = np.array(self.water_girl.get_position()) // 16
            wg_x = int(wg_x)
            wg_y = int(wg_y)
            wg_x -= 1
            if 0 <= wg_y < rgb_image.shape[0] and 0 <= wg_x < rgb_image.shape[1]:
                rgb_image[wg_y, wg_x] = color_mapping['w']
            else:
                if self.game.index == 1:
                    print(
                        f"watergirl position out of bounds: ({fb_x}, {fb_y})")

        # Update the observation history only if the state has changed

        # Save image if requested
        if draw:
            plt.figure(figsize=(15, 5))
            plt.imshow(rgb_image)
            plt.axis('off')
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            plt.savefig(f"observation.png", bbox_inches='tight', pad_inches=0)
            plt.close()

        return rgb_image

    def _apply_action(self, action):
        """
        Update the game state based on the discrete action.
        """
        # Decode the single discrete action into Fireboy and Watergirl actions

        # Check if action is an array-like (for MultiDiscrete) or a single int
        if isinstance(action, (list, tuple, np.ndarray)) and len(action) == 2:
            # print(action)
            fireboy_action = 3
            watergirl_action = 3

            # Map the action to Fireboy and Watergirl actions
            # if action <= 3:
            fireboy_action = action[0]
            # elif action >= 4:
            watergirl_action = action[1]

            # Map Fireboy's action

            # print(fireboy_action)
            if fireboy_action == 0:
                self.fire_boy.moving_left = False
                self.fire_boy.moving_right = False
                self.fire_boy.jumping = False
            elif fireboy_action == 1:
                self.fire_boy.moving_left = True
                self.fire_boy.moving_right = False
                self.fire_boy.jumping = False
            elif fireboy_action == 2:
                self.fire_boy.moving_left = False
                self.fire_boy.moving_right = True
                self.fire_boy.jumping = False
            elif fireboy_action == 3:
                self.fire_boy.moving_left = False
                self.fire_boy.moving_right = False
                self.fire_boy.jumping = True

            if watergirl_action == 0:
                self.water_girl.moving_left = False
                self.water_girl.moving_right = False
                self.water_girl.jumping = False
            elif watergirl_action == 1:
                self.water_girl.moving_left = True
                self.water_girl.moving_right = False
                self.water_girl.jumping = False
            elif watergirl_action == 2:
                self.water_girl.moving_left = False
                self.water_girl.moving_right = True
                self.water_girl.jumping = False
            elif watergirl_action == 3:
                self.water_girl.moving_left = False
                self.water_girl.moving_right = False
                self.water_girl.jumping = True

        self.game.move_player(self.board, self.gates, [
                              self.fire_boy, self.water_girl])
        self.game.check_for_gate_press(
            self.gates, [self.fire_boy, self.water_girl])
        self.game.check_for_star_collected(
            self.stars, [self.fire_boy, self.water_girl])

    def _compute_reward(self):
        reward = 0  # -0.001  # Time penalty

        # Vectorized reward for stars
        stars = np.array(self.stars)
        is_collected = np.array([star.is_collected for star in stars])
        reward_given = np.array([star.reward_given for star in stars])
        for i, star in enumerate(stars):
            if is_collected[i] and not reward_given[i]:
                reward += 10
                star.reward_given = True

        if self._check_done():
            reward *= 10

        return reward

    def _check_done(self):
        # Vectorized check if all stars are collected
        return all(star.is_collected for star in self.stars)


register(
    id="FireboyAndWatergirl-ppo-v4",  # Unique ID for the environment
    # Path to the environment class
    entry_point="cleanrl.fireboy_and_watergirl_ppo_v4:FireboyAndWatergirlEnv",
)
