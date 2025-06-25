import random
from typing import Set
import gymnasium as gym
from gymnasium import spaces
from matplotlib import pyplot as plt
import numpy as np
from gymnasium.envs.registration import register
import cv2  # Add this import at the top with other imports
import os

from fireboy_and_watergirl.board import Board
from fireboy_and_watergirl.character import FireBoy, WaterGirl
from fireboy_and_watergirl.doors import FireDoor, WaterDoor
from fireboy_and_watergirl.game import Game
from fireboy_and_watergirl.gate import Gate
from fireboy_and_watergirl.star import Star

# v9 avoid obstacles


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
        self.level = 'level_obstacles'
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

        self.games = 0

        self._load_level()

        self.steps = 0
        self.max_steps = 128 * 2  # 400
        self.envs = 8

        self.level_height = 25 - 2  # Assuming 1-tile border on top and bottom
        self.level_width = 34 - 2   # Assuming 1-tile border on left and right
        self.num_channels = 3       # RGB channels

        self.fb_visited_positions = set()  # Add this line to store visited positions
        self.wg_visited_positions = set()  # Add this line to store visited positions

        # Define the flattened observation space for 3 stacked frames
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.level_height, self.level_width, self.num_channels),
            dtype=np.uint8
        )

        self.cumulative_rewards = np.zeros(self.envs)
        self.video_scale = 16
        self.video_folder = "episode_videos"
        if not os.path.exists(self.video_folder):
            os.makedirs(self.video_folder)

        self.record_video = True  # Set to True to record videos
        self.video_writer = None

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
        # Read the level data from the file
        with open('./fireboy_and_watergirl/data/'+self.level+'.txt', 'r') as file:
            level_data = [line.strip().split(',') for line in file.readlines()]

        # Initialize game components
        self.board = Board('./fireboy_and_watergirl/data/'+self.level+'.txt')
        self.gates: list[Gate] = []
        self.doors: list[FireDoor | WaterDoor] = []
        self.stars: list[Star] = []
        self.fire_boy: FireBoy = None
        self.water_girl: WaterGirl = None

        # fire_boy_location = (200, 336)
        # self.fire_boy = FireBoy(fire_boy_location)
        # water_girl_location = (200, 336)
        # self.water_girl = WaterGirl(water_girl_location)

        # fire_door_location = (64, 48)
        # fire_door = FireDoor(fire_door_location)
        # water_door_location = (128, 48)
        # water_door = WaterDoor(water_door_location)
        # self.doors = [fire_door, water_door]

        # Parse the level data to dynamically set up components
        for y, row in enumerate(level_data):
            for x, tile in enumerate(row):
                # Assuming 16x16 tiles
                if tile == 'f':  # Fireboy starting position
                    self.fire_boy = FireBoy((x * 16, y * 16))
                elif tile == 'w':  # Watergirl starting position
                    self.water_girl = WaterGirl((x * 16, y * 16))
                elif tile == 'A':  # Fire door
                    self.doors.append(FireDoor((x * 16, y * 16)))
                elif tile == 'B':  # Water door
                    self.doors.append(WaterDoor((x * 16, y * 16)))
                # elif tile == 'D':  # Gate
                    # Add a generic gate (you can customize this further)
                    # self.gates.append(Gates((x * 16, y * 16), []))
                # elif tile == 'P':  # Plate A
                    # Add a plate that controls a gate
                    # self.gates.append(
                    #     Plate((x * 16, y * 16), [(x * 16, y * 16)]))
                elif tile == 'A':  # Plate B
                    self.gates.append(
                        FireDoor((x * 16, y * 16), [(x * 16, y * 16)]))
                elif tile == 'B':  # Plate B
                    self.gates.append(
                        WaterDoor((x * 16, y * 16), [(x * 16, y * 16)]))
                elif tile == 'a':  # Gate A
                    self.stars.append(Star([x * 16, y * 16], "fire"))
                elif tile == 'b':  # Gate B
                    self.stars.append(Star([x * 16, y * 16], "water"))

        # Add more cases as needed for other tiles

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.fb_visited_positions.clear()  # Clear visited positions on reset
        self.wg_visited_positions.clear()  # Clear visited positions on reset

        # Get the current env index and its cumulative reward
        current_env = self.game.index
        self.cumulative_rewards[current_env] = 0

        # Start a new video for the new episode (temporary name)
        if self.record_video and self.games % 10 == 0:
            if self.video_writer is not None:
                self.video_writer.release()
            video_path = os.path.join(
                self.video_folder,
                f"Temp_{self.games}_{current_env}.mp4"
            )
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                video_path, fourcc, 30.0,
                (self.level_width * self.video_scale,
                 self.level_height * self.video_scale)
            )

        self._load_level()
        self.state = self._get_state()
        self.done = False
        self.steps = 0

        self.games += 1

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

        # Update cumulative reward for current environment
        current_env = self.game.index
        self.cumulative_rewards[current_env] += reward

        # Check if the game is done
        self.done = self._check_done()

        self.steps += 1

        # Add visit counts to info dict
        info = {
            "unique_positions": -1,
            "stars_collected": sum(star.is_collected for star in self.stars),
            "finished": 1 if self.done else 0,
            "zero_reward": 1 if reward == 0 else 0,
        }

        if self.record_video and self.games % 10 == 0:
            # Capture frame for video
            if self.video_writer is not None:
                frame = cv2.cvtColor(self.state, cv2.COLOR_RGB2BGR)
                frame = cv2.resize(frame, (self.level_width * self.video_scale, self.level_height * self.video_scale),
                                   interpolation=cv2.INTER_NEAREST)
                self.video_writer.write(frame)

        # Save and close video at the end of the episode with correct reward
        if self.done:
            if self.record_video and self.games % 10 == 0:
                if self.video_writer is not None:
                    final_reward = self.cumulative_rewards[current_env]
                    temp_path = os.path.join(
                        self.video_folder,
                        f"Temp_{self.games}_{current_env}.mp4"
                    )
                    video_path = os.path.join(
                        self.video_folder,
                        f"GamePlay{self.games}_{self.game.index}_{final_reward:.2f}.mp4"
                    )
                    self.video_writer.release()
                    self.video_writer = None
                    if os.path.exists(temp_path):
                        os.rename(temp_path, video_path)
                    else:
                        print(
                            f"Warning: Temp video file not found: {temp_path}")

        if self.steps >= self.max_steps:
            self.done = True

        return self.state, reward, self.done, False, info

    def render(self, mode="human"):
        """
        Render the environment to the screen or other output.
        Shows only the RGB observation used by the agent.
        """
        if mode == "human":
            rgb_image = self._get_state()
            if not hasattr(FireboyAndWatergirlEnv, 'plt_initialized'):
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
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

    def _get_state(self):
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
            'a': [255, 200, 0],
            'b': [0, 200, 255],
            'A': [200, 100, 0],
            'B': [0, 150, 200],
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
            fb_x = int(fb_x-1)
            fb_y = int(fb_y)
            if 0 <= fb_y < rgb_image.shape[0] and 0 <= fb_x < rgb_image.shape[1]:
                rgb_image[fb_y, fb_x] = color_mapping['f']
            else:
                print(
                    f"Fireboy position out of bounds: ({fb_x}, {fb_y})")

        if self.water_girl:
            wg_x, wg_y = np.array(self.water_girl.get_position()) // 16
            wg_x = int(wg_x-1)
            wg_y = int(wg_y)
            if 0 <= wg_y < rgb_image.shape[0] and 0 <= wg_x <= rgb_image.shape[1]:
                rgb_image[wg_y, wg_x] = color_mapping['w']
            else:
                print(
                    f"watergirl position out of bounds: ({wg_x}, {wg_y})")

        return rgb_image

    def _apply_action(self, action):
        """
        Update the game state based on the discrete action.
        """

        # Check if action is an array-like (for MultiDiscrete) or a single int
        if isinstance(action, (list, tuple, np.ndarray)) and len(action) == 2:
            fireboy_action = 3
            watergirl_action = 3

            # Map the action to Fireboy and Watergirl actions
            fireboy_action = action[0]
            watergirl_action = action[1]

            fireboy_action = random.randint(0, 3)
            watergirl_action = random.randint(0, 3)

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
        else:
            print('Invalid action format. Expected a list or tuple of length 2.')
            print(action)

        fb_old = self.fire_boy.get_position()

        self.game.move_player(self.board, self.gates, [
                              self.fire_boy, self.water_girl])
        self.game.check_for_plates_press(
            self.gates, [self.fire_boy, self.water_girl])
        self.game.check_for_star_collected(
            self.stars, [self.fire_boy, self.water_girl])

    def _compute_reward(self):
        reward = 0

        # Reward for moving right
        fb_x, fb_y = np.array(self.fire_boy.get_position()) // 16
        fb_x = int(fb_x-1)
        wg_x, wg_y = np.array(self.water_girl.get_position()) // 16
        wg_x = int(wg_x-1)

        # Store previous x positions if not exists
        if not hasattr(self, '_prev_fb_x'):
            self._prev_fb_x = fb_x
            self._prev_wg_x = wg_x

        # Reward for moving right (positive x direction)
        fb_movement = fb_x - self._prev_fb_x
        wg_movement = wg_x - self._prev_wg_x
        reward += fb_movement * 5  # 5 points per tile moved right
        reward += wg_movement * 5

        # Update previous positions
        self._prev_fb_x = fb_x
        self._prev_wg_x = wg_x

        # Punish for hazards
        level_data = self.board.get_level_data()
        fb_tile = level_data[int(fb_y)][int(fb_x)]
        wg_tile = level_data[int(wg_y)][int(wg_x)]

        # Punish Fireboy for water and goo
        if fb_tile in ['W', 'G']:  # W for water, G for goo
            reward -= 10
            # self.done = True  # End episode on death

        # Punish Watergirl for fire and goo
        if wg_tile in ['F', 'G']:  # F for fire, G for goo
            reward -= 10
            # self.done = True  # End episode on death

        if fb_x >= self.level_width - 1 and wg_x >= self.level_width - 1:
            reward += 200  # Large bonus for reaching the end
            self.done = True

        # Keep star collection rewards
        for star in self.stars:
            if star.is_collected and not star.reward_given:
                # reward += 50
                star.reward_given = True

        # # Completion reward
        # if self._check_done():
        #     reward += 100

        return reward

    def _check_done(self):
        # Vectorized check if all stars are collected
        return all(star.is_collected for star in self.stars)


register(
    id="FireboyAndWatergirl-ppo-random-baseline",
    entry_point="cleanrl.v9.fireboy_and_watergirl_ppo_random_baseline:FireboyAndWatergirlEnv",
)
