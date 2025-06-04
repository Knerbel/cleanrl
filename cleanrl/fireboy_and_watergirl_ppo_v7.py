import random
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gymnasium.envs.registration import register
import cv2  # Add this import at the top with other imports
import os

from fireboy_and_watergirl.board import Board
from fireboy_and_watergirl.character import FireBoy, WaterGirl
from fireboy_and_watergirl.doors import FireDoor, WaterDoor
from fireboy_and_watergirl.game import Game
from fireboy_and_watergirl.gates import Gates
from fireboy_and_watergirl.stars import Stars

# v7 uses snake learning, but also curriculum learning


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
        self.level = 'level1_empty'
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

        self.MAX_DISTANCE = 2
        self._load_level()

        self.steps = 0
        self.max_steps = self.MAX_DISTANCE * 10
        self.envs = 8

        self.level_height = 25 - 2  # Assuming 1-tile border on top and bottom
        self.level_width = 34 - 2   # Assuming 1-tile border on left and right
        self.num_channels = 3       # RGB channels

        self.observation_space = spaces.Box(
            low=0,
            high=1,
            # walls, fb, wg, stars
            shape=(self.level_height, self.level_width, 4),
            dtype=np.uint8
        )

        self.cumulative_rewards = np.zeros(self.envs)
        self.video_scale = 16
        self.video_folder = "episode_videos"
        if not os.path.exists(self.video_folder):
            os.makedirs(self.video_folder)

        self.record_video = True  # Set to True to record videos
        self.video_writer = None

        self._previous_distances = {}

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

        with open('./fireboy_and_watergirl/data/'+self.level+'.txt', 'r') as file:
            level_data = [line.strip().split(',') for line in file.readlines()]

        # Initialize game components
        self.board = Board('./fireboy_and_watergirl/data/'+self.level+'.txt')
        self.gates: list[Gates] = []
        self.doors: list[FireDoor | WaterDoor] = []
        self.stars: list[Stars] = []
        self.fire_boy: FireBoy = None
        self.water_girl: WaterGirl = None

        # First pass: Find all valid floor positions
        valid_positions = []
        # for y in range((len(level_data)//2) - 2, -1, -1):
        y = 21
        for x in range(len(level_data[0])):
            if (level_data[y][x] == ' ' and
                y < len(level_data) - 1
                # and
                    # level_data[y + 1][x] in ['S', 'G']
                ):
                valid_positions.append((x, y))
        y = 15
        for x in range(len(level_data[0])):
            if (level_data[y][x] == ' ' and
                y < len(level_data) - 1
                # and
                    # level_data[y + 1][x] in ['S', 'G']
                ):
                valid_positions.append((x, y))

        def manhattan_distance(pos1, pos2):
            return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

        MIN_DISTANCE = 1  # Minimum distance between agent and its star
        MAX_DISTANCE = self.MAX_DISTANCE  # Maximum distance between agent and its star

        # Place characters and stars with minimum distance
        remaining_positions = valid_positions.copy()
        selected_positions = []

        # Helper function to find valid star position for an agent
        def find_valid_star_position(agent_pos, remaining_pos):
            valid_star_positions = [
                pos for pos in remaining_pos
                if MIN_DISTANCE <= manhattan_distance(pos, agent_pos) <= MAX_DISTANCE
                and pos not in selected_positions
            ]
            return random.choice(valid_star_positions) if valid_star_positions else None

        # Place Fireboy
        if remaining_positions:
            fb_pos = random.choice(remaining_positions)
            selected_positions.append(fb_pos)
            remaining_positions.remove(fb_pos)

            # Find and place Fire Star within distance constraints
            fire_star_pos = find_valid_star_position(
                fb_pos, remaining_positions)
            if fire_star_pos:
                selected_positions.append(fire_star_pos)
                remaining_positions.remove(fire_star_pos)

        # Place Watergirl
        if remaining_positions:
            wg_pos = random.choice([pos for pos in remaining_positions
                                    if pos not in selected_positions])
            selected_positions.append(wg_pos)
            remaining_positions.remove(wg_pos)

            # Find and place Water Star within distance constraints
            water_star_pos = find_valid_star_position(
                wg_pos, remaining_positions)
            if water_star_pos:
                selected_positions.append(water_star_pos)
                remaining_positions.remove(water_star_pos)

        # Create entities only if we have valid positions
        if len(selected_positions) >= 4:  # We need all 4 positions
            self.fire_boy = FireBoy(
                (selected_positions[0][0] * 16, selected_positions[0][1] * 16))
            self.fire_boy_star = Stars(
                [selected_positions[1][0] * 16, selected_positions[1][1] * 16], "fire")
            self.water_girl = WaterGirl(
                (selected_positions[2][0] * 16, selected_positions[2][1] * 16))
            self.water_girl_star = Stars(
                [selected_positions[3][0] * 16, selected_positions[3][1] * 16], "water")

            self.stars = [self.fire_boy_star, self.water_girl_star]
        else:
            print("Warning: Could not place all entities with required constraints")
            self.reset()  # Try again

            # Create entities only if we have enough positions
            if len(selected_positions) >= 2:
                self.fire_boy = FireBoy(
                    (selected_positions[0][0] * 16, selected_positions[0][1] * 16))
                self.water_girl = WaterGirl(
                    (selected_positions[1][0] * 16, selected_positions[1][1] * 16))

                if len(selected_positions) > 2:
                    self.stars.append(
                        Stars([selected_positions[2][0] * 16, selected_positions[2][1] * 16], "fire"))
                if len(selected_positions) > 3:
                    self.stars.append(
                        Stars([selected_positions[3][0] * 16, selected_positions[3][1] * 16], "water"))

        # Keep doors and gates from the original level data
        for y, row in enumerate(level_data):
            for x, tile in enumerate(row):
                pos = (x * 16, y * 16)
                if tile == 'A':
                    self.doors.append(FireDoor(pos))
                elif tile == 'B':
                    self.doors.append(WaterDoor(pos))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._previous_distances = {}  # Reset distances
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
        # Initialize 4-channel state
        state = np.zeros(
            (self.level_height, self.level_width, 4), dtype=np.uint8)

        # Channel 0: Walls and obstacles
        level_data = self.board.get_level_data()
        for y in range(1, len(level_data)-1):
            for x in range(1, len(level_data[0])-1):
                if level_data[y][x] in ['S', 'G']:
                    state[y-1, x-1, 0] = 1

        # Channel 1: Fireboy
        if self.fire_boy:
            fb_x, fb_y = np.array(self.fire_boy.get_position()) // 16
            fb_x, fb_y = int(fb_x-1), int(fb_y-1)
            if 0 <= fb_y < state.shape[0] and 0 <= fb_x < state.shape[1]:
                state[fb_y, fb_x, 1] = 1

        # Channel 2: Watergirl
        if self.water_girl:
            wg_x, wg_y = np.array(self.water_girl.get_position()) // 16
            wg_x, wg_y = int(wg_x-1), int(wg_y-1)
            if 0 <= wg_y < state.shape[0] and 0 <= wg_x < state.shape[1]:
                state[wg_y, wg_x, 2] = 1

        # Channel 3: Stars
        for star in self.stars:
            if not star.is_collected:
                s_x, s_y = np.array(star.get_position()) // 16 - 1
                if 0 <= s_y < state.shape[0] and 0 <= s_x < state.shape[1]:
                    state[s_y, s_x, 3] = 1

        return state

    def _apply_action(self, action):
        """
        Update the game state based on the discrete action.
        """
        # Decode the single discrete action into Fireboy and Watergirl actions

        # Check if action is an array-like (for MultiDiscrete) or a single int
        if isinstance(action, (list, tuple, np.ndarray)) and len(action) == 2:
            fireboy_action = 3
            watergirl_action = 3

            # Map the action to Fireboy and Watergirl actions
            fireboy_action = action[0]
            watergirl_action = action[1]

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

        self.game.move_player(self.board, self.gates, [
                              self.fire_boy, self.water_girl])
        self.game.check_for_gate_press(
            self.gates, [self.fire_boy, self.water_girl])
        self.game.check_for_star_collected(
            self.stars, [self.fire_boy, self.water_girl])

    def _compute_reward(self):
        reward = 0

        # Distance-based shaping reward
        for star in self.stars:
            if not star.is_collected:
                player = self.fire_boy if star._player == "fire" else self.water_girl
                player_pos = np.array(player.get_position()) // 16
                star_pos = np.array(star.get_position()) // 16

                curr_dist = np.sqrt(np.sum((player_pos - star_pos)**2))
                prev_dist = self._previous_distances.get(id(star), curr_dist)

                # Reward for getting closer to star
                reward += 0.1 * (prev_dist - curr_dist)
                self._previous_distances[id(star)] = curr_dist

        # Star collection reward
        for star in self.stars:
            if star.is_collected and not star.reward_given:
                reward += 50  # Increased immediate reward
                star.reward_given = True

        # Completion reward
        if self._check_done():
            reward += 100  # Fixed bonus for completing

        return reward

    def _check_done(self):
        # Vectorized check if all stars are collected
        return all(star.is_collected for star in self.stars)


# Environment with exploration reward
register(
    id="FireboyAndWatergirl-ppo-v7",  # Unique ID for the environment
    # Path to the environment class
    entry_point="cleanrl.fireboy_and_watergirl_ppo_v7:FireboyAndWatergirlEnv",
)
