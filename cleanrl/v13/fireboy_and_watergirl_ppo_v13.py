import gymnasium as gym
import random
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
from fireboy_and_watergirl.plate import Plate
from fireboy_and_watergirl.star import Star

# v11 level 1 # plates_and_gates


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
        self.level = 'level4c_plates_and_gates'

        self.game = Game()  # Instantiate the Game class
        self.board = None
        self.fire_boy: FireBoy
        self.water_girl: WaterGirl
        self.gates = None
        self.doors = None

        # Initialize game state
        self.state = None
        self.done = False

        self.games = 0

        self._load_level()

        self.steps = 0
        self.max_steps = 128 * 4  # 400
        self.envs = 8

        self.level_height = 25 - 2  # Assuming 1-tile border on top and bottom
        self.level_width = 34 - 2   # Assuming 1-tile border on left and right
        self.num_channels = 3       # RGB channels

        self.fb_visited_positions = set()  # Add this line to store visited positions
        self.wg_visited_positions = set()  # Add this line to store visited positions

        self.times_in_water = 0
        self.times_in_fire = 0
        self.times_in_goo = 0

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

        self.image_folder = "episode_images"
        if not os.path.exists(self.image_folder):
            os.makedirs(self.image_folder)

        self.record_video = True  # Set to True to record videos
        self.video_writer = None
        self.video_recording_every = 1

        self.fb_plate_steps = 0
        self.wg_plate_steps = 0

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
        # with open('./fireboy_and_watergirl/data/'+self.level+'.txt', 'r') as file:
        #     level_data = [line.strip().split(',') for line in file.readlines()]

        # Initialize game components
        self.board = Board('./fireboy_and_watergirl/data/'+self.level+'.txt')
        self.gates: list[Gate] = []
        self.plates: list[Plate] = []
        self.doors: list[FireDoor | WaterDoor] = []
        self.stars: list[Star] = []

        # Collect valid spawn positions
        valid_y = {2, 6, 10, 14, 18}
        forbidden_x = {16, 17, 18, 19}
        min_forbidden_x = min(forbidden_x)
        max_forbidden_x = max(forbidden_x)

        # Collect valid left/right x for each y
        left_x_by_y = {y: [] for y in valid_y}
        right_x_by_y = {y: [] for y in valid_y}

        # Parse the level data to dynamically set up components
        for y, row in enumerate(self.board.get_level_data()):
            for x, tile in enumerate(row):

                # Only consider empty tiles at allowed y and x
                if y in valid_y and tile == ' ':
                    if x < min_forbidden_x:
                        left_x_by_y[y].append(x)
                    elif x > max_forbidden_x:
                        right_x_by_y[y].append(x)
                # Assuming 16x16 tiles
                if tile == 'f':  # Fireboy starting position
                    self.fire_boy = FireBoy((x * 16, y * 16))
                elif tile == 'w':  # Watergirl starting position
                    self.water_girl = WaterGirl((x * 16, y * 16))
                elif tile == 'A':  # Fire door
                    self.doors.append(FireDoor((x * 16, y * 16)))
                elif tile == 'B':  # Water door
                    self.doors.append(WaterDoor((x * 16, y * 16)))

                elif tile == 'P':  # Fireboy Plate
                    self.plates.append(Plate((x * 16, y * 16), "fire"))
                elif tile == 'O':  # Watergirl Plate
                    self.plates.append(Plate((x * 16, y * 16), "water"))

                elif tile == 'D':  # Fireboy Gate
                    self.gates.append(Gate((x * 16, y * 16), "fire"))
                elif tile == 'E':  # Watergirl Gate
                    self.gates.append(Gate((x * 16, y * 16), "water"))

                elif tile == 'a':  # Gate A
                    self.stars.append(Star([x * 16, y * 16], "fire"))
                elif tile == 'b':  # Gate B
                    self.stars.append(Star([x * 16, y * 16], "water"))

        # Randomly select two different valid positions for Fireboy and Watergirl
        possible_y = [y for y in valid_y if left_x_by_y[y] and right_x_by_y[y]]
        chosen_y = random.choice(possible_y)
        fireboy_x = random.choice(left_x_by_y[chosen_y])
        watergirl_x = random.choice(right_x_by_y[chosen_y])

        # self.fire_boy = FireBoy((fireboy_x * 16, chosen_y * 16))
        # self.water_girl = WaterGirl((watergirl_x * 16, chosen_y * 16))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.fb_visited_positions.clear()  # Clear visited positions on reset
        self.wg_visited_positions.clear()  # Clear visited positions on reset

        # Get the current env index and its cumulative reward
        current_env = self.game.index
        self.cumulative_rewards[current_env] = 0

        # Start a new video for the new episode (temporary name)
        if self.record_video and self.games % self.video_recording_every == 0:
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

        self.times_in_water = 0
        self.times_in_fire = 0
        self.times_in_goo = 0

        self.games += 1

        self.fb_plate_steps = 0
        self.wg_plate_steps = 0

        return self.state, {}

    def step(self, action):
        """
        Apply the given action to the environment and return the results.
        """
        # Apply the action to the characters
        self._apply_action(action)
        # Update the game state
        self.state = self._get_state()

        doors_d_open = any(
            plate._is_pressed for plate in self.plates if plate._type == "fire")
        doors_e_open = any(
            plate._is_pressed for plate in self.plates if plate._type == "water")

        self.board.update_doors_solid_state(doors_d_open, doors_e_open)

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
            "unique_positions": len(self.fb_visited_positions) + len(self.wg_visited_positions),
            "stars_collected": sum(star.is_collected for star in self.stars),
            "finished": 1 if self.done else 0,
            "players_at_door": sum(door.player_at_door for door in self.doors),
            "zero_reward": 1 if reward <= 0 else 0,
            "times_in_water": self.times_in_water,
            "times_in_fire": self.times_in_fire,
            "times_in_goo": self.times_in_goo,
        }

        if self.record_video and self.games % self.video_recording_every == 0:
            # Capture frame for video
            if self.video_writer is not None:
                frame = cv2.cvtColor(self.state, cv2.COLOR_RGB2BGR)
                frame = cv2.resize(frame, (self.level_width * self.video_scale, self.level_height * self.video_scale),
                                   interpolation=cv2.INTER_NEAREST)
                self.video_writer.write(frame)

        if self.steps >= self.max_steps:
            self.done = True

        if self.done:
            # Save the last frame as an image
            image_path = os.path.join(
                self.image_folder,
                f"episode_{self.games-1}_{current_env}_final.png"
            )
            frame_bgr = cv2.cvtColor(self.state, cv2.COLOR_RGB2BGR)
            frame_bgr = cv2.resize(
                frame_bgr,
                (self.level_width * self.video_scale,
                 self.level_height * self.video_scale),
                interpolation=cv2.INTER_NEAREST
            )
            cv2.imwrite(image_path, frame_bgr)

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
        border_size = 1  # Assuming outer wall is 1 tile thick
        inner_height = len(level_data) - (2 * border_size)
        inner_width = len(level_data[0]) - (2 * border_size)
        rgb_image = np.zeros((inner_height, inner_width, 3), dtype=np.uint8)
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
            'O': [150, 150, 150],
            'D': [200, 200, 100],
            'E': [200, 200, 100],
            'D_open': [180, 140, 20],
        }
        light_blue = [173, 216, 230]
        light_red = [255, 182, 193]

        tile_array = np.array([[level_data[y][x] for x in range(border_size, len(level_data[0]) - border_size)]
                               for y in range(border_size, len(level_data) - border_size)])
        for tile_char, color in color_mapping.items():
            mask = tile_array == tile_char
            rgb_image[mask] = color

        # Remove static player tiles (they are drawn dynamically)
        for tile_char in ['f', 'w']:
            mask = tile_array == tile_char
            rgb_image[mask] = color_mapping[' ']

        # VISUALIZE VISITED TILES AS MAGENTA
        # for (fb_x, fb_y) in self.fb_visited_positions:
        #     if 0 <= fb_y < rgb_image.shape[0] and 0 <= fb_x < rgb_image.shape[1]:
        #         rgb_image[fb_y, fb_x] = light_red
        # for (wg_x, wg_y) in self.wg_visited_positions:
        #     if 0 <= wg_y < rgb_image.shape[0] and 0 <= wg_x < rgb_image.shape[1]:
        #         rgb_image[wg_y, wg_x] = light_blue

        # Draw stars (vectorized)
        for star in self.stars:
            s_x, s_y = np.array(star.get_position()) // 16 - border_size
            if 0 <= s_y < rgb_image.shape[0] and 0 <= s_x < rgb_image.shape[1]:
                if not star.is_collected:
                    rgb_image[s_y, s_x] = color_mapping['a' if star._player ==
                                                        "fire" else 'b']
                else:
                    rgb_image[s_y, s_x] = color_mapping[' ']

        # Draw gates ('D') as yellow only if no plate is pressed, else as empty
        for y in range(rgb_image.shape[0]):
            for x in range(rgb_image.shape[1]):
                tile = tile_array[y, x]
                if tile == 'D':
                    if any(plate._is_pressed for plate in self.plates if plate._type == "fire"):
                        rgb_image[y, x] = color_mapping[' ']
                    else:
                        rgb_image[y, x] = color_mapping['D']
                if tile == 'E':
                    if any(plate._is_pressed for plate in self.plates if plate._type == "water"):
                        rgb_image[y, x] = color_mapping[' ']
                    else:
                        rgb_image[y, x] = color_mapping['D']

        # Draw End doors
        for door in self.doors:
            d_x, d_y = np.array(door.get_position()) // 16 - border_size
            if 0 <= d_y < rgb_image.shape[0] and 0 <= d_x < rgb_image.shape[1]:
                if not door.reward_given:
                    rgb_image[d_y][d_x] = color_mapping['A' if door._player ==
                                                        "fire" else 'B']
                else:
                    rgb_image[d_y][d_x] = color_mapping[' ']

        # Draw dynamic player positions
        if self.fire_boy:
            fb_x, fb_y = np.array(self.fire_boy.get_position()) // 16
            fb_x = int(fb_x-1)
            fb_y = int(fb_y-1)
            if 0 <= fb_y < rgb_image.shape[0] and 0 <= fb_x < rgb_image.shape[1]:
                rgb_image[fb_y, fb_x] = color_mapping['f']
            else:
                print(f"Fireboy position out of bounds: ({fb_x}, {fb_y})")

        if self.water_girl:
            wg_x, wg_y = np.array(self.water_girl.get_position()) // 16
            wg_x = int(wg_x-1)
            wg_y = int(wg_y-1)
            if 0 <= wg_y < rgb_image.shape[0] and 0 <= wg_x < rgb_image.shape[1]:
                rgb_image[wg_y, wg_x] = color_mapping['w']
            else:
                print(f"Watergirl position out of bounds: ({wg_x}, {wg_y})")

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
        self.game.check_for_plates_press(
            self.plates, [self.fire_boy, self.water_girl])
        self.game.check_for_star_collected(
            self.stars, [self.fire_boy, self.water_girl])
        self.game.check_for_at_door(
            self.doors, [self.fire_boy, self.water_girl]
        )

    def _compute_reward(self):
        reward = -5  # Small negative reward for each step
        fb_reward = -0.05
        wg_reward = -0.05

        level_data = self.board.get_level_data()

        # Track previous number of visited positions
        prev_fb_positions = len(self.fb_visited_positions)
        prev_wg_positions = len(self.wg_visited_positions)

        fb_x, fb_y = np.array(self.fire_boy.get_position()) // 16
        fb_x, fb_y = int(fb_x-1), int(fb_y-1)
        wg_x, wg_y = np.array(self.water_girl.get_position()) // 16
        wg_x, wg_y = int(wg_x-1), int(wg_y-1)

        # Update visited positions (this happens in step())
        if self.fire_boy:
            if 0 <= fb_y < self.level_height and 0 <= fb_x < self.level_width:
                tile_fb = level_data[fb_y+1][fb_x+1]  # +1 for border
                if tile_fb == 'W':  # Water or Goo
                    reward -= 7.5
                    self.times_in_water += 1
                elif tile_fb == 'G':
                    reward -= 7.5
                    self.times_in_goo += 1
                else:
                    self.fb_visited_positions.add((fb_x, fb_y))

        if self.water_girl:
            if 0 <= wg_y < self.level_height and 0 <= wg_x < self.level_width:
                tile_wg = level_data[wg_y+1][wg_x+1]  # +1 for border
                if tile_wg == 'L':  # Fire or Goo
                    reward -= 7.5
                    self.times_in_fire += 1
                elif tile_wg == 'G':
                    reward -= 7.5
                    self.times_in_goo += 1
                else:
                    self.wg_visited_positions.add((wg_x, wg_y))

        # Calculate exploration reward
        new_fb_positions = len(self.fb_visited_positions) - prev_fb_positions
        new_wg_positions = len(self.wg_visited_positions) - prev_wg_positions
        exploration_reward = (new_fb_positions + new_wg_positions)
        # reward += exploration_reward * 10
        # fb_reward = new_fb_positions*0.1
        # wg_reward = new_wg_positions*0.1

        # Rewards for passing through doors
        # Reward for stepping on a 'D' (gate) tile
        # if self.fire_boy:
        #     if 0 <= fb_y < self.level_height and 0 <= fb_x < self.level_width:
        #         if level_data[fb_y+1][fb_x+1] == 'D' or level_data[fb_y+1][fb_x+1] == 'E':
        #             reward += 200
        #             fb_reward += 200
        # if self.water_girl:
        #     if 0 <= wg_y < self.level_height and 0 <= wg_x < self.level_width:
        #         if level_data[wg_y+1][wg_x+1] == 'D' or level_data[wg_y+1][wg_x+1] == 'E':
        #             reward += 200
        #             wg_reward += 200

        # Reward for stepping on a plate

        for door in self.plates:
            plate_x, plate_y = door.get_position()
            plate_x = int(plate_x // 16)
            plate_y = int(plate_y // 16)
            for player in [self.fire_boy, self.water_girl]:
                player_x, player_y = player.get_position()
                player_x = int(player_x // 16)
                player_y = int(player_y // 16)
                if player_x == plate_x and player_y == plate_y - 1:
                    reward += 0 * door.reward_annealing
                    door.reward_annealing *= 0.975

        # if self.fire_boy:
        #     if 0 <= fb_y < self.level_height and 0 <= fb_x < self.level_width:
        #         if level_data[fb_y+2][fb_x+1] == 'P':
        #             reward += 50
        # if self.water_girl:
        #     if 0 <= wg_y < self.level_height and 0 <= wg_x < self.level_width:
        #         if level_data[wg_y+2][wg_x+1] == 'P':
        #             reward += 50

        # Reward for collecting stars
        stars = np.array(self.stars)
        is_collected = np.array([star.is_collected for star in stars])
        reward_given = np.array([star.reward_given for star in stars])

        num_fireboy_stars_collected = sum(
            star.is_collected and star._player == 'fire' for star in self.stars)
        num_watergirl_stars_collected = sum(
            star.is_collected and star._player == 'water' for star in self.stars)

        for i, star in enumerate(stars):
            if is_collected[i] and not reward_given[i]:
                # reward += 1000
                if star._player == 'fire':
                    fb_reward += 0.0625 * \
                        (num_fireboy_stars_collected +
                         1) * (num_watergirl_stars_collected + 1)
                if star._player == 'water':
                    wg_reward += 0.0625 * \
                        (num_fireboy_stars_collected +
                         1) * (num_watergirl_stars_collected + 1)
                star.reward_given = True

        # Reward for goal
        doors = np.array(self.doors)
        player_at_door = np.array([door.player_at_door for door in doors])
        reward_given = np.array([door.reward_given for door in doors])
        for i, door in enumerate(doors):
            if player_at_door[i] and not reward_given[i]:
                # reward += 4000
                door.reward_given = True
                if door._player == 'fire':
                    print('FIREBOY AT DOOR!')
                if door._player == 'water':
                    print('WATERGIRL AT DOOR!')
        if all(player_at_door) and not all(reward_given):
            fb_reward += 100  # or your chosen value
            wg_reward += 100  # or your chosen value
            for door in doors:
                door.reward_given = True
            print('BOTH PLAYERS AT DOORS!')

        return fb_reward + wg_reward
        return 0.9*min(fb_reward, wg_reward) + 0.1*max(fb_reward, wg_reward)

    def _check_done(self):
        # return False
        return all(door.player_at_door for door in self.doors)
        # End episode if either agent is at a forbidden tile
        level_data = self.board.get_level_data()
        fb_x, fb_y = np.array(self.fire_boy.get_position()) // 16
        fb_x, fb_y = int(fb_x-1), int(fb_y-1)
        wg_x, wg_y = np.array(self.water_girl.get_position()) // 16
        wg_x, wg_y = int(wg_x-1), int(wg_y-1)

        forbidden_fb = False
        forbidden_wg = False

        if 0 <= fb_y < self.level_height and 0 <= fb_x < self.level_width:
            tile_fb = level_data[fb_y+1][fb_x+1]
            if tile_fb in ['W', 'G']:  # Water, Goo, Lava
                forbidden_fb = True
        if 0 <= wg_y < self.level_height and 0 <= wg_x < self.level_width:
            tile_wg = level_data[wg_y+1][wg_x+1]
            if tile_wg in ['G', 'L']:  # Water, Goo, Lava
                forbidden_wg = True
            # End episode if forbidden tile touched or both players at their doors
        return forbidden_fb or forbidden_wg or all(door.player_at_door for door in self.doors)


register(
    id="FireboyAndWatergirl-ppo-v13",
    entry_point="cleanrl.v13.fireboy_and_watergirl_ppo_v13:FireboyAndWatergirlEnv",
)
