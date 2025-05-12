import cv2
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

        # Define the action space (e.g., discrete actions for movement)
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(8,), dtype=np.float32)
        # Initialize game components
        self.level = "level1b"
        self.game = Game()  # Instantiate the Game class
        self.board = None
        self.fire_boy = None
        self.water_girl = None
        self.gates = None
        self.doors = None
        # self.controller = GeneralController()

        # Initialize game state
        self.state = None
        self.done = False
        # Load the level
        self._load_level1()

        self.steps = 0
        self.max_steps = 200

        self.level_height = 25 - 2  # Assuming 1-tile border on top and bottom
        self.level_width = 34 - 2   # Assuming 1-tile border on left and right
        self.num_channels = 3       # RGB channels
        # Define the flattened observation space
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.level_height * self.level_width * self.num_channels,),
            dtype=np.float32
        )

    def _load_level(self):
        """
        Load the level data from the file and dynamically set up the game components.
        """
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
                elif tile == 'D':  # Gate
                    # Add a generic gate (you can customize this further)
                    self.gates.append(Gates((x * 16, y * 16), []))
                # elif tile == 'P':  # Plate A
                    # Add a plate that controls a gate
                    # self.gates.append(
                    #     Plate((x * 16, y * 16), [(x * 16, y * 16)]))
                elif tile == 'B':  # Plate B
                    self.gates.append(
                        Gates((x * 16, y * 16), [(x * 16, y * 16)]))
                elif tile == 'a':  # Gate A
                    self.stars.append(Stars([x * 16, y * 16], "fire"))
                elif tile == 'b':  # Gate B
                    self.stars.append(Stars([x * 16, y * 16], "water"))
                # Add more cases as needed for other tiles

        # Flatten the level data into a 1D array for the observation space

    def _load_level1(self):
        """
        Load the level data and initialize game components.
        """
        if self.level == "level1" or self.level == "level1b":
            self.board = Board('./fireboy_and_watergirl/data/level1b.txt')
            gate_location = (285, 128)
            plate_locations = [(190, 168), (390, 168)]
            gate = Gates(gate_location, plate_locations)
            self.gates = [gate]

            fire_door_location = (64, 48)
            fire_door = FireDoor(fire_door_location)
            water_door_location = (128, 48)
            water_door = WaterDoor(water_door_location)
            self.doors = [fire_door, water_door]

            fire_boy_location = (200+16, 336)
            self.fire_boy = FireBoy(fire_boy_location)
            water_girl_location = (200+35, 336)
            self.water_girl = WaterGirl(water_girl_location)

            self.stars = [
                Stars((240, 330), "fire"),
                Stars((260, 330), "water"),

                Stars((480, 300), "fire"),
                Stars((500, 300), "water"),

                Stars((370, 240), "fire"),
                Stars((390, 240), "water"),

                Stars((30, 200), "fire"),
                Stars((50, 200), "water"),
            ]

    def reset(self, seed=None, options=None):
        """
        Reset the environment to its initial state and return the initial observation.
        """
        super().reset(seed=seed)
        self._load_level1()
        self.state = self._get_state()
        self.done = False
        self.steps = 0
        return self.state.astype(np.float32), {}

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

        # Check if the game is done
        self.done = False  # self._check_done()

        self.steps += 1
        if self.steps >= self.max_steps:
            self.done = True

        if (self.done):
            self._get_state(draw=True)

        # Optionally, provide additional info
        info = {}

        return self.state.astype(np.float32), reward, self.done, False, info

    def render(self, mode="human"):
        """
        Render the environment to the screen or other output.
        Shows only the RGB observation used by the agent.
        """
        if mode == "human":
            # Get the flattened RGB observation
            flattened_image = self._get_state()

            # Reshape the flattened image back to its original 3D shape
            rgb_image = flattened_image.reshape(
                self.level_height, self.level_width, self.num_channels
            )

            # Create a global matplotlib instance (static class variable)
            # that persists across environment instances
            if not hasattr(FireboyAndWatergirlEnv, 'plt_initialized'):
                print("Initializing matplotlib...")
                import matplotlib
                matplotlib.use('TkAgg')  # Force TkAgg backend
                import matplotlib.pyplot as plt
                plt.ion()  # Turn on interactive mode

                # Store the initialized modules as class variables
                FireboyAndWatergirlEnv.matplotlib = matplotlib
                FireboyAndWatergirlEnv.plt = plt
                FireboyAndWatergirlEnv.plt_initialized = True

                # Create the figure as a class variable
                FireboyAndWatergirlEnv.render_fig = plt.figure(
                    figsize=(8, 6), num="Fireboy & Watergirl")
                FireboyAndWatergirlEnv.render_ax = FireboyAndWatergirlEnv.render_fig.add_subplot(
                    111)
                FireboyAndWatergirlEnv.render_img = FireboyAndWatergirlEnv.render_ax.imshow(
                    rgb_image)
                FireboyAndWatergirlEnv.render_ax.axis('off')
                FireboyAndWatergirlEnv.render_fig.tight_layout()

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
                # Only print severe errors, not just window closed
                if not ("closed" in str(e).lower() or "destroy" in str(e).lower()):
                    print(f"Warning: {e}")

            return rgb_image

        elif mode == "rgb_array":
            # Return the reshaped RGB array for rendering
            flattened_image = self._get_state()
            return flattened_image.reshape(
                self.level_height, self.level_width, self.num_channels
            )

    def close(self):
        """
        Clean up resources when the environment is closed.
        """
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
            ' ': [255, 255, 255],  # Air - white
            'S': [50, 50, 50],     # Stone - dark gray
            'L': [255, 50, 0],     # Lava - red
            'W': [0, 100, 255],    # Water - blue
            'G': [50, 200, 50],    # Goo - green
            'f': [255, 0, 0],      # Fireboy - red
            'w': [0, 0, 255],      # Watergirl - blue
            'A': [200, 100, 0],    # Fire door - amber
            'B': [0, 150, 200],    # Water door - aqua
            'a': [255, 200, 0],    # Fire star - gold
            'b': [0, 200, 255],    # Water star - cyan
            'P': [150, 150, 150],  # Pressure plate - gray
            'D': [200, 200, 100],  # Gate - yellow-ish
        }

        for y in range(border_size, len(level_data) - border_size):
            for x in range(border_size, len(level_data[0]) - border_size):
                inner_y = y - border_size
                inner_x = x - border_size
                rgb_image[inner_y, inner_x] = color_mapping.get(
                    level_data[y][x], [100, 100, 100])  # Default gray for unknown tiles

        # Add dynamic elements (characters, stars, etc.)
        if self.fire_boy:
            fb_x, fb_y = self.fire_boy.get_position()
            fb_x, fb_y = int(fb_x // 16) - \
                border_size, int(fb_y // 16) - border_size
            if 0 <= fb_y < rgb_image.shape[0] and 0 <= fb_x < rgb_image.shape[1]:
                rgb_image[fb_y, fb_x] = color_mapping['f']

        if self.water_girl:
            wg_x, wg_y = self.water_girl.get_position()
            wg_x, wg_y = int(wg_x // 16) - \
                border_size, int(wg_y // 16) - border_size
            if 0 <= wg_y < rgb_image.shape[0] and 0 <= wg_x < rgb_image.shape[1]:
                rgb_image[wg_y, wg_x] = color_mapping['w']

        # Add stars to the image
        for star in self.stars:
            if not star.is_collected:
                s_x, s_y = star.get_position()
                # Convert to grid coordinates and adjust for border
                s_x, s_y = int(s_x // 16) - \
                    border_size, int(s_y // 16) - border_size
                if 0 <= s_y < rgb_image.shape[0] and 0 <= s_x < rgb_image.shape[1]:
                    if star._player == "fire":
                        rgb_image[s_y, s_x] = color_mapping['a']  # Fire star
                    else:
                        rgb_image[s_y, s_x] = color_mapping['b']  # Water star

        # Flatten the RGB image
        flattened_image = rgb_image.flatten()

        # Save image if requested
        if draw:
            plt.figure(figsize=(10, 8))
            plt.imshow(rgb_image)
            plt.axis('off')
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            plt.savefig(f"observation.png", bbox_inches='tight', pad_inches=0)
            plt.close()

        return flattened_image

    def _apply_action(self, action):
        """
        Update the game state based on the continuous action.
        The action space consists of 8 continuous values (0 to 1):
        - action[0]: Fireboy moves left
        - action[1]: Fireboy moves right
        - action[2]: Fireboy jumps
        - action[3]: Fireboy does nothing
        - action[4]: Watergirl moves left
        - action[5]: Watergirl moves right
        - action[6]: Watergirl jumps
        - action[7]: Watergirl does nothing
        """

        # Determine the index of the highest value for Fireboy's actions
        fireboy_action = np.argmax(action[:4])  # Actions [0, 1, 2, 3]

        # Map Fireboy's action
        if fireboy_action == 0:  # Move left
            self.fire_boy.moving_left = True
            self.fire_boy.moving_right = False
            self.fire_boy.jumping = False
        elif fireboy_action == 1:  # Move right
            self.fire_boy.moving_left = False
            self.fire_boy.moving_right = True
            self.fire_boy.jumping = False
        elif fireboy_action == 2:  # Jump
            self.fire_boy.moving_left = False
            self.fire_boy.moving_right = False
            self.fire_boy.jumping = True
        elif fireboy_action == 3:  # Do nothing
            self.fire_boy.moving_left = False
            self.fire_boy.moving_right = False
            self.fire_boy.jumping = False

        # Determine the index of the highest value for Watergirl's actions
        watergirl_action = np.argmax(action[4:])  # Actions [4, 5, 6, 7]

        # Map Watergirl's action
        if watergirl_action == 0:  # Move left
            self.water_girl.moving_left = True
            self.water_girl.moving_right = False
            self.water_girl.jumping = False
        elif watergirl_action == 1:  # Move right
            self.water_girl.moving_left = False
            self.water_girl.moving_right = True
            self.water_girl.jumping = False
        elif watergirl_action == 2:  # Jump
            self.water_girl.moving_left = False
            self.water_girl.moving_right = False
            self.water_girl.jumping = True
        elif watergirl_action == 3:  # Do nothing
            self.water_girl.moving_left = False
            self.water_girl.moving_right = False
            self.water_girl.jumping = False

        # Update the game state
        self.game.move_player(self.board, self.gates, [
                              self.fire_boy, self.water_girl])
        self.game.check_for_gate_press(
            self.gates, [self.fire_boy, self.water_girl])
        self.game.check_for_star_collected(
            self.stars, [self.fire_boy, self.water_girl])

    def _compute_reward(self):
        """
        Compute the reward for the current state.
        """
        # Example: +1 for reaching the door, -1 for falling into a trap
        if self.game.level_is_done(self.doors):
            return 1  # Both characters reached their doors
        # elif self.game.check_for_death(self.board, [self.fire_boy, self.water_girl]):
        #     return -1  # One of the characters died

        fireboy_reward = 0
        watergirl_reward = 0

        for star in self.stars:
            if star.is_collected:
                if (star._player == "fire"):
                    fireboy_reward += 1
                else:
                    watergirl_reward += 1

        # Reward based on proximity to the nearest star
        def distance_to_nearest_star(player, stars):
            player_x, player_y = player.get_position()
            distances = [
                np.sqrt((player_x - star.get_position()
                        [0])**2 + (player_y - star.get_position()[1])**2)
                for star in stars if not star.is_collected
            ]
            return min(distances) if distances else 0

        fireboy_distance_reward = - \
            distance_to_nearest_star(self.fire_boy, self.stars) * 0.01
        watergirl_distance_reward = - \
            distance_to_nearest_star(self.water_girl, self.stars) * 0.01

        # print(fireboy_distance_reward, watergirl_distance_reward)

        # Combine rewards for both agents
        # return min(fireboy_reward, watergirl_reward) + 0.1 * max(fireboy_reward, watergirl_reward)
        # return min(fireboy_distance_reward, watergirl_distance_reward) + 0.1 * max(fireboy_distance_reward, watergirl_distance_reward)
        # Reward for moving to the right
        fireboy_x, _ = self.fire_boy.get_position()
        watergirl_x, _ = self.water_girl.get_position()

        fireboy_right_reward = fireboy_x * 0.0001
        watergirl_right_reward = watergirl_x * 0.0001

        # Combine all rewards
        return +(fireboy_right_reward + watergirl_right_reward)

    def _check_done(self):
        """
        Check if the game is over.
        """

        # Example: game is done if both characters reach their doors or one dies
        return self.game.level_is_done(self.doors) or \
            self.game.check_for_death(
                self.board, [self.fire_boy, self.water_girl])


# Register the environment
register(
    id="FireboyAndWatergirl-td3-v0",  # Unique ID for the environment
    # Path to the environment class
    entry_point="cleanrl.fireboy_and_watergirl_td3:FireboyAndWatergirlEnv",
)
