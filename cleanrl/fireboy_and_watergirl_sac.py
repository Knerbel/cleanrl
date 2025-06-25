from matplotlib import pyplot as plt
from fireboy_and_watergirl.star import Star
import time
from fireboy_and_watergirl.gate import Gate
from fireboy_and_watergirl.game import Game
from fireboy_and_watergirl.doors import FireDoor, WaterDoor
from fireboy_and_watergirl.controller import GeneralController
from fireboy_and_watergirl.character import FireBoy, WaterGirl
from fireboy_and_watergirl.board import Board
from gymnasium.envs.registration import register
import numpy as np
from gymnasium import spaces
import gymnasium as gym
import cv2


class FireboyAndWatergirlEnv(gym.Env):
    """
    Custom Environment for Fireboy and Watergirl that follows the Gymnasium API.
    """
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self):
        super(FireboyAndWatergirlEnv, self).__init__()

        # Define the action space (e.g., discrete actions for movement)
        # Actions: 0 = Fireboy Left, 1 = Fireboy Right, 2 = Fireboy Up, 3 = Fireboy Still,
        #          4 = Watergirl Left, 5 = Watergirl Right, 6 = Watergirl Up, 7 = Watergirl Still
        self.action_space = spaces.Discrete(8)
        # Initialize game components
        self.level = "level1b"
        self.game = Game()  # Instantiate the Game class
        self.board = None
        self.fire_boy = None
        self.water_girl = None
        self.gates = None
        self.doors: list[FireDoor | WaterDoor] = None
        self.controller = GeneralController()

        # Initialize game state
        self.state = None
        self.done = False

        # Load the level
        self._load_level1()

        self.steps = 0
        self.max_steps = 400

        self.level_height = 25 - 2  # Assuming 1-tile border on top and bottom
        self.level_width = 34 - 2   # Assuming 1-tile border on left and right
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.level_height, self.level_width, 3),
            dtype=np.uint8
        )

    def get_action_meanings(self):
        return [
            "NOOP",
            "Fireboy Right",
            "Fireboy Up",
            "Fireboy Still",
            "Watergirl Left",
            "Watergirl Right",
            "Watergirl Up",
            "Watergirl Still",
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

        fire_boy_location = (200, 336)
        self.fire_boy = FireBoy(fire_boy_location)
        water_girl_location = (200, 336)
        self.water_girl = WaterGirl(water_girl_location)

        fire_door_location = (64, 48)
        fire_door = FireDoor(fire_door_location)
        water_door_location = (128, 48)
        water_door = WaterDoor(water_door_location)
        self.doors = [fire_door, water_door]

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

    def _load_level1(self):
        """
        Load the level data and initialize game components.
        """
        if self.level == "level1" or self.level == "level1b":
            self.board = Board('./fireboy_and_watergirl/data/level1b.txt')
            gate_location = (285, 128)
            plate_locations = [(190, 168), (390, 168)]
            gate = Gate(gate_location, plate_locations)
            self.gates = [gate]

            fire_door_location = (64, 48)
            fire_door = FireDoor(fire_door_location)
            water_door_location = (128, 48)
            water_door = WaterDoor(water_door_location)
            self.doors = [fire_door, water_door]

            fire_boy_location = (16, 336)
            self.fire_boy = FireBoy(fire_boy_location)
            water_girl_location = (35, 336)
            self.water_girl = WaterGirl(water_girl_location)

            self.stars = [
                Star((240, 330), "fire"),
                Star((260, 330), "water"),

                Star((480, 300), "fire"),
                Star((500, 300), "water"),

                Star((370, 240), "fire"),
                Star((390, 240), "water"),

                Star((30, 200), "fire"),
                Star((50, 200), "water"),
            ]

        # Add more levels as needed

    # def _load_level2(self):
    #     """
    #     Load the level data from the file and dynamically set up the game components.
    #     """
    #     # Read the level data from the file
    #     with open('./fireboy_and_watergirl/data/'+self.level+'.txt', 'r') as file:
    #         level_data = [line.strip().split(',') for line in file.readlines()]
    #    # Initialize game components
    #     self.board = Board('./fireboy_and_watergirl/data/'+self.level+'.txt')
    #     self.gates: list[Gates] = []
    #     self.doors: list[FireDoor | WaterDoor] = []
    #     self.stars: list[Stars] = []
    #     self.fire_boy: FireBoy = None
    #     self.water_girl: WaterGirl = None
    #    # Parse the level data to dynamically set up components
    #     for y, row in enumerate(level_data):
    #         for x, tile in enumerate(row):
    #             # Assuming 16x16 tiles
    #             if tile == 'f':  # Fireboy starting position
    #                 self.fire_boy = FireBoy((x * 16, y * 16))
    #             elif tile == 'w':  # Watergirl starting position
    #                 self.water_girl = WaterGirl((x * 16, y * 16))
    #             elif tile == 'A':  # Fire door
    #                 self.doors.append(FireDoor((x * 16, y * 16)))
    #             elif tile == 'B':  # Water door
    #                 self.doors.append(WaterDoor((x * 16, y * 16)))
    #             # elif tile == 'D':  # Gate
    #                 # Add a generic gate (you can customize this further)
    #                 # self.gates.append(Gates((x * 16, y * 16), []))
    #             # elif tile == 'P':  # Plate A
    #                 # Add a plate that controls a gate
    #                 # self.gates.append(
    #                 #     Plate((x * 16, y * 16), [(x * 16, y * 16)]))
    #             # elif tile == 'B':  # Plate B
    #                 # self.gates.append(
    #                 # Gates((x * 16, y * 16), [(x * 16, y * 16)]))
    #             elif tile == 'a':  # Star A
    #                 self.stars.append(Stars([x * 16, y * 16], "fire"))
    #             elif tile == 'b':  # Star B
    #                 self.stars.append(Stars([x * 16, y * 16], "water"))
    #             # Add more cases as needed for other tiles

    def reset(self, seed=None, options=None):
        """
        Reset the environment to its initial state and return the initial observation.
        """
        super().reset(seed=seed)
        self.steps = 0

        self._load_level1()
        self.state = self._get_state()
        self.done = False
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

        # Check if the game is done
        self.done = False  # self._check_done()

        self.steps += 1
        if self.steps % 800 == 0:  # and self.game.index % 4 == 0:
            self._get_state(draw=True)
        if self.steps >= self.max_steps:
            self.done = True

        # Optionally, provide additional info
        info = {}

        return self.state, reward, self.done, False, info

    def render(self, mode="human"):
        """
        Render the environment to the screen or other output.
        Shows only the RGB observation used by the agent.
        """
        if mode == "human":
            # Get the RGB observation
            rgb_image = self._get_state()

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
                # Only print severe errors, not just window closed
                if not ("closed" in str(e).lower() or "destroy" in str(e).lower()):
                    print(f"Warning: {e}")

            return rgb_image

        elif mode == "rgb_array":
            return self._get_state()

    def close(self):
        """
        Clean up resources when the environment is closed.
        """
        pass

    def _get_state(self, draw=False):
        level_data = self.board.get_level_data()

        # Ignore the outer border (typically 1 tile) since it's always constant
        # Determine the playable area boundaries (ignore outer walls)
        border_size = 1  # Assuming outer wall is 1 tile thick

        # Initialize empty RGB grid for the inner area only
        inner_height = len(level_data) - (2 * border_size)
        inner_width = len(level_data[0]) - (2 * border_size)
        rgb_image = np.zeros((inner_height, inner_width, 3), dtype=np.uint8)

        # Color mapping for each tile (RGB format)
        color_mapping = {
            # Base tiles
            ' ': [255, 255, 255],  # Air - white
            'S': [50, 50, 50],     # Stone - dark gray

            # Hazards
            'L': [255, 50, 0],     # Lava - red
            'W': [0, 100, 255],    # Water - blue
            'G': [50, 200, 50],    # Goo - green

            # Characters
            'f': [255, 0, 0],      # Fireboy - red
            'w': [0, 0, 255],      # Watergirl - blue

            # Doors
            'A': [200, 100, 0],    # Fire door - amber
            'B': [0, 150, 200],    # Water door - aqua

            # Stars
            'a': [255, 200, 0],    # Fire star - gold
            'b': [0, 200, 255],    # Water star - cyan

            # Other elements
            'P': [150, 150, 150],  # Pressure plate - gray
            'D': [200, 200, 100],  # Gate - yellow-ish
        }

        # Fill the RGB image based on the inner level data
        for y in range(border_size, len(level_data) - border_size):
            for x in range(border_size, len(level_data[0]) - border_size):
                # Adjust coordinates for the inner grid
                inner_y = y - border_size
                inner_x = x - border_size
                rgb_image[inner_y, inner_x] = color_mapping.get(
                    level_data[y][x], [100, 100, 100])  # Default gray for unknown tiles

        # Adjust coordinates for dynamic elements to account for border removal

        # Add dynamic elements (characters, stars, etc.)
        if self.fire_boy:
            fb_x, fb_y = self.fire_boy.get_position()
            # Convert to grid coordinates and adjust for border
            fb_x, fb_y = int(fb_x // 16) - \
                border_size, int(fb_y // 16) - border_size
            if 0 <= fb_y < rgb_image.shape[0] and 0 <= fb_x < rgb_image.shape[1]:
                rgb_image[fb_y, fb_x] = color_mapping['f']  # Fireboy color

        if self.water_girl:
            wg_x, wg_y = self.water_girl.get_position()
            # Convert to grid coordinates and adjust for border
            wg_x, wg_y = int(wg_x // 16) - \
                border_size, int(wg_y // 16) - border_size
            if 0 <= wg_y < rgb_image.shape[0] and 0 <= wg_x < rgb_image.shape[1]:
                rgb_image[wg_y, wg_x] = color_mapping['w']  # Watergirl color

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

        # Add doors to the image
        # for door in self.doors:
        #     d_x, d_y = door.get_position()
        #     # Convert to grid coordinates and adjust for border
        #     d_x, d_y = int(d_x // 16) - \
        #         border_size, int(d_y // 16) - border_size
        #     if 0 <= d_y < rgb_image.shape[0] and 0 <= d_x < rgb_image.shape[1]:
        #         if isinstance(door, FireDoor):
        #             rgb_image[d_y, d_x] = color_mapping['A']  # Fire door
        #         else:
        #             rgb_image[d_y, d_x] = color_mapping['B']  # Water door

        # Add gates to the image
        # for gate in self.gates:
        #     g_x, g_y = gate.get_position()
        #     # Convert to grid coordinates and adjust for border
        #     g_x, g_y = int(g_x // 16) - \
        #         border_size, int(g_y // 16) - border_size
        #     if 0 <= g_y < rgb_image.shape[0] and 0 <= g_x < rgb_image.shape[1]:
        #         rgb_image[g_y, g_x] = color_mapping['D']  # Gate

        #     # Add pressure plates
        #     for plate_loc in gate.get_plate_positions():
        #         p_x, p_y = plate_loc
        #         # Convert to grid coordinates and adjust for border
        #         p_x, p_y = int(p_x // 16) - \
        #             border_size, int(p_y // 16) - border_size
        #         if 0 <= p_y < rgb_image.shape[0] and 0 <= p_x < rgb_image.shape[1]:
        #             rgb_image[p_y, p_x] = color_mapping['P']  # Pressure plate

        # Save image if requested
        if draw:
            plt.figure(figsize=(10, 8))
            plt.imshow(rgb_image)
            plt.axis('off')
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            plt.savefig(f"observation.png", bbox_inches='tight', pad_inches=0)
            plt.close()

        # Update observation space if this is the first time running
        if self.observation_space.shape != (inner_height, inner_width, 3):
            self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(inner_height, inner_width, 3),
                dtype=np.uint8
            )
            print(
                f"Updated observation space to: {self.observation_space.shape}")

        return rgb_image

    def _apply_action(self, action):
        """
        Update the game state based on the discrete action.
        """
        # Decode the single discrete action into Fireboy and Watergirl actions
        # fireboy_action = action // 4  # Integer division to get Fireboy's action
        # watergirl_action = action % 4  # Modulo to get Watergirl's action

        fireboy_action = 3
        watergirl_action = 3

        # Map the action to Fireboy and Watergirl actions
        if action <= 3:
            fireboy_action = action
        elif action >= 4:
            watergirl_action = action - 4

        # Map Fireboy's action
        if fireboy_action == 0:
            self.fire_boy.moving_left = True
            self.fire_boy.moving_right = False
            self.fire_boy.jumping = False
        elif fireboy_action == 1:
            self.fire_boy.moving_left = False
            self.fire_boy.moving_right = True
            self.fire_boy.jumping = False
        elif fireboy_action == 2:
            self.fire_boy.moving_left = False
            self.fire_boy.moving_right = False
            self.fire_boy.jumping = True
        elif fireboy_action == 3:
            self.fire_boy.moving_left = False
            self.fire_boy.moving_right = False
            self.fire_boy.jumping = False

        # Map Watergirl's action
        if watergirl_action == 0:
            self.water_girl.moving_left = True
            self.water_girl.moving_right = False
            self.water_girl.jumping = False
        elif watergirl_action == 1:
            self.water_girl.moving_left = False
            self.water_girl.moving_right = True
            self.water_girl.jumping = False
        elif watergirl_action == 2:
            self.water_girl.moving_left = False
            self.water_girl.moving_right = False
            self.water_girl.jumping = True
        elif watergirl_action == 3:
            self.water_girl.moving_left = False
            self.water_girl.moving_right = False
            self.water_girl.jumping = False

        # Update the game state
        self.game.move_player(self.board, self.gates, [
                              self.fire_boy, self.water_girl])
        self.game.check_for_plates_press(
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

        # self.game.check_for_at_door(self.doors, self.fire_boy)
        # self.game.check_for_at_door(self.doors, self.water_girl)

        fireboy_reward = 0
        watergirl_reward = 0

        for star in self.stars:
            if star.is_collected and not star.reward_given:
                if star._player == "fire":
                    fireboy_reward += 0.05
                else:
                    watergirl_reward += 0.05
                # star.reward_given = True

        for door in self.doors:
            if door.player_at_door and not door.reward_given:
                print('PLAYER AT DOOR')
                if (door._player == "fire"):
                    fireboy_reward += 0.4
                else:
                    watergirl_reward += 0.4
                # door.reward_given = True
        # print(fireboy_reward, watergirl_reward)

        # Reward for exploring new areas

        return (0.9*min(fireboy_reward, watergirl_reward) + 0.1 * max(fireboy_reward, watergirl_reward))

    # def _compute_reward(self):
    #     reward = 0

    #     # Death penalty
    #     if self.fire_boy.is_dead() or self.water_girl.is_dead():
    #         reward -= 1

    #     # Star collection rewards
    #     for star in self.stars:
    #         if star.is_collected and not star.reward_given:
    #             reward += 5
    #             star.reward_given = True  # Flag to avoid repeated rewards

    #     # Distance-based shaping
    #     closest_star_dist = float('inf')
    #     for star in self.stars:
    #         if not star.is_collected:
    #             fb_pos = self.fire_boy.get_position()
    #             wg_pos = self.water_girl.get_position()

    #             # Get appropriate character position based on star type
    #             char_pos = fb_pos if star._player == "fire" else wg_pos

    #             # Calculate distance
    #             star_pos = star.get_position()
    #             dist = np.sqrt((char_pos[0] - star_pos[0])
    #                            ** 2 + (char_pos[1] - star_pos[1])**2)

    #             closest_star_dist = min(closest_star_dist, dist)

    #     # Store previous distance for comparison
    #     if not hasattr(self, 'prev_closest_star_dist'):
    #         self.prev_closest_star_dist = closest_star_dist

    #     # Reward for getting closer to a star
    #     if closest_star_dist < self.prev_closest_star_dist:
    #         reward += 0.01 * (self.prev_closest_star_dist - closest_star_dist)

    #     self.prev_closest_star_dist = closest_star_dist

    #     return reward

    def _check_done(self):
        """
        Check if the game is over.
        """

        # Example: game is done if both characters reach their doors or one dies
        return self.game.level_is_done(self.doors) or \
            self.game.check_for_death(
                self.board, [self.fire_boy, self.water_girl])

    def lives(self):
        # Return a dummy value (e.g., 1 life remaining)
        return 1


# Register the environment
register(
    id="FireboyAndWatergirl-sac-v0",  # Unique ID for the environment
    # Path to the environment class
    entry_point="cleanrl.fireboy_and_watergirl_sac:FireboyAndWatergirlEnv",
)
