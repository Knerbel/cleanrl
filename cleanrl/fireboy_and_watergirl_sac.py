from matplotlib import pyplot as plt
from fireboy_and_watergirl.stars import Stars
import time
from fireboy_and_watergirl.gates import Gates
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
        # 4 actions for Fireboy, 4 actions for Watergirl
        # self.action_space = spaces.MultiDiscrete([4, 4])

        # Define the observation space (feature-based representation)
        # [Fireboy_x, Fireboy_y, Watergirl_x, Watergirl_y, Gate_status, Door_status]
        # self.observation_space = spaces.Box(
        #     low=np.array([0, 0, 0, 0, 0, 0]),
        #     # Adjust max_x and max_y as needed
        #     high=np.array([1000, 1000, 1000, 1000, 1, 1]),
        #     dtype=np.float32
        # )

        # Initialize game components
        self.level = "level1b"
        self.game = Game()  # Instantiate the Game class
        self.board = None
        self.fire_boy = None
        self.water_girl = None
        self.gates = None
        self.doors = None
        self.controller = GeneralController()

        # Initialize game state
        self.state = None
        self.done = False

        # Load the level
        self._load_level1()

        self.steps = 0
        self.max_steps = 400

        self.level_height = 25
        self.level_width = 34
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            # Example: (25, 34, 3)
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

            fire_boy_location = (16, 336)
            self.fire_boy = FireBoy(fire_boy_location)
            water_girl_location = (35, 336)
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
        if self.steps >= self.max_steps:
            self.done = True

        if self.done:
            self._get_state(draw=True)

        # Optionally, provide additional info
        info = {}

        return self.state, reward, self.done, False, info

    def render(self, mode="human"):
        """
        Render the environment to the screen or other output.
        """

        if mode == "human":
            # Use the Game class to render the game
            self.game.draw_level_background(self.board)
            self.game.draw_board(self.board)
            if self.gates:
                self.game.draw_gates(self.gates)
            self.game.draw_doors(self.doors)
            self.game.draw_player([self.fire_boy, self.water_girl])
            self.game.draw_stars(self.stars)
            self.game.refresh_window()

    def close(self):
        """
        Clean up resources when the environment is closed.
        """
        pass

    def _get_state(self, draw=False):
        level_data = self.board.get_level_data()

        # Initialize empty RGB grid
        rgb_image = np.zeros(
            (len(level_data), len(level_data[0]), 3), dtype=np.uint8)

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
            # Fireboy - orange-red (distinguishable from lava)
            'f': [255, 0, 0],
            # Watergirl - light blue (distinguishable from water)
            'w': [0, 0, 255],

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

        # Fill the RGB image based on the level data
        for y, row in enumerate(level_data):
            for x, tile in enumerate(row):
                rgb_image[y, x] = color_mapping.get(
                    tile, [100, 100, 100])  # Default gray for unknown tiles

        # Add dynamic elements (characters, stars, etc.)
        # Get positions and add them to the image
        if self.fire_boy:
            fb_x, fb_y = self.fire_boy.get_position()
            # Convert to grid coordinates
            fb_x, fb_y = int(fb_x // 16), int(fb_y // 16)
            if 0 <= fb_y < rgb_image.shape[0] and 0 <= fb_x < rgb_image.shape[1]:
                rgb_image[fb_y, fb_x] = color_mapping['f']  # Fireboy color

        if self.water_girl:
            wg_x, wg_y = self.water_girl.get_position()
            # Convert to grid coordinates
            wg_x, wg_y = int(wg_x // 16), int(wg_y // 16)
            if 0 <= wg_y < rgb_image.shape[0] and 0 <= wg_x < rgb_image.shape[1]:
                rgb_image[wg_y, wg_x] = color_mapping['w']  # Watergirl color

        # Add stars to the image
        for star in self.stars:
            if not star.is_collected:
                s_x, s_y = star.get_position()
                # Convert to grid coordinates
                s_x, s_y = int(s_x // 16), int(s_y // 16)
                if 0 <= s_y < rgb_image.shape[0] and 0 <= s_x < rgb_image.shape[1]:
                    if star._player == "fire":
                        rgb_image[s_y, s_x] = color_mapping['a']  # Fire star
                    else:
                        rgb_image[s_y, s_x] = color_mapping['b']  # Water star

        # Add doors to the image
        for door in self.doors:
            d_x, d_y = door.get_position()
            # Convert to grid coordinates
            d_x, d_y = int(d_x // 16), int(d_y // 16)
            if 0 <= d_y < rgb_image.shape[0] and 0 <= d_x < rgb_image.shape[1]:
                if isinstance(door, FireDoor):
                    rgb_image[d_y, d_x] = [200, 100, 0]  # Fire door
                else:
                    rgb_image[d_y, d_x] = [0, 150, 200]  # Water door

        # Save image if requested
        if draw:
            plt.figure(figsize=(10, 8))
            plt.imshow(rgb_image)
            plt.axis('off')  # Remove axes for a clean image
            plt.savefig(f"observation_{self.steps}.png",
                        bbox_inches='tight', pad_inches=0)
            plt.close()
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
        elif self.game.check_for_death(self.board, [self.fire_boy, self.water_girl]):
            return -1  # One of the characters died

        fireboy_reward = 0
        watergirl_reward = 0

        for star in self.stars:
            if star.is_collected:
                if (star._player == "fire"):
                    fireboy_reward += 5
                else:
                    watergirl_reward += 5

        # Combine rewards for both agents
        return min(fireboy_reward, watergirl_reward) + 0.1 * max(fireboy_reward, watergirl_reward)

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
