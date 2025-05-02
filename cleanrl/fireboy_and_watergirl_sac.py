import cv2
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gymnasium.envs.registration import register

from fireboy_and_watergirl.board import Board
from fireboy_and_watergirl.character import FireBoy, WaterGirl
from fireboy_and_watergirl.controller import GeneralController
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
        self.level = "level1"
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

        self.visited_positions_x_fireboy = set()
        self.visited_positions_y_fireboy = set()
        self.visited_positions_y_watergirl = set()
        self.visited_positions_x_watergirl = set()

        self.flattened_level = []

        # Load the level
        self._load_level()

        self.level_height = 25
        self.level_width = 34
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            # Example: (34, 25, 1)
            shape=(self.level_width, self.level_height, 3),
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
        with open('./fireboy_and_watergirl/data/level1.txt', 'r') as file:
            level_data = [line.strip().split(',') for line in file.readlines()]

        # Initialize game components
        self.board = Board('./fireboy_and_watergirl/data/level1.txt')
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
        tile_mapping_numeric = {
            'S': 1, ' ': 0, 'L': 2, 'W': 3, 'G': 4, 'w': 5, 'f': 6,
            'a': 7, 'b': 8, 'P': 9, 'D': 10, 'A': 11, 'B': 12
        }
        level_numeric = [[tile_mapping_numeric[tile]
                          for tile in row] for row in level_data]
        self.flattened_level = np.array(
            level_numeric, dtype=np.float32).flatten()

    def reset(self, seed=None, options=None):
        """
        Reset the environment to its initial state and return the initial observation.
        """
        super().reset(seed=seed)
        self._load_level()
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

        # Update the board map dynamically
        self._update_board_map()

        # Compute reward
        reward = self._compute_reward()

        # Check if the game is done
        self.done = self._check_done()

        # Optionally, provide additional info
        info = {}

        return self.state, reward, self.done, False, info

    def _update_board_map(self):
        """
        Update the board map dynamically based on the current state of the game.
        """
        # Get the current level data
        level_data = self.board.get_level_data()

        # Create a copy of the level data to modify
        updated_level_data = [list(row) for row in level_data]

        # Clear previous player positions
        for y, row in enumerate(updated_level_data):
            for x, tile in enumerate(row):
                if tile in ['f', 'w']:  # Clear Fireboy and Watergirl positions
                    updated_level_data[y][x] = ' '

        # Update Fireboy's position
        fireboy_pos = self.fire_boy.get_position()
        if (fireboy_pos[0] > 0
           and fireboy_pos[0] < 1000
           and fireboy_pos[1] > 0
           and fireboy_pos[1] < 1000):
            # Convert to grid coordinates
            fireboy_x, fireboy_y = min(fireboy_pos[0] // 16, 33), min(
                fireboy_pos[1] // 16, 24)
            # Ensure 'S' is not overwritten
            if updated_level_data[fireboy_y][fireboy_x] != 'S':
                updated_level_data[fireboy_y][fireboy_x] = 'f'

        # Update Watergirl's position
        watergirl_pos = self.water_girl.get_position()
        if (watergirl_pos[0] > 0
           and watergirl_pos[0] < 1000
           and watergirl_pos[1] > 0
           and watergirl_pos[1] < 1000):

            # Convert to grid coordinates
            watergirl_x, watergirl_y = min(watergirl_pos[0] // 16, 33), min(
                watergirl_pos[1] // 16, 24)
            # Ensure 'S' is not overwritten
            if updated_level_data[watergirl_y][watergirl_x] != 'S':
                updated_level_data[watergirl_y][watergirl_x] = 'w'

        # Update gates and doors
        for gate in self.gates:
            gate_pos = gate.gate_position
            gate_x, gate_y = gate_pos[0] // 16, gate_pos[1] // 16
            # Ensure 'S' is not overwritten
            if updated_level_data[gate_y][gate_x] != 'S':
                updated_level_data[gate_y][gate_x] = 'D'

        for door in self.doors:
            door_pos = door.position
            door_x, door_y = door_pos[0] // 16, door_pos[1] // 16
            # Ensure 'S' is not overwritten
            if updated_level_data[door_y][door_x] != 'S':
                if isinstance(door, FireDoor):
                    updated_level_data[door_y][door_x] = 'A'
                elif isinstance(door, WaterDoor):
                    updated_level_data[door_y][door_x] = 'B'

        for star in self.stars:
            star_pos = star.postion
            star_x, star_y = star_pos[0] // 16, star_pos[1] // 16
            # Ensure 'S' is not overwritten
            if updated_level_data[star_y][star_x] != 'S':
                # print(star.is_collected)
                if not star.is_collected:
                    if star._player == "fire":
                        updated_level_data[star_y][star_x] = 'a'
                    else:
                        updated_level_data[star_y][star_x] = 'b'

        # Update the board with the new level data
        self.board.set_game_map(updated_level_data)

    def render(self, mode="human"):
        """
        Render the environment to the screen or other output.
        """

        if mode == "human":
            # Use the Game class to render the game
            self.game.draw_level_background(self.board)
            self.game.draw_board(self.board)
            # if self.gates:
            #     self.game.draw_gates(self.gates)
            # self.game.draw_doors(self.doors)
            # self.game.draw_player([self.fire_boy, self.water_girl])
            # self.game.draw_stars(self.stars)
            self.game.refresh_window()

    def close(self):
        """
        Clean up resources when the environment is closed.
        """
        pass

    def _get_state(self):
        level_data = self.board.get_level_data()
        tile_mapping = {
            'S': 1, ' ': 0, 'L': 2, 'W': 3, 'G': 4, 'w': 5, 'f': 6,
            'a': 7, 'b': 8, 'P': 9, 'D': 10, 'A': 11, 'B': 12
        }
        level_grid = np.array([[tile_mapping[tile] for tile in row]
                              for row in level_data], dtype=np.uint8)

        # Resize the grid
        resized_grid = cv2.resize(
            level_grid, (self.level_height, self.level_width), interpolation=cv2.INTER_NEAREST)

        # Duplicate the single channel to create an RGB image
        rgb_image = np.stack(
            [resized_grid, resized_grid, resized_grid], axis=-1)

        return rgb_image

    def _apply_action(self, action):
        """
        Update the game state based on the discrete action.
        """
        # Decode the single discrete action into Fireboy and Watergirl actions
        fireboy_action = action // 4  # Integer division to get Fireboy's action
        watergirl_action = action % 4  # Modulo to get Watergirl's action

        # fireboy_action = 3
        # watergirl_action = 3

        # # Map the action to Fireboy and Watergirl actions
        # if action <= 3:
        #     fireboy_action = action
        # elif action >= 4:
        #     watergirl_action = action - 4

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
        # Reward is based on how far right the agents have moved
        # fireboy_reward = self.fire_boy.get_position(
        # )[0] / 1000  # Normalize by max x-coordinate
        # watergirl_reward = self.water_girl.get_position(
        # )[0] / 1000  # Normalize by max x-coordinate

        # # Reward is heavily based on how high the agents have moved
        # fireboy_height_reward = self.fire_boy.get_position(
        # )[1] / 1000  # Normalize by max y-coordinate
        # watergirl_height_reward = self.water_girl.get_position(
        # )[1] / 1000  # Normalize by max y-coordinate

        # # Combine rewards with height being more important
        # return fireboy_reward * 0.5 + \
        #     watergirl_reward * 0.5 + \
        #     fireboy_height_reward * 0.5 + \
        #     watergirl_height_reward * 0.5

        # Calculate the distance to the respective doors
        fireboy_pos = self.fire_boy.get_position()
        watergirl_pos = self.water_girl.get_position()
        fire_door_pos = self.doors[0].get_position()
        water_door_pos = self.doors[1].get_position()

        fireboy_distance = np.sqrt(
            (fire_door_pos[0]-fireboy_pos[0]) *
            (fire_door_pos[0]-fireboy_pos[0])
            + (fire_door_pos[1]-fireboy_pos[1]) *
              (fire_door_pos[1]-fireboy_pos[1]))

        watergirl_distance = np.sqrt(
            (water_door_pos[0]-watergirl_pos[0]) *
            (water_door_pos[0]-watergirl_pos[0])
            + (water_door_pos[1]-watergirl_pos[1]) *
            (water_door_pos[1]-watergirl_pos[1]))

        # Normalize distances (assuming max distance is 1000 for simplicity)
        fireboy_reward = min(1000 / max(fireboy_distance, 1), 10)
        watergirl_reward = min(1000 / max(watergirl_distance, 1), 10)

        # Give a big reward when players are at their respective doors
        # if fireboy_distance < 1:
        #     fireboy_reward += 10
        # if watergirl_distance < 1:
        #     watergirl_reward += 10

        old_x_fireboy = len(self.visited_positions_x_fireboy)
        old_x_watergirl = len(self.visited_positions_x_watergirl)
        old_y_fireboy = len(self.visited_positions_y_fireboy)
        old_y_watergirl = len(self.visited_positions_y_watergirl)

        self.visited_positions_x_fireboy.add(int(fireboy_pos[0]*10))
        self.visited_positions_x_watergirl.add(int(watergirl_pos[0]*10))
        self.visited_positions_y_watergirl.add(int(fireboy_pos[1]*10))
        self.visited_positions_y_fireboy.add(int(watergirl_pos[1]*10))
        # Reward for exploring new positions

        # Apply a small penalty for jumping to discourage unnecessary jumps

        # print(fireboy_reward, watergirl_reward)
        # print(len(self.visited_positions_x))

        fireboy_reward = np.round(len(self.visited_positions_x_fireboy) * 0.01 +
                                  len(self.visited_positions_y_fireboy) * 0.01, 2)

        watergirl_reward = np.round(len(self.visited_positions_x_watergirl) * 0.01 +
                                    len(self.visited_positions_y_watergirl) * 0.01, 2)

        fireboy_reward += np.round((len(self.visited_positions_x_fireboy) - old_x_fireboy) * 0.05 +
                                   (len(self.visited_positions_y_fireboy) - old_y_fireboy) * 0.05, 2)

        watergirl_reward += np.round((len(self.visited_positions_x_watergirl) - old_x_watergirl) * 0.05 +
                                     (len(self.visited_positions_y_watergirl) - old_y_watergirl) * 0.05, 2)

        # Reward is based on how far right the agents have moved
        # Normalize by max x-coordinate
        fireboy_reward += self.fire_boy.get_position()[0] * 0.01
        # Normalize by max x-coordinate
        watergirl_reward += self.water_girl.get_position()[0] * 0.01

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
        return min(fireboy_reward, watergirl_reward)
        # Reward is also based on how far left the agents have moved
        # Normalize by max x-coordinate
        # fireboy_left_reward = (- self.fire_boy.get_position()[0]) / 1000
        # # Normalize by max x-coordinate
        # watergirl_left_reward = (
        #     - self.water_girl.get_position()[0]) / 1000

        # Reward is based on how high the agents have moved
        # fireboy_height_reward = self.fire_boy.get_position(
        # )[1] / 1000  # Normalize by max y-coordinate
        # watergirl_height_reward = self.water_girl.get_position(
        # )[1] / 1000  # Normalize by max y-coordinate

        # Reward is based on how low the agents have moved
        # Normalize by max y-coordinate
        # fireboy_depth_reward = (10000 - self.fire_boy.get_position()[1]) / 1000
        # Normalize by max y-coordinate
        # watergirl_depth_reward = (
        #     1000 - self.water_girl.get_position()[1]) / 1000

        # Combine rewards for both agents
        return fireboy_reward + watergirl_reward

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
