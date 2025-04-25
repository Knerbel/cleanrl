import gymnasium as gym
from gymnasium import spaces
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
        # Actions: 0 = Fireboy Left, 1 = Fireboy Right, 2 = Fireboy Up, 3 = Fireboy Still,
        #          4 = Watergirl Left, 5 = Watergirl Right, 6 = Watergirl Up, 7 = Watergirl Still
        self.action_space = spaces.Discrete(8)

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
        # self.controller = GeneralController()

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

        board_size = 34 * 25
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            # 6 features + flattened board + 4 player positions + 3 features per star
            shape=(4 + board_size + len(self.stars) * 3,),
            dtype=np.float32
        )

    def _load_level(self):
        """
        Load the level data and initialize game components.
        """
        if self.level == "level1":
            self.board = Board('./fireboy_and_watergirl/data/level1.txt')
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

            # Flatten the level 1 board into a 1D array
            level_data = self.board.get_level_data()  # Assuming this returns a 2D array
            # Map tile types to numeric values
            tile_mapping = {'S': 1, 'L': 2, 'W': 3,
                            'G': 4, ' ': 5, 'w': 6, 'f': 7}

            # Convert the 2D array of tiles to a 2D array of numbers
            level_data = [[tile_mapping[tile] for tile in row]
                          for row in level_data]
            self.flattened_level = np.array(level_data, dtype=np.float32).flatten(
            ) if level_data is not None else np.array([], dtype=np.float32)

        # Add more levels as needed

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

        # Compute reward
        reward = self._compute_reward()

        # Check if the game is done
        self.done = self._check_done()

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

    def _get_state(self):
        """
        Return the current state of the game as a feature vector.
        """
        # fireboy_pos = self.fire_boy.get_position()  # (x, y)
        # watergirl_pos = self.water_girl.get_position()  # (x, y)
        # gate_status = int(all(gate.is_open()
        #                   for gate in self.gates))  # 1 if all gates are open
        # door_status = int(
        #     self.doors[0].is_door_open() and self.doors[1].is_door_open()
        # )  # 1 if both doors are reached

        # # Combine all features into a single state vector
        # return np.array([
        #     fireboy_pos[0], fireboy_pos[1],
        #     watergirl_pos[0], watergirl_pos[1],
        #     gate_status, door_status
        # ], dtype=np.float32)

        # Get the positions of Fireboy and Watergirl
        fireboy_pos = self.fire_boy.get_position()  # (x, y)
        watergirl_pos = self.water_girl.get_position()  # (x, y)

        # Get the status of the gates (1 if all gates are open, 0 otherwise)
        gate_status = int(all(gate.is_open() for gate in self.gates))

        # Get the status of the doors (1 if both doors are open, 0 otherwise)
        door_status = int(
            self.doors[0].is_door_open() and self.doors[1].is_door_open()
        )

        # Flatten the level 1 board into a 1D array
        # level_data = self.board.get_level_data()  # Assuming this returns a 2D array
        # Map tile types to numeric values
        # tile_mapping = {'S': 1, 'L': 2, 'W': 3, 'G': 4, ' ': 5}

        # # Ignore the borders (right, top, left, bottom) of the level data
        # level_data = [row for row in level_data]

        # # Convert the 2D array of tiles to a 2D array of numbers
        # level_data = [[tile_mapping[tile] for tile in row]
        #               for row in level_data]
        # flattened_level = np.array(level_data, dtype=np.float32).flatten(
        # ) if level_data is not None else np.array([], dtype=np.float32)

        # print(flattened_level)

        # Combine all features into a single state vector
        star_features = []
        for star in self.stars:
            star_features.extend(
                [star.star_location[0], star.star_location[1], int(star.is_collected)])

        state = np.concatenate((
            np.array([fireboy_pos[0],
                      fireboy_pos[1],
                      watergirl_pos[0],
                      watergirl_pos[1],
                      ],
                     dtype=np.float32),
            self.flattened_level,
            np.array(star_features, dtype=np.float32)
        ))

        return state

    def _apply_action(self, action):
        """
        Update the game state based on the action.
        """
        # Map actions to character movements

        # self.game.move_player(self.board, self.gates, [FireBoy, WaterGirl])
        if action == 0:
            self.fire_boy.moving_left = True
            self.fire_boy.moving_right = False
            self.fire_boy.jumping = False
        elif action == 1:
            self.fire_boy.moving_left = False
            self.fire_boy.moving_right = True
            self.fire_boy.jumping = False
        elif action == 2:
            if self.fire_boy.air_timer < 6:
                self.fire_boy.moving_left = False
                self.fire_boy.moving_right = False
                self.fire_boy.jumping = True
        elif action == 3:
            self.fire_boy.jumping = False
            self.fire_boy.moving_left = False
            self.fire_boy.moving_right = False
        elif action == 4:
            self.water_girl.moving_left = True
            self.water_girl.moving_right = False
            self.jumping = False
        elif action == 5:
            self.water_girl.moving_left = False
            self.water_girl.moving_right = True
            self.water_girl.jumping = False
        elif action == 6:
            if self.water_girl.air_timer < 6:
                self.water_girl.moving_left = False
                self.water_girl.moving_right = False
                self.water_girl.jumping = True
        elif action == 7:
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


# Register the environment
register(
    id="FireboyAndWatergirl-ppo-v0",  # Unique ID for the environment
    # Path to the environment class
    entry_point="cleanrl.fireboy_and_wategirl_ppo:FireboyAndWatergirlEnv",
)
