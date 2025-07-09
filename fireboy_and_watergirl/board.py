# from fireboy_and_watergirl.rect import Rect
from pygame import Rect


class Board:
    def __init__(self, path):
        self.CHUNK_SIZE = 16
        self.load_map(path)
        self.make_solid_blocks()
        self.make_water_pools()
        self.make_lava_pools()
        self.make_goo_pools()

    def load_map(self, path):
        self._game_map = []
        with open(path) as f:
            for line in f:
                line = line.strip().split(',')  # convert string to list of str
                self._game_map.append(line)

    def make_solid_blocks(self):
        """
        Iterate through the map and make the walls and ground solid blocks
        which the player can collide with.
        """
        # create empty list to contain solid block rects
        self._solid_blocks = []
        for y, row in enumerate(self._game_map):
            for x, tile in enumerate(row):
                # if block is not air or a liquid
                if tile not in [' ', '0', '2', '3', '4', 'L', 'W', 'G', 'a', 'b', 'A', 'B']:
                    # create a 16 x 16 rect and add it to the list
                    self._solid_blocks.append(
                        Rect(x * self.CHUNK_SIZE, y * self.CHUNK_SIZE,
                             self.CHUNK_SIZE, self.CHUNK_SIZE)
                    )

    def update_doors_solid_state(self, doors_d__open: bool, doors_e__open: bool):
        """
        If doors_open is True, remove all 'D' tiles from _solid_blocks.
        If doors_open is False, ensure all 'D' tiles are in _solid_blocks.
        """
        # Remove all D tiles from _solid_blocks
        self._solid_blocks = [
            rect for rect in self._solid_blocks
            if not self._is_door_rect(rect)
        ]
        if not doors_d__open:
            # Add all D tiles back as solid blocks
            for y, row in enumerate(self._game_map):
                for x, tile in enumerate(row):
                    if tile == 'D':
                        rect = Rect(x * self.CHUNK_SIZE, y * self.CHUNK_SIZE,
                                    self.CHUNK_SIZE, self.CHUNK_SIZE)
                        if rect not in self._solid_blocks:
                            self._solid_blocks.append(rect)
        if not doors_e__open:
            # Add all E tiles back as solid blocks
            for y, row in enumerate(self._game_map):
                for x, tile in enumerate(row):
                    if tile == 'E':
                        rect = Rect(x * self.CHUNK_SIZE, y * self.CHUNK_SIZE,
                                    self.CHUNK_SIZE, self.CHUNK_SIZE)
                        if rect not in self._solid_blocks:
                            self._solid_blocks.append(rect)

    def _is_door_rect(self, rect):
        x = rect.x // self.CHUNK_SIZE
        y = rect.y // self.CHUNK_SIZE
        return self._game_map[y][x] == 'D' or self._game_map[y][x] == 'E'

    def get_level_data(self):
        """
        Return level data
        """
        return self._game_map

    def get_solid_blocks(self):
        return self._solid_blocks

    def make_lava_pools(self):
        self._lava_pools = []
        for y, row in enumerate(self._game_map):
            for x, tile in enumerate(row):
                # if number in game map represents lava
                if tile == "2" or tile == "L":
                    # add a 16x8 rect to the list
                    self._lava_pools.append(
                        Rect(x * self.CHUNK_SIZE, y * self.CHUNK_SIZE + self.CHUNK_SIZE / 2,
                             self.CHUNK_SIZE, self.CHUNK_SIZE / 2)
                    )

    def get_lava_pools(self):
        """
        Return list containing lava pool rects
        """
        return self._lava_pools

    def make_water_pools(self):
        """
        Create list containing water pool rects
        """
        # Create empty list to store water pool rects
        self._water_pools = []
        for y, row in enumerate(self._game_map):
            for x, tile in enumerate(row):
                # if number in game map represents water
                if tile == "3" or tile == "W":
                    # add a 16x8 rect to the list
                    self._water_pools.append(
                        Rect(x * self.CHUNK_SIZE, y * self.CHUNK_SIZE + self.CHUNK_SIZE / 2,
                             self.CHUNK_SIZE, self.CHUNK_SIZE / 2)
                    )

    def get_water_pools(self):
        return self._water_pools

    def make_goo_pools(self):
        self._goo_pools = []
        for y, row in enumerate(self._game_map):
            for x, tile in enumerate(row):
                # if number in game map represents goo
                if tile == "4" or tile == "G":
                    # add a 16x8 rect to the list
                    self._goo_pools.append(
                        Rect(x * self.CHUNK_SIZE, y * self.CHUNK_SIZE + self.CHUNK_SIZE / 2,
                             self.CHUNK_SIZE, self.CHUNK_SIZE / 2)
                    )

    def get_goo_pools(self):
        """
        Return list containing goo pool rects
        """
        return self._goo_pools

    def get_level_data(self):
        """
        Return level data
        """
        return self._game_map  # , self._solid_blocks, self._water_pools, self._lava_pools, self._goo_pools
