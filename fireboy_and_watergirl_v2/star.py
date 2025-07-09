from pygame import Rect


class Star:
    def __init__(self, position, player):
        """
        Initialize the Stars class.

        Args:
            star_locations (list of tuples): List of (x, y) coordinates for the stars.
        """
        self.position = position
        self.is_collected = False
        self._player = player
        self.make_rects()

        self.reward_given = False

    def make_rects(self):
        """
        Make pygame rects for gate and plates
        """
        self.rect = Rect(self.position[0], self.position[1], 16, 16)

    def get_position(self):
        """
        Return the position of the star.
        """
        return self.position

    def collect_star(self):
        """
        Collect a star by removing it from the list of stars.
        """
        self.is_collected = True
