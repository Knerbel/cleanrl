from pygame import Rect


class Doors:
    def __init__(self, position):
        # Set doors' initial height and state
        self.position = position
        # self._player = player

        self.make_rects()
        self.reward_given = False
        self.player_at_door = False

    def make_rects(self):
        """
        Create a custom Rect for the door.
        """
        self._rect = (
            Rect(self.position[0], self.position[1],
                 16,
                 16)
        )

    def get_door(self):
        """
        Return a Rect containing the location and size of the door.
        """
        return self._rect

    def get_position(self):
        """
        Return the position of the door.
        """
        return self.position


class FireDoor(Doors):
    def __init__(self, position):
        self._player = "fire"
        super(position).__init__()


class WaterDoor(Doors):
    def __init__(self, position):
        self._player = "water"
        super(position).__init__()
