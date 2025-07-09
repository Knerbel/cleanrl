from pygame import Rect


class Plate:
    def __init__(self, position, type):
        self._position = position
        self._type = type
        self._is_pressed = False
        self.reward_annealing = 1

        self.make_rects()

    def make_rects(self):
        """
        Make pygame rects for gate and plates
        """

        self.rect = Rect(self.position[0], self.position[1], 16, 16)

    def get_type(self):
        """
        Return the type of the plate
        """
        return self._type

    def get_position(self):
        return self._position
