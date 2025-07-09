from pygame import Rect


class Gate:
    def __init__(self, position, type):
        self.position = position
        self.type = type
        self.is_open = False

        self.make_rects()

    def make_rects(self):
        """
        Make pygame rects for gate and plates
        """

        self.rect = Rect(self.position[0], self.position[1], 16, 16)

    def is_open(self):
        """
        Return boolean that indicates if gate is open or closed
        """
        return self.is_open

    def get_type(self):
        """
        Return the type of the gate
        """
        return self.type
