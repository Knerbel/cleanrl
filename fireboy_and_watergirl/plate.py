

import pygame


class Plate:
    def __init__(self, position, type):
        self._position = position
        self._type = type
        self._is_pressed = False
        self.make_rects()

    def make_rects(self):
        """
        Make pygame rect for the plate
        """
        x_cord = self._position[0]
        y_cord = self._position[1]
        self.rect = pygame.Rect(x_cord, y_cord, 16, 16)

    def get_type(self):
        """
        Return the type of the plate
        """
        return self._type

    def get_position(self):
        return self._position
