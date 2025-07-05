import pygame
from pygame.locals import *


class Box:
    def __init__(self, position: list[2]):
        """
        Initialize the Box class.

        Args:
            position (tuple): (x, y) coordinates for the box.
        """
        self.position = position
        self.load_image()
        self.make_rect()
        self.is_being_pushed = False  # Flag to indicate if the box is being pushed

    def load_image(self):
        """
        Load the box image.
        """
        image_path = './fireboy_and_watergirl/data/board_textures/box.png'
        # self.box_image = pygame.image.load(image_path)

    def make_rect(self):
        """
        Create pygame rect for the box.
        """
        self._rect = pygame.Rect(
            self.position[0],
            self.position[1],
            16,
            16
        )

    def get_position(self):
        """
        Return the position of the box.
        """
        return self.position

    def set_position(self, new_position):
        """
        Set a new position for the box and update its rect.
        """
        self.position = new_position
        self._rect = pygame.Rect(
            self.position[0],
            self.position[1],
            16,
            16
        )

    def move(self, dx, dy):
        """
        Move the box by (dx, dy).
        """
        new_x = self.position[0] + dx
        new_y = self.position[1] + dy
        self.set_position((new_x, new_y))

    def check_collision(self, rect):
        """
        Check if the box collides with another rect (e.g., player).
        """
        return self._rect.colliderect(rect)
