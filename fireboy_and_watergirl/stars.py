import pygame
from pygame.locals import *


class Stars:
    def __init__(self, star_location, player):
        """
        Initialize the Stars class.

        Args:
            star_locations (list of tuples): List of (x, y) coordinates for the stars.
        """
        self.star_location = star_location
        self.is_collected = False
        self._player = player
        self.load_images()
        self.make_rects()

    def load_images(self):
        """
        Load the star image.
        """
        # Load star image and make transparent
        if self._player == "fire":
            self.star_image = pygame.image.load(
                './fireboy_and_watergirl/data/board_textures/fireboy_star.png'
            )
        if self._player == "water":
            self.star_image = pygame.image.load(
                './fireboy_and_watergirl/data/board_textures/watergirl_star.png'
            )
        self.star_image.set_colorkey((255, 0, 255))

    def make_rects(self):
        """
        Create pygame rects for the stars.
        """
        self._rect = (
            pygame.Rect(self.star_location[0], self.star_location[1],
                        self.star_image.get_width(),
                        self.star_image.get_height())
        )

    def check_for_collection(self, players):
        """
        Check if any player has collected a star.

        Args:
            players (list of player objects): List of player objects (e.g., FireBoy, WaterGirl).
        """
        for i, star in enumerate(self._stars):
            if i in self.is_collected:
                continue  # Skip already collected stars
            for player in players:
                if star.colliderect(player.rect):
                    self.is_collected = True
                    # Mark star as collected

    def collect_star(self):
        """
        Collect a star by removing it from the list of stars.
        """
        self.is_collected = True
