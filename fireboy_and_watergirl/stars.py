import pygame
from pygame.locals import *


class Stars:
    def __init__(self, postion, player):
        """
        Initialize the Stars class.

        Args:
            star_locations (list of tuples): List of (x, y) coordinates for the stars.
        """
        self.postion = postion
        self.is_collected = False
        self._player = player
        self.load_images()
        self.make_rects()

        self.reward_given = False

    def load_images(self):
        """
        Load the star image.
        """
        # Load star image and make transparent
        image_paths = {
            "fire": './fireboy_and_watergirl/data/board_textures/fireboy_star.png',
            "water": './fireboy_and_watergirl/data/board_textures/watergirl_star.png'
        }
        self.star_image = pygame.image.load(
            image_paths.get(self._player, image_paths["fire"]))
        # self.star_image.set_colorkey((255, 0, 255))

    def make_rects(self):
        """
        Create pygame rects for the stars.
        """
        self._rect = (
            pygame.Rect(self.postion[0], self.postion[1],
                        self.star_image.get_width(),
                        self.star_image.get_height())
        )

    def get_position(self):
        """
        Return the position of the star.
        """
        return self.postion

    # def check_for_collection(self, players):
    #     """
    #     Check if any player has collected a star.

    #     Args:
    #         players (list of player objects): List of player objects (e.g., FireBoy, WaterGirl).
    #     """
    #     for i, star in enumerate(self._stars):
    #         if i in self.is_collected:
    #             continue  # Skip already collected stars
    #         for player in players:
    #             if star.colliderect(player.rect):
    #                 self.is_collected = True
    #                 # Mark star as collected

    def collect_star(self):
        """
        Collect a star by removing it from the list of stars.
        """
        self.is_collected = True
