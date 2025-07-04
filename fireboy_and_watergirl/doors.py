from pygame import Rect
import pygame


class Doors:
    def __init__(self):
        # Set doors' initial height and state
        # self.postion = postion
        self.player_at_door = False
        # self._player = player

        self.door_width = 16
        self.door_height = 32
        self.load_images()
        self.make_rects()
        self.reward_given = False

    def load_images(self):
        """
        Load the star image.
        """
        # Load star image and make transparent
        self.star_image = pygame.image.load(
            './fireboy_and_watergirl/data/door_images/door_frame.png'
        )

    def make_rects(self):
        """
        Create a custom Rect for the door.
        """
        self._rect = (
            Rect(self.position[0], self.position[1],
                 self.star_image.get_width(),
                 self.star_image.get_height())
        )

    def get_door(self):
        """
        Return a Rect containing the location and size of the door.
        """
        return self._rect

    def is_door_open(self):
        """
        Return a boolean containing the status of the door.
        """
        return self._door_open

    def try_raise_door(self):
        """
        Try to raise the door if conditions are met.

        Raise the door if the player is at the door and the door is closed.
        """
        # Set door opening/closing speed
        DOOR_SPEED = 1.5
        door_x = self.position[0]
        door_y = self.position[1]
        # If there is a player at the door and the door isn't open yet
        if self.player_at_door and not self._door_open:
            # Move the door up
            self.position = (door_x, door_y - DOOR_SPEED)
            self.make_rects()  # Update the rect position
            # Update internal measure of door height
            self._height_raised += DOOR_SPEED
            # If the door has raised 31 pixels
            if self._height_raised >= 31:
                # Set the door to being fully raised
                self._door_open = True
        # If there is no player at the door and the door is fully open
        elif not self.player_at_door and self._height_raised > 0:
            # Move the door down
            self.position = (door_x, door_y + DOOR_SPEED)
            self.make_rects()  # Update the rect position
            # Update internal measure of door height
            self._height_raised -= DOOR_SPEED
            # Set the door as being not open
            self._door_open = False

    def get_position(self):
        """
        Return the position of the door.
        """
        return self.position


class FireDoor(Doors):
    def __init__(self, position):
        # CHUNK_SIZE = 16
        # Set door location as input door location
        self.position = position
        # Set door background location as the same as the door
        # self.background_location = position        self.load_images()

        # Since the frame is larger than the door, it has to be offset
        # self.frame_location = (
        #     position[0] - CHUNK_SIZE, position[1] - 2 * CHUNK_SIZE)
        self._player = "fire"
        super().__init__()


class WaterDoor(Doors):
    def __init__(self, position):
        # CHUNK_SIZE = 16
        # Set door location as input door location
        self.position = position
        # Set door background location as the same as the door
        # self.background_location = position
        # Since the frame is larger than the door, it has to be offset
        # self.frame_location = (
        #     position[0] - CHUNK_SIZE, position[1] - 2 * CHUNK_SIZE)
        self._player = "water"
        super().__init__()
