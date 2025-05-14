from fireboy_and_watergirl.rect import Rect


class Doors:
    def __init__(self):
        # Set doors' initial height and state
        self._player = None
        self.player_at_door = False
        self._height_raised = 0
        self._door_open = False

        # Initialize door position and dimensions
        self.position = (0, 0)
        self.door_width = 16
        self.door_height = 32
        self.reward_given = False
        self._rect = None
        self.make_rects()

    def make_rects(self):
        """
        Create a custom Rect for the door.
        """
        x_cord = self.position[0]
        y_cord = self.position[1]
        self._rect = Rect(x_cord, y_cord, self.door_width, self.door_height)

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
        CHUNK_SIZE = 16
        # Set door location as input door location
        self.position = position
        # Set door background location as the same as the door
        self.background_location = position
        # Since the frame is larger than the door, it has to be offset
        self.frame_location = (
            position[0] - CHUNK_SIZE, position[1] - 2 * CHUNK_SIZE)
        self._player = "fire"
        super().__init__()


class WaterDoor(Doors):
    def __init__(self, position):
        CHUNK_SIZE = 16
        # Set door location as input door location
        self.position = position
        # Set door background location as the same as the door
        self.background_location = position
        # Since the frame is larger than the door, it has to be offset
        self.frame_location = (
            position[0] - CHUNK_SIZE, position[1] - 2 * CHUNK_SIZE)
        self._player = "water"
        super().__init__()
