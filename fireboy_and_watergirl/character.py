from pygame import Rect


class Character:
    def __init__(self, position):
        self.rect = Rect(position[0], position[1], 16, 16)
        # motion
        self.moving_right = False
        self.moving_left = False
        self.jumping = False
        self.y_velocity = 0
        self.can_jump = True
        # current state
        self._alive = True

    def calc_movement(self):
        """
        Set motion and physics constants and calculate movement
        """

        MOVEMENT_MULTIPLIER = 4
        # Motion constants
        LATERAL_SPEED = 4 * MOVEMENT_MULTIPLIER
        JUMP_SPEED = -5
        GRAVITY = 0.2 * MOVEMENT_MULTIPLIER
        TERMINAL_VELOCITY = 3 * MOVEMENT_MULTIPLIER

        # set initially to not moving
        self._movement = [0, 0]
        # calculate horizontal movement
        if self.moving_right:
            self._movement[0] = LATERAL_SPEED
        if self.moving_left:
            self._movement[0] = -LATERAL_SPEED

        # calculate vertical movement
        if self.jumping and self.can_jump:
            self.y_velocity = JUMP_SPEED
            self.jumping = False
        self._movement[1] += self.y_velocity
        self.y_velocity += GRAVITY
        # establish terminal velocity of 3px/frame
        if self.y_velocity > TERMINAL_VELOCITY:
            self.y_velocity = TERMINAL_VELOCITY

    def kill_player(self):
        """
        Kill the player by setting the alive status of the player to False
        """
        self._alive = False

    def get_movement(self):
        """
        Return a list containing movement of the character
        """
        return self._movement

    def is_dead(self):
        """
        Return a boolean that indicates if the player is alive or dead
        """
        return self._alive is False

    def get_type(self):
        """
        Return string that contains the character type (fire or water)
        """
        return self._type

    def get_position(self):
        """
        Return the current position of the character
        """
        return self.rect.x, self.rect.y


class FireBoy(Character):
    def __init__(self, position):
        # self.image = pygame.image.load(
        #     './fireboy_and_watergirl/data/player_images/fireboy.png')
        # self.side_image = pygame.image.load(
        #     './fireboy_and_watergirl/data/player_images/fireboy-side.png')
        self._type = "fire"
        super().__init__(position)


class WaterGirl(Character):
    def __init__(self, position):
        # self.image = pygame.image.load(
        #     './fireboy_and_watergirl/data/player_images/watergirl.png')
        # self.side_image = pygame.image.load(
        #     './fireboy_and_watergirl/data/player_images/watergirl-side.png')
        self._type = "water"
        super().__init__(position)
