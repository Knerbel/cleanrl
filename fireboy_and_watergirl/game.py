
from fireboy_and_watergirl.board import Board
from fireboy_and_watergirl.character import Character
from fireboy_and_watergirl.doors import FireDoor, WaterDoor
from fireboy_and_watergirl.gate import Gate
from fireboy_and_watergirl.plate import Plate
from fireboy_and_watergirl.star import Star
from pygame import Rect


class Game:
    # game meta functions
    def __init__(self):
        """
        Initialize game.

        Create a public display that the user sees. Also create an internal
        display that only the game handles. The internal will be scaled to
        fit public display.
        """
        self.index = Game.increment_game_count()

        # create external pygame window
        # WINDOW_SIZE = (640, 480)
        # self.screen = pygame.display.set_mode(WINDOW_SIZE, pygame.RESIZABLE)
        # pygame.display.set_caption("Fire Boy and Water Girl")

        # create internal pygame window
        # CHUNK_SIZE = 16
        # DISPLAY_SIZE = (34 * CHUNK_SIZE, 25 * CHUNK_SIZE)
        # self.display = pygame.Surface(DISPLAY_SIZE)

    @staticmethod
    def increment_game_count():
        if not hasattr(Game, "_game_count"):
            Game._game_count = 0
        else:
            Game._game_count += 1
        return Game._game_count

    def draw_level_screen(self, level_select):
        """
        Draw level selection screen.

        Args:
            level_select::level_select class object
                A class object that contains the images for the level selection
                screen.
        """
        # display main level selection screen background
        self.display.blit(level_select.screen, (0, 0))

        # display the 5 level titles
        for level in range(5):
            # get image from level_select titles dictionary
            image = level_select.titles[level + 1]
            # center title in x direction
            title_x = (self.display.get_width() - image.get_width()) / 2
            # move titles down so that they don't overlap
            title_y = 50 * level + 100
            self.display.blit(image, (title_x, title_y))

        # display the characters on the left and right of level titles
        left_cords = (50, 150)
        right_cords = (430, 150)
        self.display.blit(level_select.left_player, left_cords)
        self.display.blit(level_select.right_player, right_cords)

    # def refresh_window(self):
    #     """
    #     Refresh and draw the game screen
    #     """
    #     new_window_size, center_cords = self.adjust_scale()
    #     # scale internal display to match window)
    #     # new_disp = pygame.transform.scale(self.display, new_window_size)
    #     # self.screen.blit(new_disp, center_cords)
    #     # pygame.display.update()

    # def adjust_scale(self):
    #     """
    #     Adjust internal screen for window scaling

    #     If the window size is changed, scale the game to the maximum amount
    #     while keeping the same aspect ratio. Also keep the game centered in the
    #     window.

    #     Returns:
    #         display_size::tuple (height, width)
    #             The updated height and width of the internal game display
    #         cords::tuple (x_cord, y_cord)
    #             The coordinates of the upper left corner of the internal game
    #             display so that when it is blit onto window, it is centered.
    #     """
    #     window_size = self.screen.get_size()

    #     # if window is longer than aspect ratio
    #     if window_size[0] / window_size[1] >= 1.5:
    #         display_size = (int(1.5 * window_size[1]), window_size[1])
    #     # if window is taller than aspect ratio
    #     else:
    #         display_size = (window_size[0], int(.75 * window_size[0]))
    #     # find cords so that display is centered
    #     cords = ((window_size[0] - display_size[0]) / 2,
    #              (window_size[1] - display_size[1]) / 2)

    #     return display_size, cords

    # game mechanics

    # def draw_level_background(self, board: Board):
    #     """
    #     Draw the background of the level.

    #     Args:
    #         board::board class object
    #             board class object that contains information on chunk images
    #             and their locations
    #     """
    #     # self.display.blit(board.get_background(), (0, 0))

    # def draw_board(self, board: Board):
    #     """
    #     Draw the board.

    #     Args:
    #         board::board class object
    #             board class object that contains information on chunk images
    #             and their locations
    #     """
    #     # draw the full background
    #     board_textures = board.get_board_textures()
    #     # draw the solid blocks and liquids
    #     for y, row in enumerate(board.get_game_map()):
    #         for x, tile in enumerate(row):
    #             if tile != "0" and tile != " ":
    #                 self.display.blit(
    #                     board_textures[f"{tile}"], (x * 16, y * 16)
    #                 )

    # def draw_gates(self, gates):
    #     """
    #     Draw gates and buttons.

    #     Args:
    #         gates::[gate object, ...]
    #             A list of gate objects with image and location information.
    #     """
    #     # for gate in gates:
    #     #     # display gate
    #     #     self.display.blit(gate.gate_image, gate.gate_location)

    #     #     for location in gate.plate_locations:
    #     #         # display plate location
    #     #         self.display.blit(gate.plate_image, location)

    # def draw_doors(self, doors):
    #     """
    #     Draw doors

    #     Args:
    #         doors::[door object, door object]
    #             A list of door class objects containing image and location
    #             information of door, door background, and fame.
    #     """
    #     # for door in doors:
    #     #     # draw door background
    #     #     self.display.blit(door.door_background, door.background_location)
    #     #     # draw door
    #     #     self.display.blit(door.door_image, door.door_location)
    #     #     # draw door frame
    #     #     self.display.blit(door.frame_image, door.frame_location)

    # def draw_player(self, players: list[Character]):
    #     """
    #     Draw the player.

    #     If the player is moving right or left, draw the player as facing that
    #     direction.

    #     Args:
    #         player::[player object, player object]
    #             a list of player objects that contains movement data as well as
    #             different images, one for each direction it can face.
    #     """
    #     # for player in players:
    #     #     if player.moving_right:
    #     #         player_image = player.side_image
    #     #     elif player.moving_left:
    #     #         player_image = pygame.transform.flip(
    #     #             player.side_image, True, False)
    #     #     else:
    #     #         player_image = player.image
    #     #     player_image.set_colorkey((255, 0, 255))
    #     #     self.display.blit(player_image, (player.rect.x, player.rect.y))

    # def draw_stars(self, stars):
    #     """
    #     Draw the stars on the screen.

    #     Args:
    #         screen (pygame.Surface): The game screen to draw on.
    #     """
    #     # for i, star in enumerate(stars):
    #     #     if not star.is_collected:  # Only draw uncollected stars
    #     #         self.display.blit(
    #     #             star.star_image, (star.star_location[0], star.star_location[1]))

    def move_player(self, board: Board, doors: list[FireDoor | WaterDoor], players: list[Character]):
        """
        Move player

        This function primarily deals with collisions. The function moves the
        player then checks for collisions with the board and gates. It then
        adjusts the location of the player to account for these collisions.

        Args:
            board::board class object
                board class object that contains information on where solid
                blocks are.
            doors::[door object, ...]
                A list of door class objects that contains information on where
                the solid aspects of the door are.
            players::[player object, player object]
                A list of player objects that contain information on movement
                and position.
        """
        # Get the level boundaries
        level_width = len(board.get_level_data()[0]) * board.CHUNK_SIZE
        level_height = len(board.get_level_data()) * board.CHUNK_SIZE

        for player in players:
            player.calc_movement()
            movement = player.get_movement()

            collide_blocks = board.get_solid_blocks()
            collision_types = {'top': False,
                               'bottom': False, 'right': False, 'left': False}

            # Try moving the player horizontally
            player.rect.x += movement[0]
            hit_list = self.collision_test(player.rect, collide_blocks)
            for tile in hit_list:
                if movement[0] > 0:  # Moving right
                    player.rect.x = tile.left - player.rect.width
                    collision_types['right'] = True
                elif movement[0] < 0:  # Moving left
                    player.rect.x = tile.right
                    collision_types['left'] = True

            # Clamp the player's horizontal position within the level boundaries
            if player.rect.left < 0:
                player.rect.x = 0
            if player.rect.right > level_width:
                player.rect.x = level_width - player.rect.width

            # Try moving the player vertically
            player.rect.y += movement[1]
            hit_list = self.collision_test(player.rect, collide_blocks)
            for tile in hit_list:
                if movement[1] > 0:  # Moving down
                    player.rect.y = tile.top - player.rect.height
                    collision_types['bottom'] = True
                elif movement[1] < 0:  # Moving up
                    player.rect.y = tile.bottom
                    collision_types['top'] = True

            # Clamp the player's vertical position within the level boundaries
            if player.rect.top < 0:
                player.rect.y = 0
            if player.rect.bottom > level_height:
                player.rect.y = level_height - player.rect.height

            # Handle collisions
            if collision_types['bottom']:
                player.y_velocity = 0
                player.can_jump = True
            else:
                player.can_jump = False

            if collision_types['top']:
                player.y_velocity = 0

    def check_for_death(self, board: Board, players: list[Character]):
        """
        Check to see if player has faden in pool that kills them or if they are
        crushed by a gate.

        If a fire type player collides with a water pool, they die. Likewise,
        if a water type player collides with a lava pool, they die. If either
        type of player collides with a goo pool, they die.
        Args:
            board::board class object
                class object with information on board layout
            gates::gate class object
                class object with information on gate location and state
            players::[player object, player object]
                A list of player class objects.
        """

        # return False

        for player in players:
            # if the player is water_girl
            if player.get_type() == "water":
                # see if she collides with lava
                is_killed = self.collision_test(
                    player.rect, board.get_lava_pools())
            # if the player is fire_boy
            if player.get_type() == "fire":
                # see if he collides wit water
                is_killed = self.collision_test(
                    player.rect, board.get_water_pools())
            # see if either collide with goo
            is_killed += self.collision_test(player.rect,
                                             board.get_goo_pools())

            # if the is_killed list is longer than 0, kill player

            if is_killed:
                player.kill_player()
                return True
        return False

    def check_for_star_collected(self, stars: list[Star], players: list[Character]):
        """
        Check if any player has collected a star.

        Args:
            stars::[star object, ...]
                A list of star objects containing information on their location.
            players::[player object, player object]
                A list of player objects containing information on their location.
        """
        for player in players:
            for star in stars:
                # Check if the player collides with the star
                if player.rect.colliderect(star._rect):
                    if player.get_type() == star._player:
                        # Mark the star as collected
                        star.collect_star()

    def check_for_plates_press(self, plates: list[Plate], players: list[Character]):
        """
        Check to see if either player is standing one tile above a plate.

        Args:
            plates::[plate object, ...]
                A list of plate class objects containing information on location
                of the plate.
            players::[player object, player object]
                A list of player class objects containing information on their
                location.
        """
        CHUNK_SIZE = 16  # Make sure this matches your board's chunk size
        for plate in plates:
            plate._is_pressed = False  # Assume not pressed
            plate_x = plate.rect.x // CHUNK_SIZE
            plate_y = plate.rect.y // CHUNK_SIZE
            for player in players:
                player_x = player.rect.x // CHUNK_SIZE
                player_y = player.rect.y // CHUNK_SIZE
                # Check if player is exactly one tile above the plate
                if player_x == plate_x and player_y == plate_y - 1:
                    plate._is_pressed = True
                    break  # No need to check other players for this plate

    def check_for_at_door(self, doors: FireDoor | WaterDoor, players: list[Character]):
        """
        Check to see if a player is at the door.

        Args:
            door::door class object
                A door object containing information on its location and state
            player::player class object
                A player object containing information on its location
        """
        # check to see if the player is at the door
        for player in players:
            for door in doors:
                if player.rect.colliderect(door._rect):
                    if player.get_type() == door._player:
                        door.player_at_door = True

    # def check_for_door_open(self, door: FireDoor | WaterDoor, player: Character):
    #     """
    #     Check to see if a player is at the door.

    #     Args:
    #         door::door class object
    #             A door object containing information on its location and state
    #         player::player class object
    #             A player object containing information on its location
    #     """
    #     # check to see if the player is at the door
    #     door_collision = self.collision_test(player.rect, [door.get_door()])
    #     # if the collision list is greater than zero, player is at door
    #     if door_collision:
    #         door.player_at_door = True
    #     # otherwise, player is not at door
    #     else:
    #         door.player_at_door = False
    #     # attempt to raise door. If nobody is at door, try to close the door
    #     door.try_raise_door()

    @staticmethod
    def level_is_done(doors: list[FireDoor | WaterDoor]):
        """
        Check to see if the level is complete

        Args:
            doors::[door object, door object]
                A list of door class objects that contain information on their
                state.
        Return:
            is_win::bool
                Return True if level is complete, or False if it is not
        """
        # by default set win to true
        is_win = True
        for door in doors:
            # if either door are not open, set win to False
            if not door.is_door_open():
                is_win = False
        return is_win

    @staticmethod
    def collision_test(rect: Rect, tiles: list[Rect]) -> list[Rect]:
        """
        Create a list of tiles a pygame rect is colliding with.

        Args:
            rect::pygame.rect
                A pygame rect that may be colliding with other rects.
            tiles::[rect, rect, rect]
                A list of pygame rects. The function checks to see if the
                argument "rect" collides with any of these "tiles".
        Returns:
            hit_list::list
                A list of all "tiles" that the argument rect is colliding with.
                If an empty list is returned, the rect is not colliding with
                any tile.
        """
        hit_list = []
        for tile in tiles:
            if rect.colliderect(tile):
                hit_list.append(tile)
        return hit_list
