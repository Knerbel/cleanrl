"""
Main file for Fire Boy and Water Girl game.
"""

# import pygame and other needed libraries
import sys
import pygame
from pygame.locals import *

# import classes
from game import Game
from board import Board
from character import FireBoy, WaterGirl
from controller import ArrowsController, WASDController, GeneralController
from gates import Gates
from doors import FireDoor, WaterDoor
# from level_select import LevelSelect


def main():
    pygame.init()
    controller = GeneralController()
    game = Game()
    run_simulation(game, controller)
    # show_intro_screen(game, controller)


# def show_intro_screen(game, controller):
#     intro_screen = pygame.image.load('data/screens/intro_screen.png')
#     game.display.blit(intro_screen, (0, 0))
#     while True:
#         game.refresh_window()
#         if controller.press_key(pygame.event.get(), K_RETURN):
#             print('Exiting')
#             # show_level_screen(game, controller)


# def show_level_screen(game, controller):
#     level_select = LevelSelect()
#     level = game.user_select_level(level_select, controller)
#     run_game(game, controller, level)


# def show_win_screen(game, controller):
#     win_screen = pygame.image.load('data/screens/win_screen.png')
#     win_screen.set_colorkey((255, 0, 255))
#     game.display.blit(win_screen, (0, 0))

#     while True:
#         game.refresh_window()
#         if controller.press_key(pygame.event.get(), K_RETURN):
#             show_level_screen(game, controller)


# def show_death_screen(game, controller, level):
#     death_screen = pygame.image.load('data/screens/death_screen.png')
#     death_screen.set_colorkey((255, 0, 255))
#     game.display.blit(death_screen, (0, 0))
#     while True:
#         game.refresh_window()
#         events = pygame.event.get()
#         if controller.press_key(events, K_RETURN):
#             run_game(game, controller, level)
#         if controller.press_key(events, K_ESCAPE):
#             show_level_screen(game, controller)


def run_simulation(game, controller):
    while True:
        run_game(game, controller)


def run_game(game, controller, level="level1"):
    # load level data
    if level == "level1":
        board = Board('data/level1.txt')
        gate_location = (285, 128)
        plate_locations = [(190, 168), (390, 168)]
        gate = Gates(gate_location, plate_locations)
        gates = [gate]

        fire_door_location = (64, 48)
        fire_door = FireDoor(fire_door_location)
        water_door_location = (128, 48)
        water_door = WaterDoor(water_door_location)
        doors = [fire_door, water_door]

        fire_boy_location = (16, 336)
        fire_boy = FireBoy(fire_boy_location)
        water_girl_location = (35, 336)
        water_girl = WaterGirl(water_girl_location)

    if level == "level2":
        board = Board('data/level2.txt')
        gates = []

        fire_door_location = (390, 48)
        fire_door = FireDoor(fire_door_location)
        water_door_location = (330, 48)
        water_door = WaterDoor(water_door_location)
        doors = [fire_door, water_door]

        fire_boy_location = (16, 336)
        fire_boy = FireBoy(fire_boy_location)
        water_girl_location = (35, 336)
        water_girl = WaterGirl(water_girl_location)

    if level == "level3":
        board = Board('data/level3.txt')
        gates = []

        fire_door_location = (5 * 16, 4 * 16)
        fire_door = FireDoor(fire_door_location)
        water_door_location = (28 * 16, 4 * 16)
        water_door = WaterDoor(water_door_location)
        doors = [fire_door, water_door]

        fire_boy_location = (28 * 16, 4 * 16)
        fire_boy = FireBoy(fire_boy_location)
        water_girl_location = (5 * 16, 4 * 16)
        water_girl = WaterGirl(water_girl_location)

    # initialize needed classes

    arrows_controller = ArrowsController()
    wasd_controller = WASDController()

    clock = pygame.time.Clock()

    frames_survived = 0
    time_survived = pygame.time.get_ticks()
    # main game loop
    while True:
        # pygame management
        clock.tick(0)
        events = pygame.event.get()

        # draw features of level
        game.draw_level_background(board)
        game.draw_board(board)
        if gates:
            game.draw_gates(gates)
        game.draw_doors(doors)
        game.draw_player([fire_boy, water_girl])

        # move player
        arrows_controller.control_player(events, fire_boy)
        wasd_controller.control_player(events, water_girl)

        game.move_player(board, gates, [fire_boy, water_girl])

        # check for player at special location
        game.check_for_death(board, [fire_boy, water_girl])

        game.check_for_gate_press(gates, [fire_boy, water_girl])

        game.check_for_door_open(fire_door, fire_boy)
        game.check_for_door_open(water_door, water_girl)

        # refresh window
        game.refresh_window()

        # special events
        if water_girl.is_dead() or fire_boy.is_dead():
            delta = pygame.time.get_ticks() - time_survived
            print(frames_survived/delta)
            # print(frames_survived)
            return
            # show_death_screen(game, controller, level)

        if game.level_is_done(doors):
            delta = pygame.time.get_ticks() - time_survived
            print(frames_survived/delta)
            return
            # show_win_screen(game, controller)

        # if controller.press_key(events, K_ESCAPE):
        #     show_level_screen(game, controller)

        # close window is player clicks on [x]
        for event in events:
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
        frames_survived += 1


if __name__ == '__main__':
    main()
