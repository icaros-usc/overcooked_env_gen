import os
import pygame
import time
import numpy as np
import overcooked_ai_py
from overcooked_ai_py import ASSETS_DIR, PCG_EXP_IMAGE_DIR
from overcooked_ai_py.mdp.actions import Action, Direction

pygame.init()

INFO_PANEL_HEIGHT = 0  #60  # height of the game info panel
INFO_PANEL_COLOR = (230, 180, 83)  # some sort of yellow
SPRITE_LENGTH = 50  # length of each sprite square
TERRAIN_DIR = 'terrain'
CHEF_DIR = 'chefs'
OBJECT_DIR = 'objects'
FONTS_DIR = 'fonts'
ARIAL_FONT = os.path.join(ASSETS_DIR, FONTS_DIR, 'arial.ttf')
TEXT_SIZE = 25

TERRAIN_TO_IMG = {
    ' ': os.path.join(ASSETS_DIR, TERRAIN_DIR, 'floor.png'),
    'X': os.path.join(ASSETS_DIR, TERRAIN_DIR, 'counter.png'),
    'P': os.path.join(ASSETS_DIR, TERRAIN_DIR, 'pot.png'),
    'O': os.path.join(ASSETS_DIR, TERRAIN_DIR, 'onions.png'),
    'T': os.path.join(ASSETS_DIR, TERRAIN_DIR, 'tomatoes.png'),
    'D': os.path.join(ASSETS_DIR, TERRAIN_DIR, 'dishes.png'),
    'S': os.path.join(ASSETS_DIR, TERRAIN_DIR, 'serve.png'),
}

PLAYER_HAT_COLOR = {
    0: 'greenhat',
    1: 'bluehat',
}

PLAYER_ARROW_COLOR = {0: (0, 255, 0, 128), 1: (0, 0, 255, 128)}

PLAYER_ARROW_ORIENTATION = {
    Direction.DIRECTION_TO_STRING[Direction.NORTH]:
    ((15, 300), (35, 300), (35, 100), (50, 100), (25, 0), (0, 100), (15, 100)),
    Direction.DIRECTION_TO_STRING[Direction.SOUTH]:
    ((15, 0), (35, 0), (35, 200), (50, 200), (25, 300), (0, 200), (15, 200)),
    Direction.DIRECTION_TO_STRING[Direction.EAST]:
    ((0, 15), (0, 35), (200, 35), (200, 50), (300, 25), (200, 0), (200, 15)),
    Direction.DIRECTION_TO_STRING[Direction.WEST]:
    ((300, 15), (300, 35), (100, 35), (100, 50), (0, 25), (100, 0), (100, 15)),
}

PLAYER_ARROW_POS_SHIFT = {
    Direction.DIRECTION_TO_STRING[Direction.NORTH]:
    ((1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0)),
    Direction.DIRECTION_TO_STRING[Direction.SOUTH]:
    ((1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0)),
    Direction.DIRECTION_TO_STRING[Direction.EAST]:
    ((0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)),
    Direction.DIRECTION_TO_STRING[Direction.WEST]:
    ((0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)),
}


def get_curr_pos(x, y, mode="human"):
    """
    Returns pygame.Rect object that specifies the position

    Args:
        x, y: position of the terrain in the terrain matrix
        mode: mode of rendering
    """
    if mode == "full":
        return pygame.Rect(
            x * SPRITE_LENGTH,
            y * SPRITE_LENGTH + INFO_PANEL_HEIGHT,
            SPRITE_LENGTH,
            SPRITE_LENGTH,
        )

    else:
        return pygame.Rect(
            x * SPRITE_LENGTH,
            y * SPRITE_LENGTH,
            SPRITE_LENGTH,
            SPRITE_LENGTH,
        )


def get_text_sprite(show_str):
    """
    Returns pygame.Surface object to show the text

    Args:
        show_str(string): The text to show
    """
    font = pygame.font.Font(ARIAL_FONT, TEXT_SIZE)
    text_surface = font.render(show_str, True, (255, 0, 0))
    return text_surface


def load_image(path):
    """
    Returns loaded pygame.Surface object from file path

    Args:
        path(string): file path to the image file
    """
    obj = pygame.image.load(path).convert()
    obj.set_colorkey((255, 255, 255))
    return pygame.transform.scale(obj, (SPRITE_LENGTH, SPRITE_LENGTH))


def blit_terrain(x, y, terrain_mtx, viewer, mode="human"):
    """
    Helper function to blit given position to specified terrain

    Args:
        x, y: position of the terrain in the terrain matrix
        terrain_mtx: terrain matrix
        viewer: pygame surface that displays the game
    """
    curr_pos = get_curr_pos(x, y, mode)
    # render the terrain
    terrain = terrain_mtx[y][x]
    terrain_pgobj = load_image(TERRAIN_TO_IMG[terrain])
    viewer.blit(terrain_pgobj, curr_pos)


def get_player_sprite(player, player_index):
    """
    Returns loaded image of player(aka chef), the player's hat, and the color of the array to draw on top of the player

    Args:
        player(PlayerState)
        player_index(int)
    """
    orientation_str = get_orientation_str(player)

    player_img_path = ""
    hat_color = PLAYER_HAT_COLOR[player_index]
    hat_img_path = os.path.join(
        ASSETS_DIR, CHEF_DIR,
        '%s-%s.png' % (orientation_str, PLAYER_HAT_COLOR[player_index]))

    player_object = player.held_object
    # player holding object
    if player_object:
        # player holding soup
        obj_name = player_object.name
        if obj_name == 'soup':
            soup_type = player_object.state[0]
            player_img_path = os.path.join(
                ASSETS_DIR, CHEF_DIR,
                '%s-soup-%s.png' % (orientation_str, soup_type))

        # player holding non-soup
        else:
            player_img_path = os.path.join(
                ASSETS_DIR, CHEF_DIR,
                '%s-%s.png' % (orientation_str, obj_name))

    # player not holding object
    else:
        player_img_path = os.path.join(ASSETS_DIR, CHEF_DIR,
                                       '%s.png' % orientation_str)

    return load_image(player_img_path), load_image(hat_img_path)


def get_object_sprite(obj, on_pot=False):
    """
    Returns loaded image of object

    Args:
        obj(ObjectState)
        on_pot(boolean): whether the object lies on a pot
    """
    obj_name = obj.name

    if not on_pot:
        if obj_name == 'soup':
            soup_type = obj.state[0]
            obj_img_path = os.path.join(ASSETS_DIR, OBJECT_DIR,
                                        'soup-%s-dish.png' % soup_type)
        else:
            obj_img_path = os.path.join(ASSETS_DIR, OBJECT_DIR,
                                        '%s.png' % obj_name)
    else:
        soup_type, num_items, cook_time = obj.state
        obj_img_path = os.path.join(
            ASSETS_DIR, OBJECT_DIR,
            'soup-%s-%d-cooking.png' % (soup_type, num_items))
    return load_image(obj_img_path)


def draw_arrow(window, player, player_index, pos, time_left):
    """
    Draw an arrow indicating orientation of the player
    """
    shift = 10.0
    orientation_str = get_orientation_str(player)
    arrow_orientation = PLAYER_ARROW_ORIENTATION[orientation_str]
    arrow_position = [[j * shift * time_left for j in i]
                      for i in PLAYER_ARROW_POS_SHIFT[orientation_str]]
    arrow_orientation = np.add(np.array(arrow_orientation),
                               arrow_position).tolist()
    arrow_color = PLAYER_ARROW_COLOR[player_index]

    arrow = pygame.Surface((300, 300)).convert()

    pygame.draw.polygon(arrow, arrow_color, arrow_orientation)
    arrow.set_colorkey((0, 0, 0))

    arrow = pygame.transform.scale(arrow, (SPRITE_LENGTH, SPRITE_LENGTH))
    window.blit(arrow, pos)
    # tmp = input()


def get_orientation_str(player):
    orientation = player.orientation
    # make sure the orientation exists
    assert orientation in Direction.ALL_DIRECTIONS

    orientation_str = Direction.DIRECTION_TO_STRING[orientation]
    return orientation_str


def render_from_grid(lvl_grid, log_dir, filename):
    """
    Render a single frame of game from grid level.
    This function is used for visualization the levels generated which
    are possibily broken or invalid. It also does not take the orientation
    of the players into account. So this method should not be used for
    actual game rendering.
    """
    width = len(lvl_grid[0])
    height = len(lvl_grid)
    window_size = width * SPRITE_LENGTH, height * SPRITE_LENGTH
    viewer = pygame.display.set_mode(window_size)
    viewer.fill((255, 255, 255))
    for y, terrain_row in enumerate(lvl_grid):
        for x, terrain in enumerate(terrain_row):
            curr_pos = get_curr_pos(x, y)

            # render player
            if str.isdigit(terrain):
                player = overcooked_ai_py.mdp.overcooked_mdp.PlayerState(
                    (x, y), Direction.SOUTH)
                player_idx = int(terrain)
                player_pgobj, player_hat_pgobj = get_player_sprite(
                    player, player_idx - 1)

                # render floor as background
                terrain_pgobj = load_image(TERRAIN_TO_IMG[" "])
                viewer.blit(terrain_pgobj, curr_pos)

                # then render the player
                viewer.blit(player_pgobj, curr_pos)
                viewer.blit(player_hat_pgobj, curr_pos)

            # render terrain
            else:
                terrain_pgobj = load_image(TERRAIN_TO_IMG[terrain])
                viewer.blit(terrain_pgobj, curr_pos)

    pygame.display.update()

    # save image
    pygame.image.save(viewer, os.path.join(log_dir, filename))


def render_game_info_panel(window, game_window_size, num_orders_remaining,
                           time_passed):
    #<<<<<<< HEAD
    #    pass
    # game_window_width, game_window_height = game_window_size

    # # get panel rect
    # panel_rect = pygame.Rect(0, 0, game_window_width,
    #                          INFO_PANEL_HEIGHT)

    # # fill with background color
    # window.fill(INFO_PANEL_COLOR, rect=panel_rect)

    # # update num orders left
    # if num_orders_remaining == np.inf:
    #     num_orders_remaining = "inf"
    # num_order_t_surface = get_text_sprite(
    #     f"Number of orders left: {num_orders_remaining}")
    # num_order_text_pos = num_order_t_surface.get_rect()
    # num_order_text_pos.topleft = panel_rect.topleft
    # window.blit(num_order_t_surface, num_order_text_pos)

    # # update time passed
    # t_surface = get_text_sprite("Time passed: %3d s" % time_passed)
    # time_passed_text_pos = t_surface.get_rect()
    # _, num_order_txt_height = num_order_t_surface.get_size()
    # time_passed_text_pos.y = num_order_text_pos.y + num_order_txt_height
    # window.blit(t_surface, time_passed_text_pos)
    #=======
    game_window_width, game_window_height = game_window_size

    # get panel rect
    panel_rect = pygame.Rect(0, 0, game_window_width, INFO_PANEL_HEIGHT)

    # fill with background color
    window.fill(INFO_PANEL_COLOR, rect=panel_rect)

    # update num orders left
    if num_orders_remaining == np.inf:
        num_orders_remaining = "inf"
    num_order_t_surface = get_text_sprite(
        f"Number of orders left: {num_orders_remaining}")
    num_order_text_pos = num_order_t_surface.get_rect()
    num_order_text_pos.topleft = panel_rect.topleft
    window.blit(num_order_t_surface, num_order_text_pos)

    # update time passed
    t_surface = get_text_sprite("Time passed: %3d s" % time_passed)
    time_passed_text_pos = t_surface.get_rect()
    _, num_order_txt_height = num_order_t_surface.get_size()
    time_passed_text_pos.y = num_order_text_pos.y + num_order_txt_height
    window.blit(t_surface, time_passed_text_pos)


#>>>>>>> bce3ff4b5f40e334f942ddc27276ace0cdea63ea
