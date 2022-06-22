import os
from gym.envs.registration import register
from overcooked_ai_py.utils import load_dict_from_file

register(
    id='Overcooked-v0',
    entry_point='overcooked_ai_py.mdp.overcooked_env:Overcooked',
)

_current_dir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = _current_dir + "/data/"
ASSETS_DIR = _current_dir + "/assets/"
COMMON_TESTS_DIR = "/".join(_current_dir.split("/")[:-1]) + "/common_tests/"
HUMAN_DATA_DIR = DATA_DIR + "human_data/"
LAYOUTS_DIR = DATA_DIR + "layouts/"
IMAGE_DIR = os.path.join(_current_dir, "images")
PCG_EXP_IMAGE_DIR = os.path.join(IMAGE_DIR, "pcg_exp")
HUMAN_LVL_IMAGE_DIR = os.path.join(IMAGE_DIR, "human_levels")


def read_layout_dict(layout_name):
    return load_dict_from_file(
        os.path.join(LAYOUTS_DIR, layout_name + ".layout"))
