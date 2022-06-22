import argparse
import json
import os
import subprocess
import time
from tqdm import tqdm

import numpy as np
import torch
from torch.autograd import Variable

from overcooked_ai_pcg import GAN_TRAINING_DIR, LSI_CONFIG_EXP_DIR
from overcooked_ai_pcg.GAN_training import dcgan
from overcooked_ai_pcg.helper import (gen_int_rnd_lvl, lvl_number2str,
                                      lvl_str2grid, obj_types, read_gan_param,
                                      read_in_lsi_config, run_overcooked_game,
                                      setup_env_from_grid, visualize_lvl,
                                      read_layout_dict, lvl_str2number)
from overcooked_ai_pcg.LSI.qd_algorithms import Individual
from overcooked_ai_pcg.milp_repair import repair_lvl
from overcooked_ai_py import PCG_EXP_IMAGE_DIR, LAYOUTS_DIR, HUMAN_LVL_IMAGE_DIR
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld


class DocplexFailedError(Exception):
    pass


def generate_lvl(batch_size,
                 generator=None,
                 latent_vector=None,
                 worker_id=0,
                 return_unrepaired=False,
                 lvl_int_unrepaired=None,
                 lvl_size=(10, 15),
                 mode="GAN"):
    """
    Generate level string from random latent vector given the path to the train netG model, and use MILP solver to repair it

    Args:
        generator (DCGAN): netG model
        latent_vector: np.ndarray with the required dimension.
                       When it is None, a new vector will be randomly sampled
        lvl_int_unrepaired: np.ndarray unrepaired level in int format. If
                            passed in, just repaire the level passed in.
        mode (string): "GAN" to generate level using GAN;
                       "random" to generate level randomly
    """
    # if an unrepaired level is already passed in,
    # just repaire it.
    if lvl_int_unrepaired is None:
        # generate the level randomly
        if mode == "random":
            lvl_int_unrepaired = gen_int_rnd_lvl(lvl_size)

        # generate level from the GAN
        elif mode == "GAN":
            # read in G constructor params from file
            # G_params = read_gan_param()
            nz = generator.nz
            x = np.random.randn(batch_size, nz, 1, 1)

            # generator = dcgan.DCGAN_G(**G_params)
            # generator.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
            if latent_vector is None:
                latent_vector = torch.FloatTensor(x).view(batch_size, nz, 1, 1)
            else:
                latent_vector = torch.FloatTensor(latent_vector).view(
                    batch_size, nz, 1, 1)
            with torch.no_grad():
                levels = generator(Variable(latent_vector))
            levels.data = levels.data[:, :, :lvl_size[0], :lvl_size[1]]
            im = levels.data.cpu().numpy()
            im = np.argmax(im, axis=1)
            lvl_int_unrepaired = im[0]

    lvl_unrepaired = lvl_number2str(lvl_int_unrepaired)

    print("worker(%d): Before repair:\n" % (worker_id) + lvl_unrepaired)

    # In order to avoid dealing with memory leaks that may arise with docplex,
    # we run `repair_lvl` in a separate process. We can't create a child process
    # since the Dask workers are daemonic processes (which cannot spawn
    # children), so we run with subprocess. The code below essentially calls
    # `python` with a small bit of code and gets back the repr of a numpy array.
    # We then eval that output to get the repaird level.
    #
    # Yes, this is sketchy.

    # Allows us to separate the array from the rest of the process's output.
    delimiter = "----DELIMITER----DELIMITER----"
    try:
        output = subprocess.run(
            [
                'python', '-c', f"""\
import numpy as np
from numpy import array
from overcooked_ai_pcg.milp_repair import repair_lvl
np_lvl = eval(\"\"\"{np.array_repr(lvl_int_unrepaired)}\"\"\")
repaired_lvl = np.array_repr(repair_lvl(np_lvl))
print("{delimiter}")
print(repaired_lvl)
"""
            ],
            stdout=subprocess.PIPE,
        ).stdout.decode('utf-8')
    except OSError:
        raise DocplexFailedError
    # The result array comes after the delimiter.
    try:
        output = output.split(delimiter)[1]
    except IndexError:
        # The delimiter was not printed due to some error, so split() only gave
        # one token.
        raise DocplexFailedError
    # The repr uses array and uint8 without np, so we make it available for eval
    # here.
    array, uint8 = np.array, np.uint8  # pylint: disable = unused-variable
    # Get the array.
    lvl_repaired = eval(output)

    lvl_str = lvl_number2str(lvl_repaired)

    print("worker(%d): After repair:\n" % (worker_id) + lvl_str)
    if return_unrepaired:
        return lvl_unrepaired, lvl_str
    return lvl_str


def generate_rnd_lvl(size, worker_id=0):
    """
    generate random level of specified size and fix it using MILP solver

    Args:
        size: 2D tuple of integers with format (height, width)
    """
    rnd_lvl_int = gen_int_rnd_lvl(size)

    print("worker(%d): Before repair:\n" % (worker_id) +
          lvl_number2str(rnd_lvl_int))

    # print("Start MILP repair...")
    lvl_repaired = repair_lvl(rnd_lvl_int)
    lvl_str = lvl_number2str(lvl_repaired)

    print("worker(%d): After repair:\n" % (worker_id) + lvl_str)
    return lvl_str


def generate_all_human_lvl():
    for layout_file in os.listdir(LAYOUTS_DIR):
        if layout_file.endswith(".layout") and layout_file.startswith("gen"):
            layout_name = layout_file.split('.')[0]
            raw_layout = read_layout_dict(layout_name)
            raw_layout = raw_layout['grid'].split('\n')
            np_lvl = lvl_str2number(raw_layout)
            lvl_str = lvl_number2str(np_lvl.astype(np.int))
            visualize_lvl(lvl_str, HUMAN_LVL_IMAGE_DIR, layout_name + ".png")


def main(config, lvl_size, gan_pth_path):
    all_lvl_strs = {
        "gan_only": [],
        "gan_milp": [],
        "milp_only": [],
    }
    for _ in tqdm(range(1000)):
        # initialize saving directory
        time_str = time.strftime("%Y-%m-%d_%H-%M-%S")
        base_log_dir = time_str
        log_dir = os.path.join(PCG_EXP_IMAGE_DIR, base_log_dir)
        os.mkdir(log_dir)

        # generate using full pipeline
        G_params = read_gan_param()
        gan_state_dict = torch.load(gan_pth_path,
                                    map_location=lambda storage, loc: storage)
        generator = dcgan.DCGAN_G(**G_params)
        generator.load_state_dict(gan_state_dict)
        lvl_gan_only, lvl_gan_milp = generate_lvl(1,
                                                  generator,
                                                  lvl_size=lvl_size,
                                                  return_unrepaired=True)
        visualize_lvl(lvl_gan_only, log_dir, "gan_only_unrepaired.png")
        visualize_lvl(lvl_gan_milp, log_dir, "gan_milp_repaired.png")

        #from IPython import embed
        #embed()
        # generate randomly then using milp to repair
        #lvl_str = generate_rnd_lvl(lvl_size)
        #visualize_lvl(lvl_str, log_dir, "milp_only.png")
        lvl_milp_only = generate_rnd_lvl(lvl_size)
        visualize_lvl(lvl_milp_only, log_dir, "milp_only.png")

        all_lvl_strs["gan_only"].append(lvl_gan_only)
        all_lvl_strs["gan_milp"].append(lvl_gan_milp)
        all_lvl_strs["milp_only"].append(lvl_milp_only)

    # save lvl_str
    with open('all_lvl_strs.json', 'w') as outfile:
        json.dump(all_lvl_strs, outfile)

    # lvl_str = """XXPXX
    #              T  2T
    #              X1  O
    #              XXDSX
    #              """
    # lvl_str = """XXXPPXXX
    #              X  2   X
    #              D XXXX S
    #              X  1   X
    #              XXXOOXXX
    #              """
    # lvl_str = """XXXXXXXXXXXXXXX
    #              O 1     XX    D
    #              X  X2XXXXXXXX S
    #              O XX     XXXX X
    #              X         X   X
    #              O          X  X
    #              X  XXXXXXX    X
    #              X  XXXX  PXXX X
    #              X          X  X
    #              XXXXXXXXXXXXXXX
    #              """

    # _, _, _, agent_configs = read_in_lsi_config(config)
    # ind = Individual()
    # ind.level = lvl_str
    # fitness, _, _, _, _ = run_overcooked_game(ind, agent_configs[0], render=False)
    # print("fitness: %d" % fitness)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c',
                        '--config',
                        help='path of experiment config file',
                        required=False,
                        default=os.path.join(
                            LSI_CONFIG_EXP_DIR,
                            "MAPELITES_workloads_diff_fixed_plan.tml"))
    parser.add_argument('--size_version',
                        type=str,
                        default="large",
                        help='Size of the level. \
                             "small" for (6, 9), \
                             "large" for (10, 15)')
    opt = parser.parse_args()

    lvl_size = None
    gan_pth_path = None
    if opt.size_version == "small":
        lvl_size = (6, 9)
        gan_pth_path = os.path.join(GAN_TRAINING_DIR,
                                    "netG_epoch_49999_999_small.pth")
    elif opt.size_version == "large":
        lvl_size = (10, 15)
        gan_pth_path = os.path.join(GAN_TRAINING_DIR,
                                    "netG_epoch_49999_999_large.pth")
    main(opt.config, lvl_size, gan_pth_path)
