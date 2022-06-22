import gc
import json
import os
import pickle
import time
from statistics import median

import numpy as np
import pygame
import toml
import torch
from matplotlib import pyplot as plt
from overcooked_ai_py import LAYOUTS_DIR, read_layout_dict
from overcooked_ai_py.agents.agent import *
from overcooked_ai_py.mdp.graphics import render_from_grid
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.planning.planners import (
    Heuristic, MediumLevelActionManager, MediumLevelMdpPlanner,
    MediumLevelPlanner)

from overcooked_ai_pcg import (ERR_LOG_PIC, G_PARAM_FILE, LSI_CONFIG_AGENT_DIR,
                               LSI_CONFIG_ALGO_DIR, LSI_CONFIG_MAP_DIR)

obj_types = "12XSPOD "
num_obj_type = len(obj_types)

CONFIG = {
    "start_order_list": ['onion'] * 2,
    "cook_time": 10,
    "num_items_for_soup": 3,
    "delivery_reward": 20,
    "rew_shaping_params": None
}

BASE_PARAMS = {
    'start_orientations': False,
    'wait_allowed': False,
    'counter_goals': [],
    'counter_drop': [],
    'counter_pickup': [],
    'same_motion_goals': True
}


def vertical_flip(np_lvl):
    """
    Return the vertically flipped version of the input np level.
    """
    np_lvl_vflip = np.zeros(np_lvl.shape)
    height, width = np_lvl.shape
    for x in range(height):
        for y in range(width):
            np_lvl_vflip[x][y] = np_lvl[x][width - y - 1]

    return np_lvl_vflip.astype(np.uint8)


def horizontal_flip(np_lvl):
    """
    Return the horizontally flipped version of the input np level.
    """
    np_lvl_hflip = np.zeros(np_lvl.shape)
    height = np_lvl.shape[0]
    for x in range(height):
        np_lvl_hflip[x] = np_lvl[height - x - 1]
    return np_lvl_hflip.astype(np.uint8)


def lvl_str2number(raw_layout):
    """
    Turns pure string formatted lvl to num encoded format
    """
    np_lvl = np.zeros((len(raw_layout), len(raw_layout[0])))
    for x, row in enumerate(raw_layout):
        row = row.strip()
        for y, tile in enumerate(row):
            np_lvl[x, y] = obj_types.index(tile)
    return np_lvl


def lvl_number2str(np_lvl):
    """
    Turns num encoded format to pure string formatted lvl
    """
    lvl_str = ""
    for lvl_row in np_lvl:
        for tile_int in lvl_row:
            lvl_str += obj_types[int(tile_int)]
        lvl_str += "\n"
    return lvl_str


def lvl_str2grid(lvl_str):
    """
    Turns pure string formatted lvl to grid format compatible with overcooked-AI env
    """
    return [layout_row.strip() for layout_row in lvl_str.split("\n")][:-1]


def read_in_training_data(data_path, sub_dir=None):
    """
    Read in .layouts file and return the data

    Args:
        data_path: path to the directory containing the training data

    returns: a 3D np array of size num_lvl x lvl_height x lvl_width 
             containing the encoded levels
    """
    lvls = []
    for layout_file in os.listdir(data_path):
        if layout_file.endswith(".layout") and layout_file.startswith("gen"):
            layout_name = layout_file.split('.')[0]
            if sub_dir is None:
                raw_layout = read_layout_dict(layout_name)
            else:
                raw_layout = read_layout_dict(sub_dir + "/" + layout_name)
            raw_layout = raw_layout['grid'].split('\n')

            np_lvl = lvl_str2number(raw_layout)

            # data agumentation: add flipped levels to data set
            np_lvl = np_lvl.astype(np.uint8)
            np_lvl_vflip = vertical_flip(np_lvl)
            np_lvl_hflip = horizontal_flip(np_lvl)
            np_lvl_vhflip = vertical_flip(np_lvl_hflip)
            lvls.append(np_lvl)
            lvls.append(np_lvl_vflip)
            lvls.append(np_lvl_hflip)
            lvls.append(np_lvl_vhflip)

    return np.array(lvls)


def read_in_lsi_config(exp_config_file):
    experiment_config = toml.load(exp_config_file)
    algorithm_config = toml.load(
        os.path.join(LSI_CONFIG_ALGO_DIR,
                     experiment_config["algorithm_config"]))
    elite_map_config = toml.load(
        os.path.join(LSI_CONFIG_MAP_DIR,
                     experiment_config["elite_map_config"]))
    agent_configs = []
    for agent_config_file in experiment_config["agent_config"]:
        agent_config = toml.load(
            os.path.join(LSI_CONFIG_AGENT_DIR, agent_config_file))
        agent_configs.append(agent_config)
    return experiment_config, algorithm_config, elite_map_config, agent_configs


def plot_err(average_errG_log, average_errD_log, average_errD_fake_log,
             average_errD_real_log, average_D_x_log, average_D_G_z1_log,
             average_D_G_z2_log):
    """
    Given lists of recorded errors and plot them.
    """
    plt.subplot(2, 2, 1)
    plt.plot(average_errD_log, 'b', label="err_D")
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(average_errD_fake_log, 'r', label="err_D_fake")
    plt.plot(average_errD_real_log, 'g', label="err_D_real")
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(average_errG_log, 'r', label="err_G")
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(average_D_x_log, 'r', label="D(x)")
    plt.plot(average_D_G_z1_log, 'g', label="D(G(z1))")
    plt.plot(average_D_G_z2_log, 'b', label="D(G(z2))")
    plt.legend()

    plt.savefig(ERR_LOG_PIC)
    plt.show()


def reset_env_from_mdp(mdp):
    env = OvercookedEnv.from_mdp(mdp, info_level=0, horizon=100)
    return env


def setup_env_from_grid(layout_grid,
                        agent_config,
                        worker_id=0,
                        human_preference=0.3,
                        human_adaptiveness=0.5):
    """
    Set up random agents and overcooked env to run demo game.
    Args:
        layout_grid: list of string each representing a row of layout
    """

    mdp = OvercookedGridworld.from_grid(layout_grid, CONFIG)
    env = OvercookedEnv.from_mdp(mdp, info_level=0, horizon=100)

    start_time = time.time()

    agent1_config = agent_config["Agent1"]
    agent2_config = agent_config["Agent2"]

    # # Set up 1: two coupled planning agent
    # mlp_planner1 = MediumLevelPlanner(mdp, base_params)
    # mlp_planner2 = MediumLevelPlanner(mdp, base_params)

    # agent1 = CoupledPlanningAgent(mlp_planner1)
    # agent2 = CoupledPlanningAgent(mlp_planner2)

    # # Set up 2: Stayagent + GreedyHumanModel
    # mlp_planner = MediumLevelPlanner(mdp, base_params)
    # agent1 = StayAgent()
    # agent2 = GreedyHumanModel(mlp_planner, env)

    # Set up 3: Fixed plan agents
    if agent1_config["name"] == "fixed_plan_agent" and agent2_config[
            "name"] == "fixed_plan_agent":
        print("worker(%d): Pre-constructing graph..." % (worker_id))
        mlp_planner = MediumLevelPlanner(mdp, BASE_PARAMS)
        print("worker(%d): Planning..." % (worker_id))
        joint_plan = \
            mlp_planner.get_low_level_action_plan(
                env.state,
                Heuristic(mlp_planner.mp).simple_heuristic,
                delivery_horizon=agent_config["joint_plan"]["delivery_horizon"],
                goal_info=agent_config["joint_plan"]["goal_info"])

        plan1 = []
        plan2 = []
        for joint_action in joint_plan:
            action1, action2 = joint_action
            plan1.append(action1)
            plan2.append(action2)

        agent1 = FixedPlanAgent(plan1)
        agent2 = FixedPlanAgent(plan2)

        del mlp_planner

    agent1.set_agent_index(0)
    agent2.set_agent_index(1)

    gc.collect()

    return agent1, agent2, env, mdp


def save_gan_param(G_params):
    with open(G_PARAM_FILE, "w") as f:
        json.dump(G_params, f)


def read_gan_param():
    with open(G_PARAM_FILE, "r") as f:
        G_params = json.load(f)
    return G_params


def visualize_lvl(lvl_str, log_dir, filename):
    """
    Render and save the level without running game
    """
    grid = [layout_row for layout_row in lvl_str.split("\n")][:-1]
    assert len(set([len(layout_row) for layout_row in grid])) == 1
    render_from_grid(grid, log_dir, filename)


def run_overcooked_game(ind,
                        agent_config,
                        render=True,
                        worker_id=0,
                        num_iters=1,
                        track_belief=False,
                        delay=500,
                        img_name=None):
    """
    Run one turn of overcooked game and return the sparse reward as fitness

    Args:
        render (bool): Whether to render the environment with pygame.
        delay (int): Milliseconds to wait between rendering frames.
        img_name (callable): If passed in, this should be a callable that takes
            in a timestep and outputs the filename for an image. An image of the
            env will then be saved to this file. Note: `render` must be True for
            this callable to be used.
    """
    agent1, agent2, env, mdp = init_env_and_agent(ind,
                                                  agent_config,
                                                  worker_id=worker_id)

    fitnesses = []
    total_sparse_rewards = []
    checkpointses = []
    workloadses = []
    joint_actionses = []
    concurr_actives = []
    stuck_times = []
    np.random.seed(ind.rand_seed)

    for num_iter in range(num_iters):
        done = False
        total_sparse_reward = 0
        last_state = None
        timestep = 0

        # Saves when each soup (order) was delivered
        checkpoints = [env.horizon - 1] * env.num_orders
        cur_order = 0

        # store all actions
        joint_actions = []

        while not done:
            if render:
                env.render()
                time.sleep(0.5)
            joint_action = (agent1.action(env.state)[0],
                            agent2.action(env.state)[0])
            # print(joint_action)
            joint_actions.append(joint_action)
            next_state, timestep_sparse_reward, done, info = env.step(
                joint_action)
            total_sparse_reward += timestep_sparse_reward

            if timestep_sparse_reward > 0:
                checkpoints[cur_order] = timestep
                cur_order += 1

            last_state = next_state
            timestep += 1

        workloads = last_state.get_player_workload()
        concurr_active = last_state.cal_concurrent_active_sum()
        stuck_time = last_state.cal_total_stuck_time()

        # Smooth fitness is the total reward tie-broken by soup delivery times.
        # Later soup deliveries are higher priority.
        fitness = total_sparse_reward + 1
        for timestep in reversed(checkpoints):
            fitness *= env.horizon
            fitness -= timestep

        #print("fitness is: " + str(fitness))

        fitnesses.append(fitness)
        total_sparse_rewards.append(total_sparse_reward)
        checkpointses.append(checkpoints)
        workloadses.append(workloads)
        joint_actionses.append(joint_actions)
        concurr_actives.append(concurr_active)
        stuck_times.append(stuck_time)

        env = reset_env_from_mdp(mdp)

    if agent_config["Search"]["multi_iter"] == False:
        fitnesses = [fitnesses[0] for i in range(num_iters)]
        total_sparse_rewards = [
            total_sparse_rewards[0] for i in range(num_iters)
        ]
        checkpointses = [checkpointses[0] for i in range(num_iters)]
        workloadses = [workloadses[0] for i in range(num_iters)]
        joint_actionses = [joint_actionses[0] for i in range(num_iters)]
        concurr_actives = [concurr_actives[0] for i in range(num_iters)]
        stuck_times = [stuck_times[0] for i in range(num_iters)]

    if num_iters > 1:
        checkpointses = np.array(checkpointses)
        fitnesses.append(median(fitnesses))
        total_sparse_rewards.append(
            sum(total_sparse_rewards) / len(total_sparse_rewards))
        checkpoint = [
            sum(checkpointses[:, i]) / len(checkpointses[:, i])
            for i in range(len(checkpointses[0]))
        ]
        checkpointses = np.append(checkpointses, [checkpoint], axis=0)
        workloadses.append(get_workload_avg(workloadses))
        concurr_actives.append(sum(concurr_actives) / len(concurr_actives))
        stuck_times.append(sum(stuck_times) / len(stuck_times))

    # Free up some memory
    del agent1, agent2, env, mdp

    # # set necessary variables for ind
    # ind.fitnesses.append(fitness)
    # ind.scores.append(total_sparse_reward)
    # ind.checkpoints.append(checkpoints)
    # ind.player_workloads.append(workloads)
    # ind.joint_actions.append(joint_actions)

    return fitnesses, total_sparse_rewards, checkpointses, workloadses, joint_actionses, concurr_actives, stuck_times
    # return fitness, total_sparse_reward, checkpoints, workloads, joint_actions, concurr_active, stuck_time
    # return ind


def get_workload_avg(workloadses):
    avg_workloads = []
    for agent in range(2):
        a = 0
        b = 0
        c = 0
        for workloads in workloadses:
            a += workloads[agent]['num_ingre_held']
            b += workloads[agent]['num_plate_held']
            c += workloads[agent]['num_served']
        a /= len(workloadses)
        b /= len(workloadses)
        c /= len(workloadses)
        avg_workloads.append({
            "num_ingre_held": round(a),
            "num_plate_held": round(b),
            "num_served": round(c),
        })
    return avg_workloads


def init_env_and_agent(ind, agent_config, worker_id=0):
    lvl_str = ind.level
    grid = lvl_str2grid(lvl_str)
    agent1, agent2, env, mdp = setup_env_from_grid(
        grid,
        agent_config,
        worker_id=worker_id,
        human_preference=ind.human_preference,
        human_adaptiveness=ind.human_adaptiveness)

    return agent1, agent2, env, mdp


def init_env(lvl_str, horizon=100):
    grid = lvl_str2grid(lvl_str)
    mdp = OvercookedGridworld.from_grid(grid, CONFIG)
    env = OvercookedEnv.from_mdp(mdp, info_level=0, horizon=horizon)
    return env


def gen_int_rnd_lvl(size):
    """
    Randomly generate an unfixed integer level of specified size

    Args:
        size: 2D tuple of integers with format (height, width)
    """
    return np.random.randint(len(obj_types), size=size)
