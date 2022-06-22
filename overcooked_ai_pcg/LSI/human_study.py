import pygame
import os
import csv
import ast
import time
import toml
import argparse
import pickle
import numpy as np
import pandas as pd
from pprint import pprint
from overcooked_ai_pcg import (LSI_HUMAN_STUDY_RESULT_DIR,
                               LSI_HUMAN_STUDY_CONFIG_DIR,
                               LSI_HUMAN_STUDY_AGENT_DIR)
from overcooked_ai_pcg.helper import init_env, init_qmdp_agent
from overcooked_ai_py.agents.agent import HumanPlayer, StayAgent
from overcooked_ai_py.mdp.overcooked_mdp import Direction, Action
from overcooked_ai_py.mdp.overcooked_env import MAX_HORIZON

HUMAN_STUDY_ENV_HORIZON = 150

SUB_STUDY_TYPES = [
    'even_workloads',
    'uneven_workloads',
    'high_team_fluency',
    'low_team_fluency',
]

SUB_STUDY_TYPES_VALUE_RANGE = [[i for i in range(-6, 7, 1)],
                               [i for i in range(-6, 7, 1)],
                               [i for i in range(0, 101, 1)],
                               [i for i in range(0, 101, 1)]]

NON_TRIAL_STUDY_TYPES = [
    'all',
    *SUB_STUDY_TYPES,
]

DETAILED_STUDY_TYPES = [f"{x}-{i}" for x in SUB_STUDY_TYPES for i in range(3)]

ALL_STUDY_TYPES = [
    'all',
    'trial',
    *SUB_STUDY_TYPES,
]


class OvercookedGame:
    """Class for human player to play an Overcooked game with given AI agent.
       Most of the code from http://pygametutorials.wikidot.com/tutorials-basic.

    Args:
        env: Overcooked Environment.
        agent: AI agent that the human plays game with.
        rand_seed: random seed.
        agent_idx: index of the AI agent.
        slow_time (bool): whether the AI agent take multiple actions each time
            the human take one action.
    """
    def __init__(self, env, agent, agent_idx, rand_seed, slow_time=False):
        self._running = True
        self._display_surf = None
        self.env = env
        self.agent = agent
        self.agent_idx = agent_idx
        self.human_player = HumanPlayer()
        self.slow_time = slow_time
        self.total_sparse_reward = 0
        self.last_state = None
        self.rand_seed = rand_seed

        # Saves when each soup (order) was delivered
        self.checkpoints = [env.horizon - 1] * env.num_orders
        self.cur_order = 0
        self.timestep = 0
        self.joint_actions = []

    def on_init(self):
        pygame.init()

        # Adding AI agent as teammate
        self.agent.set_agent_index(self.agent_idx)
        self.agent.set_mdp(self.env.mdp)
        self.human_player.set_agent_index(1 - self.agent_idx)
        self.human_player.set_mdp(self.env.mdp)

        self.env.render("full")
        self._running = True
        np.random.seed(self.rand_seed)

    def on_event(self, event):
        done = False
        next_state = None

        if event.type == pygame.KEYDOWN:
            pressed_key = event.dict['key']
            action = None

            if pressed_key == pygame.K_UP:
                action = Direction.NORTH
            elif pressed_key == pygame.K_RIGHT:
                action = Direction.EAST
            elif pressed_key == pygame.K_DOWN:
                action = Direction.SOUTH
            elif pressed_key == pygame.K_LEFT:
                action = Direction.WEST
            elif pressed_key == pygame.K_SPACE:
                action = Action.INTERACT
            elif pressed_key == pygame.K_s:
                action = Action.STAY

            if action in Action.ALL_ACTIONS:

                done, next_state = self.step_env(action)

                if self.slow_time and not done:
                    for _ in range(2):
                        action = Action.STAY
                        done, next_state = self.step_env(action)
                        if done:
                            break

        if event.type == pygame.QUIT or done:
            self._running = False
            self.last_state = next_state

    def step_env(self, my_action):
        agent_action = self.agent.action(self.env.state)[0]

        if self.agent_idx == 1:
            joint_action = (my_action, agent_action)
        else:
            joint_action = (agent_action, my_action)

        self.joint_actions.append(joint_action)
        next_state, timestep_sparse_reward, done, info = self.env.step(
            joint_action)

        # update logs of human player for bc calculations
        self.human_player.update_logs(next_state, my_action)

        if timestep_sparse_reward > 0:
            self.checkpoints[self.cur_order] = self.timestep
            self.cur_order += 1

        self.total_sparse_reward += timestep_sparse_reward
        self.timestep += 1
        # print("Timestep:", self.timestep)
        return done, next_state

    def on_loop(self):
        pass

    def on_render(self):
        self.env.render(mode="full")

    def on_cleanup(self):
        pygame.quit()

    def on_execute(self):
        if self.on_init() == False:
            self._running = False

        while (self._running):
            for event in pygame.event.get():
                self.on_event(event)
            self.on_render()
            self.on_loop()
        self.on_cleanup()
        workloads = self.last_state.get_player_workload()
        concurr_active = self.last_state.cal_concurrent_active_sum()
        stuck_time = self.last_state.cal_total_stuck_time()

        # from IPython import embed
        # embed()

        fitness = self.total_sparse_reward + 1
        for checked_time in reversed(self.checkpoints):
            fitness *= self.env.horizon
            fitness -= checked_time

        return (fitness, self.total_sparse_reward, self.checkpoints, workloads,
                self.joint_actions, concurr_active, stuck_time)


def load_qmdp_agent(env, agent_save_path):
    ai_agent = None
    if agent_save_path is not None:
        # agent saved before, load it.
        if os.path.exists(agent_save_path):
            with open(agent_save_path, 'rb') as f:
                ai_agent = pickle.load(f)

    # agent not found, recreate it and save it if a path is given.
    if ai_agent == None:
        ai_agent = init_qmdp_agent(env.mdp)
        if agent_save_path is not None:
            with open(agent_save_path, 'wb') as f:
                pickle.dump(ai_agent, f)
    return ai_agent


def load_human_log_data(log_index):
    human_log_csv = os.path.join(LSI_HUMAN_STUDY_RESULT_DIR, log_index,
                                 "human_log.csv")
    if not os.path.exists(human_log_csv):
        print("Log dir does not exit.")
        exit(1)
    human_log_data = pd.read_csv(human_log_csv)
    return human_log_csv, human_log_data


def replay_with_joint_actions(lvl_str, joint_actions, plot=True):
    """Replay a game play with given level and joint actions.

    Args:
        joint_actions (list of tuple of tuple): Joint actions.
    """
    env = init_env(lvl_str, horizon=HUMAN_STUDY_ENV_HORIZON)
    done = False
    # Hacky: use human agent for replay.
    ai_agent = HumanPlayer()
    player = HumanPlayer()

    ai_agent.set_agent_index(0)
    ai_agent.set_mdp(env.mdp)
    player.set_agent_index(1)
    player.set_mdp(env.mdp)
    i = 0
    last_state = None
    total_sparse_reward = 0
    checkpoints = [env.horizon - 1] * env.num_orders
    cur_order = 0

    while not done:
        if plot:
            env.render()
            time.sleep(0.2)
        ai_agent.update_logs(env.state, joint_actions[i][0])
        player.update_logs(env.state, joint_actions[i][1])
        next_state, timestep_sparse_reward, done, info = env.step(
            joint_actions[i])
        total_sparse_reward += timestep_sparse_reward

        if timestep_sparse_reward > 0:
            checkpoints[cur_order] = i
            cur_order += 1
        # print(joint_actions[i])
        last_state = next_state
        i += 1

    # recalculate the bcs
    workloads = next_state.get_player_workload()
    concurr_active = next_state.cal_concurrent_active_sum()
    stuck_time = next_state.cal_total_stuck_time()
    return workloads, concurr_active, stuck_time, checkpoints, i


def human_play(
    lvl_str,
    ai_agent=None,
    agent_save_path=None,
    horizon=HUMAN_STUDY_ENV_HORIZON,
):
    """Function that allows human to play with an ai_agent.

    Args:
        lvl_str (str): Level string.
        ai_agent (Agent): Agent that human plays with. Default is QMDP agent.
        agent_save_path (str): Path to the pre-saved ai agent. If nothing is
            found, it will be saved to there.
        horizon (int): max number of timesteps to play.
    """
    env = init_env(lvl_str, horizon=horizon)
    if ai_agent is None:
        ai_agent = load_qmdp_agent(env, agent_save_path)
    theApp = OvercookedGame(env, ai_agent, agent_idx=0, rand_seed=10)
    return theApp.on_execute()


def read_in_study_lvl(_dir):
    """Read in levels used for human study in the given directory."""
    lvls = []
    for i, _file in enumerate(os.listdir(_dir)):
        if _file.endswith(".tml"):
            lvl = toml.load(os.path.join(_dir, _file))
            lvls.append(lvl)
    return lvls


def write_row(csv_file, to_add):
    """Append a row to csv file"""
    with open(csv_file, 'a+') as f:
        writer = csv.writer(f)
        writer.writerow(to_add)


def write_to_human_exp_log(lvl_type_full, results, lvl_config):
    """Write to human exp log.

    Args:
        human_log_csv (str): Path to the human_log.csv file.
        results (tuple) all of the results returned from the human study.
        lvl_config (dic): toml config dic of the level.
    """
    assert os.path.exists(human_log_csv)

    to_write = [
        lvl_config["lvl_type"] if "lvl_type" in lvl_config else None,
        lvl_config["ID"] if "ID" in lvl_config else None,
        lvl_config["exp_log_dir"] if "exp_log_dir" in lvl_config else None,
        lvl_config["row_index"] if "row_index" in lvl_config else None,
        lvl_config["column_index"] if "column_index" in lvl_config else None,
        lvl_config["matrix_index"] if "matrix_index" in lvl_config else None,
        *results,
        lvl_config["lvl_str"],
    ]

    write_row(human_log_csv, to_write)


def create_human_exp_log():
    """ Create human_study/result/<exp_id>. <exp_id> would be determined by the
        first digit that does not exist under the human_exp directory.

        Returns:
            Path to the csv file to which the result is be written.
    """
    # increment from 0 to find a directory name that does not exist yet.
    exp_dir = 0
    while os.path.isdir(os.path.join(LSI_HUMAN_STUDY_RESULT_DIR,
                                     str(exp_dir))):
        exp_dir += 1
    exp_dir = os.path.join(LSI_HUMAN_STUDY_RESULT_DIR, str(exp_dir))
    os.mkdir(exp_dir)

    # create csv file to store results
    human_log_csv = os.path.join(exp_dir, 'human_log.csv')

    # construct labels
    data_labels = [
        "lvl_type",
        "ID",
        "exp_log_dir",  # exp log directory which determines elite map
        "row_index",
        "column_index",
        "matrix_index",  # could be None cuz elite map can be 2D or 3D
        "fitness",
        "total_sparse_reward",
        "checkpoints",
        "workloads",
        "joint_actions",
        "concurr_active",
        "stuck_time",
        "lvl_str",
    ]

    write_row(human_log_csv, data_labels)

    return human_log_csv


def correct_study_type(study_type, lvl_type):
    if study_type == "all" and lvl_type != "trial":
        return True
    else:
        return lvl_type.startswith(study_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--study',
        help=
        "Which set of study to run. Should be one of 'trial', 'even_workloads', 'uneven_workloads', 'high_team_fluency', 'low_team_fluency' and 'all'.",
        default=False)

    parser.add_argument('--replay',
                        action='store_true',
                        help='Whether use the replay mode',
                        default=False)
    parser.add_argument('--reload_study',
                        action='store_true',
                        help='Whether to continue running a previous study',
                        default=False)
    parser.add_argument('-l',
                        '--log_index',
                        help='Integer: index of the study log',
                        required=False,
                        default=-1)
    parser.add_argument('-type',
                        help='Integer: type of the game level.',
                        required=False,
                        default=None)
    opt = parser.parse_args()

    np.random.seed(1)
    # not replay, run the study
    if not opt.replay:
        # read in human study levels
        study_lvls = pd.read_csv(
            os.path.join(LSI_HUMAN_STUDY_CONFIG_DIR, "study_lvls.csv"))

        # running a new study
        if not opt.reload_study:
            # quite if study type not recognized
            if opt.study not in ALL_STUDY_TYPES:
                print(
                    "Study type not supported. Must be one of the following:",
                    ", ".join(ALL_STUDY_TYPES))
                exit(1)

            if opt.study == 'trial':
                study_lvls = study_lvls[study_lvls["lvl_type"] == 'trial']
                lvl_config = study_lvls.iloc[0]
                agent_save_path = os.path.join(
                    LSI_HUMAN_STUDY_AGENT_DIR,
                    "{lvl_type}.pkl".format(lvl_type=lvl_config["lvl_type"]))

                print("trial")
                print(lvl_config["lvl_str"])

                # play 3 times
                # first 2 times with stay agent and infinite horizon
                for _ in range(2):
                    human_play(
                        lvl_config["lvl_str"],
                        ai_agent=StayAgent(),
                        agent_save_path=agent_save_path,
                        horizon=MAX_HORIZON,
                    )
                # thrid time with qmdp agent with finite horizon
                human_play(
                    lvl_config["lvl_str"],
                    agent_save_path=agent_save_path,
                )

            elif opt.study in NON_TRIAL_STUDY_TYPES:
                # initialize the result log files
                human_log_csv = create_human_exp_log()

                # shuffle the order if playing all
                if opt.study == 'all':
                    study_lvls = study_lvls.sample(frac=1)

                # play all of the levels
                for index, lvl_config in study_lvls.iterrows():
                    # check study type:
                    if correct_study_type(opt.study, lvl_config["lvl_type"]):
                        agent_save_path = os.path.join(
                            LSI_HUMAN_STUDY_AGENT_DIR, "{lvl_type}.pkl".format(
                                lvl_type=lvl_config["lvl_type"]))
                        print(lvl_config["lvl_type"])
                        print(lvl_config["lvl_str"])
                        results = human_play(lvl_config["lvl_str"],
                                             agent_save_path=agent_save_path)
                        # write the results
                        if lvl_config["lvl_type"] != "trial":
                            write_to_human_exp_log(human_log_csv, results,
                                                   lvl_config)

        # loading an existing study and continue running it.
        else:
            log_index = opt.log_index
            assert int(log_index) >= 0
            human_log_csv, human_log_data = load_human_log_data(log_index)

            # find levels need to run and play them
            for lvl_type in DETAILED_STUDY_TYPES:
                if lvl_type not in human_log_data["lvl_type"].to_list():
                    lvl_config = study_lvls[study_lvls["lvl_type"] ==
                                            lvl_type].iloc[0]
                    lvl_str = lvl_config["lvl_str"]
                    print(lvl_config["lvl_type"])
                    print(lvl_str)
                    agent_save_path = os.path.join(
                        LSI_HUMAN_STUDY_AGENT_DIR,
                        "{lvl_type}.pkl".format(lvl_type=lvl_type))
                    results = human_play(lvl_str,
                                         agent_save_path=agent_save_path)

                    # write the results
                    if lvl_config["lvl_type"] != "trial":
                        write_to_human_exp_log(human_log_csv, results,
                                               lvl_config)

    # replay the specified study
    else:
        log_index = opt.log_index
        lvl_type = opt.type

        # get level string and logged joint actions from log file
        _, human_log_data = load_human_log_data(log_index)
        lvl_str = human_log_data[human_log_data["lvl_type"] ==
                                 lvl_type]["lvl_str"].iloc[0]
        joint_actions = ast.literal_eval(human_log_data[
            human_log_data["lvl_type"] == lvl_type]["joint_actions"].iloc[0])

        # replay the game
        replay_with_joint_actions(lvl_str, joint_actions)
