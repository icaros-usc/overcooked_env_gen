import argparse, toml, os, json
import collections
import random, time
import numpy as np
import dask.distributed
from dask_jobqueue import SLURMCluster

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.agents.agent import (HRLTrainingAgent, GreedyHumanModel,
                                           StayAgent, RandomAgent)
from overcooked_ai_py.planning.planners import (HumanSubtaskQMDPPlanner,
                                                MediumLevelPlanner)

#Hyperparameters
learning_rate = 0.0005
gamma = 0.98
buffer_limit = 50000
batch_size = 32


def setup_env_w_agents(config, n_epi=None, env_list=None, human=None, q=None):
    """
    Setup environment and agents.

    NOTE: Green hat is agent 1/robot; Blue hat is agent 2/human
    """
    print('Setting environment and agent...')
    env_config, human_config = config["Env"], config["Human"]
    env_config["layout_dir"].split('/')[-1]
    if env_list is not None:
        if n_epi < len(env_list):
            env_config["layout_name"] = env_config["layout_dir"].split(
                '/')[-1] + '/' + env_list[n_epi].split('.')[0]
        else:
            r = random.randint(0, len(env_list) - 1)
            env_config["layout_name"] = env_config["layout_dir"].split(
                '/')[-1] + '/' + env_list[r].split('.')[0]

    mdp = OvercookedGridworld.from_layout_name(
        env_config["layout_name"], **env_config["params_to_overwrite"])
    env = OvercookedEnv.from_mdp(mdp,
                                 info_level=0,
                                 horizon=env_config["horizon"])

    human_type = human_config["name"] if human is None else human

    if human_type == "greedy_agent":
        mlp_planner = MediumLevelPlanner(mdp, env_config["planner"])
        human_agent = GreedyHumanModel(
            mlp_planner, auto_unstuck=human_config["auto_unstuck"])

    if human_type == 'stay_agent':
        human_agent = StayAgent()

    human_agent.set_agent_index(1)
    human_agent.set_mdp(mdp)

    mlp_planner = MediumLevelPlanner(mdp, env_config["planner"])
    ai_agent = GreedyHumanModel(mlp_planner,
                                auto_unstuck=human_config["auto_unstuck"])
    ai_agent.set_agent_index(0)
    ai_agent.set_mdp(mdp)

    return ai_agent, human_agent, env, mdp


def get_env_info(t_obj, state, mdp):
    onion_loc = mdp.get_onion_dispenser_locations()
    pot_loc = mdp.get_pot_locations()
    dish_loc = mdp.get_dish_dispenser_locations()
    serving_loc = mdp.get_serving_locations()

    # create 4x2 matrix
    for i, loc in enumerate(onion_loc):
        t_obj['onion_loc' + str(i)] = [loc[0], loc[1]]

    for i, loc in enumerate(pot_loc):
        t_obj['pot_loc' + str(i)] = [loc[0], loc[1]]

    for i, loc in enumerate(dish_loc):
        t_obj['dish_loc' + str(i)] = [loc[0], loc[1]]

    for i, loc in enumerate(serving_loc):
        t_obj['serving_loc' + str(i)] = [loc[0], loc[1]]

    num_item_in_pot = 0
    if state.objects is not None and len(state.objects) > 0:
        for obj_state in state.objects.values():
            if obj_state.name == 'soup' and obj_state.state[
                    1] > num_item_in_pot:
                num_item_in_pot = obj_state.state[1]
    t_obj['order_list'] = 0 if state.order_list is None else len(
        state.order_list)
    t_obj['num_item_in_pot'] = num_item_in_pot
    return t_obj


def get_agent_info(t_obj, agent, agent_id=0):
    # hstate: subtask state and key object position
    # objects = {'onion': 0, 'soup': 1, 'dish': 2, 'None': 3}
    objects = ['onion', 'soup', 'dish', 'None']
    for obj in objects:
        t_obj['agent' + str(agent_id) + '_hold_' + obj] = 0

    t_obj['agent' + str(agent_id) + '_pos'] = agent.position
    # t_obj['held_obj'] = objects[agent.held_object.name] if agent.held_object is not None else objects['None']

    if agent.held_object is not None:
        t_obj['agent' + str(agent_id) + '_hold_' + agent.held_object.name] = 1
    else:
        t_obj['agent' + str(agent_id) + '_hold_None'] = 1

    return t_obj


def reset_env_from_mdp(mdp, config):
    print('Resetting mdp...')
    env_config = config["Env"]
    env = OvercookedEnv.from_mdp(mdp,
                                 info_level=0,
                                 horizon=env_config["horizon"])
    return env


def run_env_w_agent(n_epi, env_list, multi_agent=False):
    if config['Env']['multi']:
        ai_agent, human_agent, env, mdp = setup_env_w_agents(
            config, n_epi, env_list)

    env = reset_env_from_mdp(mdp, config)
    done = False
    log = []
    timestep = 0
    t_obj = {}
    t_obj = get_env_info(t_obj, env.state, mdp)
    if multi_agent:
        for i, player in enumerate(env.state.players):
            t_obj = get_agent_info(t_obj, player, i)
    else:
        t_obj = get_agent_info(t_obj, env.state.players[0])
    t_obj = get_agent_info(t_obj, env.state.players[0])
    t_obj['timestep'] = timestep
    log.append(t_obj)

    print('Start simulating agents...')

    while not done:
        joint_action = (ai_agent.action(env.state)[0],
                        human_agent.action(env.state)[0])
        next_state, timestep_sparse_reward, done, info = env.step(joint_action)
        timestep += 1

        t_obj = {}
        t_obj = get_env_info(t_obj, env.state, mdp)
        if multi_agent:
            for i, player in enumerate(env.state.players):
                t_obj = get_agent_info(t_obj, player, i)
        else:
            t_obj = get_agent_info(t_obj, env.state.players[0])
        t_obj['timestep'] = timestep
        log.append(t_obj)

    del ai_agent, human_agent, env, mdp

    return log


def init_log(log_path, filename):
    # to do: automize this function

    if os.path.exists(os.path.join(log_path, filename + '.json')):
        os.remove(os.path.join(log_path, filename + '.json'))

    # data_labels = ['policy_id', 'timestep']
    # # for env_info in env_infos:
    # data_labels +=  ['onion_loc', 'pot_loc', 'dish_loc', 'serving', 'order_list', 'num_in_pot']
    # # for agent_info in agent_infos:
    # data_labels += ['agent_pos', 'held_obj']

    # f = open(os.path.join(log_path, filename+'.json'), 'a+')
    # json.dump(data_labels, f)

    # # f = open(os.path.join(log_path, filename+'.csv'), 'a+')
    # # writer = csv.writer(f)
    # # writer.writerow(data_labels)
    # f.close()


def insert_log(log_path, filename, policy_count, data):
    f = open(os.path.join(log_path, filename + '.json'), 'a+')
    for line in data:
        line['policy_id'] = policy_count
        json.dump(line, f)
        f.write('\n')
    # f = open(os.path.join(log_path, filename+'.csv'), 'a+')
    # writer = csv.writer(f)
    # for line in data:
    #     writer.writerow([policy_count]+line)
    f.close()


def main(dask_client, config):
    # config overcooked env and human agent
    env_list = os.listdir(config['Env']['layout_dir'])
    env_list.remove('base.layout')

    log_path = os.path.join(config["Experiment"]["log_dir"],
                            config["Experiment"]["log_name"])

    if not os.path.exists(log_file):
        os.mkdir(log_file)
    with open(os.path.join(log_file, 'config.tml'), "w") as toml_file:
        toml.dump(config, toml_file)

    evaluations = []
    active_evals = 0
    n_epi = 0
    completed_evals = 0

    start_time = time.time()
    init_log(log_path, config["Experiment"]["log_name"])

    def _request_envs_w_agent(before_loop):
        nonlocal active_evals, n_epi
        while active_evals < config["num_cores"]:
            future = dask_client.submit(run_env_w_agent, n_epi, env_list,
                                        config["Experiment"]["multi_agent"])
            active_evals += 1
            if before_loop:
                evaluations.append(future)  # pylint: disable=no-member
            else:
                evaluations.add(future)
            print(f"{n_epi}/{len(env_list)}")
            n_epi += 1
        print(f"Active evaluations: {active_evals}")

    _request_envs_w_agent(True)
    evaluations = dask.distributed.as_completed(evaluations)
    print(f"Started {active_evals} simulations")

    # completion time of the latest simulation
    last_eval = time.time()

    for completion in evaluations:
        # process the individual
        active_evals -= 1

        try:
            log = completion.result()
            insert_log(log_path, config["Experiment"]["log_name"],
                       completed_evals, log)
            completed_evals += 1

        except dask.distributed.scheduler.KilledWorker as err:
            # worker may fail due to, for instance, memory
            print("Worker failed with the following error; continuing anyway\n"
                  "-------------------------------------------\n"
                  f"{err}\n"
                  "-------------------------------------------")
            # avoid not sending out more evaluations while evaluating initial
            # populations
            # algorithm.individuals_disbatched -= 1
            continue

        del completion

        if completed_evals < len(env_list):
            # request more evaluations if still running
            _request_envs_w_agent(False)
        else:
            # otherwise, terminate
            break

        finish_time = time.time()
    print("Total time:", str(finish_time - start_time), "seconds")


def init_dask(experiment_config, log_dir):
    """Initializes Dask with a local or SLURM cluster.

    Args:
        experiment_config (toml): toml config object of experiment
        log_dir (str): directory for storing logs
    Returns:
        A Dask client for the cluster created.
    """
    num_cores = experiment_config["num_cores"]

    if experiment_config.get("slurm", False):
        worker_logs = os.path.join(log_dir, "worker_logs")
        if not os.path.isdir(worker_logs):
            os.mkdir(worker_logs)
        output_file = os.path.join(worker_logs, 'slurm-%j.out')

        cores_per_worker = experiment_config["num_cores_per_slurm_worker"]

        # 1 process per CPU since cores == processes
        cluster = SLURMCluster(
            project=experiment_config["slurm_project"],
            cores=cores_per_worker,
            memory=f"{experiment_config['mem_gib_per_slurm_worker']}GiB",
            processes=cores_per_worker,
            walltime=experiment_config['slurm_worker_walltime'],
            job_extra=[
                f"--output {output_file}",
                f"--error {output_file}",
            ],
            death_timeout=3600,
        )

        print("### SLURM Job script ###")
        print("--------------------------------------")
        print(cluster.job_script())
        print("--------------------------------------")

        cluster.scale(cores=num_cores)
        return dask.distributed.Client(cluster)

    # Single machine -- run with num_cores worker processes.
    cluster = dask.distributed.LocalCluster(n_workers=num_cores,
                                            threads_per_worker=1,
                                            processes=True)
    return dask.distributed.Client(cluster)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c',
                        '--config',
                        help='path of config file',
                        required=True)
    opt = parser.parse_args()

    n_eqi = None
    with open(opt.config) as f:
        config = toml.load(f)

    log_file = os.path.join(config["Experiment"]["log_dir"],
                            config["Experiment"]["log_name"])
    main(init_dask(config, log_file), config)
