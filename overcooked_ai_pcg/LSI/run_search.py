"""Runs a search to illuminate the latent space."""
import argparse
import os
import pickle
import time
import shutil
import random
import dask.distributed
import toml
import torch
from dask_jobqueue import SLURMCluster
from overcooked_ai_pcg import GAN_TRAINING_DIR, LSI_CONFIG_EXP_DIR, LSI_LOG_DIR
from overcooked_ai_pcg.helper import read_gan_param, read_in_lsi_config
from overcooked_ai_pcg.LSI.evaluator import run_overcooked_eval
from overcooked_ai_pcg.LSI.logger import (FrequentMapLog, MapSummaryLog,
                                          RunningIndividualLog)
from overcooked_ai_pcg.LSI.qd_algorithms import (CMA_ME_Algorithm, FeatureMap,
                                                 MapElitesAlgorithm,
                                                 MapElitesBaselineAlgorithm,
                                                 RandomGenerator)

# How many iterations to wait before saving algorithm state.
RELOAD_FREQ = 200


def init_logging_dir(config_path, experiment_config, algorithm_config,
                     elite_map_config, agent_configs):
    """Creates the logging directory, saves configs to it, and starts a README.

    Args:
        config_path (str): path to the experiment config file
        experiment_config (toml): toml config object of current experiment
        algorithm_config: toml config object of QD algorithm
        elite_map_config: toml config object of the feature maps
        agent_configs: toml config object of the agents used
    Returns:
        log_dir: full path to the logging directory
        base_log_dir: the path without LSI_LOG_DIR prepended
    """
    # create logging directory
    exp_name = os.path.basename(config_path).replace(".tml",
                                                     "").replace("_", "-")
    time_str = time.strftime("%Y-%m-%d_%H-%M-%S")
    base_log_dir = time_str + "_" + exp_name
    log_dir = os.path.join(LSI_LOG_DIR, base_log_dir)
    os.mkdir(log_dir)

    # save configs
    with open(os.path.join(log_dir, "config.tml"), "w") as file:
        toml.dump(
            {
                "experiment_config": experiment_config,
                "algorithm_config": algorithm_config,
                "elite_map_config": elite_map_config,
                "agent_configs": agent_configs,
            },
            file,
        )

    # start a README
    with open(os.path.join(log_dir, "README.md"), "w") as file:
        file.write(f"# {exp_name}, {time_str}\n")

    return log_dir, base_log_dir


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


def create_loggers(base_log_dir, elite_map_config, agent_configs):
    """Creates the loggers for the algorithm object.
    Args:
        base_log_dir: log directory path without LSI_LOG_DIR prepended
        elite_map_config: toml config object of the feature maps
        agent_configs: toml config object of the agents used
    """
    running_individual_log = RunningIndividualLog(
        os.path.join(base_log_dir, "individuals_log.csv"), elite_map_config,
        agent_configs)
    frequent_map_log = FrequentMapLog(
        os.path.join(base_log_dir, "elite_map.csv"),
        len(elite_map_config["Map"]["Features"]),
    )
    map_summary_log = MapSummaryLog(
        os.path.join(base_log_dir, "map_summary.csv"))
    return running_individual_log, frequent_map_log, map_summary_log


def create_algorithm(base_log_dir, num_simulations, elite_map_config,
                     agent_configs, algorithm_config, lvl_size):
    """Creates a new algorithm instance.

    Args:
        base_log_dir: log directory path without LSI_LOG_DIR prepended
        num_simulations (int): total number of evaluations of QD algorithm to run
        elite_map_config: toml config object of the feature maps
        agent_configs (list): list of toml config object of agents
        algorithm_config: toml config object of the QD algorithm
    """
    feature_map = FeatureMap(
        num_simulations,
        feature_ranges=[(bc["low"], bc["high"])
                        for bc in elite_map_config["Map"]["Features"]],
        resolutions=[
            bc["resolution"] for bc in elite_map_config["Map"]["Features"]
        ],
    )

    (running_individual_log, frequent_map_log,
     map_summary_log) = create_loggers(base_log_dir, elite_map_config,
                                       agent_configs)

    algorithm_name = algorithm_config["name"]

    # take the max of all num params of all agent configs
    num_params = 0
    for agent_config in agent_configs:
        num_params = max(num_params, agent_config["Search"]["num_param"])
    if algorithm_name == "MAPELITES":
        print("Start Running MAPELITES")
        mutation_power = algorithm_config["mutation_power"]
        initial_population = algorithm_config["initial_population"]
        # pylint: disable=no-member
        algorithm = MapElitesAlgorithm(mutation_power, initial_population,
                                       num_simulations, feature_map,
                                       running_individual_log,
                                       frequent_map_log, map_summary_log,
                                       num_params)
    elif algorithm_name == "RANDOM":
        print("Start Running RANDOM")
        # pylint: disable=no-member
        algorithm = RandomGenerator(num_simulations, feature_map,
                                    running_individual_log, frequent_map_log,
                                    map_summary_log, num_params)
    elif algorithm_name == "CMAME":
        print("Start CMA-ME")
        mutation_power = algorithm_config["mutation_power"]
        pop_size = algorithm_config["population_size"]
        algorithm = CMA_ME_Algorithm(mutation_power, num_simulations, pop_size,
                                     feature_map, running_individual_log,
                                     frequent_map_log, map_summary_log,
                                     num_params)
    elif algorithm_name == "MAPELITES-BASE":
        print("Start Running MAPELITES-BASE")
        mutation_k = algorithm_config["mutation_k"]
        mutation_power = algorithm_config["mutation_power"]
        initial_population = algorithm_config["initial_population"]
        algorithm = MapElitesBaselineAlgorithm(
            mutation_k, mutation_power, initial_population, num_simulations,
            feature_map, running_individual_log, frequent_map_log,
            map_summary_log, num_params, lvl_size)

    # Super hacky! This is where we add bounded constraints for the human model.
    if num_params > 32:
        for i in range(32, num_params):
            algorithm.add_bound_constraint(i, (0.0, 1.0))

    return algorithm


def search(dask_client, num_simulations, algorithm_config, elite_map_config,
           agent_configs, model_path, visualize, num_cores, lvl_size,
           reload_saver, algorithm):
    """Run search with the specified algorithm and elite map
    Args:
        dask_client (dask.distributed.Client): client for accessing a Dask
            cluster
        num_simulations (int): total number of evaluations of QD algorithm to run
        algorithm_config: toml config object of QD algorithm
        elite_map_config: toml config object of the feature maps
        agent_configs (list): list of toml config object of agents
        model_path (string): file path to the GAN model
        visualize (bool): render the game or not
        num_cores (int): number of processes to run
        reload_saver (callable): function which saves data necessary for a
            reload to a pickle file
        algorithm (QDAlgorithmBase): QD algorithm instance, either created from
            scratch or reloaded from a checkpoint
        lvl_size (tuple): size of the level to generate. Currently only supports
            (6, 9) and (10, 15)
    """
    start_time = time.time()

    # GAN data
    G_params = read_gan_param()
    gan_state_dict = torch.load(model_path,
                                map_location=lambda storage, loc: storage)

    # initialize the workers with num_cores jobs
    evaluations = []
    active_evals = 0

    def _request_evals(before_loop):
        """Submits more evaluations for the algorithm.
        Args:
            before_loop (bool): Whether this is being called before the for loop
                below. If in the loop, the `evaluations` list has been converted
                into a Dask `as_completed` list.
        """
        nonlocal active_evals
        while active_evals < num_cores and not algorithm.is_blocking():
            print("Starting simulation: ", end="")
            new_ind = algorithm.generate_individual()
            future = dask_client.submit(
                run_overcooked_eval,
                new_ind,
                visualize,
                elite_map_config,
                agent_configs,
                algorithm_config,
                G_params,
                gan_state_dict,
                algorithm.individuals_disbatched,
                lvl_size,
            )
            active_evals += 1
            if before_loop:
                evaluations.append(future)  # pylint: disable=no-member
            else:
                evaluations.add(future)
            print(f"{algorithm.individuals_disbatched}/{num_simulations}")
        print(f"Active evaluations: {active_evals}")

    _request_evals(True)
    evaluations = dask.distributed.as_completed(evaluations)
    print(f"Started {active_evals} simulations")

    # completion time of the latest simulation
    last_eval = time.time()

    # repeatedly grab completed evaluations, return them to the algorithm, and
    # send out new evaluations
    for completion in evaluations:
        # process the individual
        active_evals -= 1
        try:
            evaluated_ind = completion.result()

            if evaluated_ind is None:
                print("Received a failed evaluation.")
                # algorithm.individuals_disbatched -= 1
            elif (evaluated_ind is not None
                  and algorithm.insert_if_still_running(evaluated_ind)):
                cur_time = time.time()
                print("Finished simulation.\n"
                      f"Total simulations done: "
                      f"{algorithm.individuals_evaluated}/{num_simulations}\n"
                      f"Time since last simulation: {cur_time - last_eval}s\n"
                      f"Active evaluations: {active_evals}")
                last_eval = cur_time
                if algorithm.individuals_evaluated % RELOAD_FREQ == 0:
                    reload_saver(algorithm)
        except dask.distributed.scheduler.KilledWorker as err:  # pylint: disable=no-member
            # worker may fail due to, for instance, memory
            print("Worker failed with the following error; continuing anyway\n"
                  "-------------------------------------------\n"
                  f"{err}\n"
                  "-------------------------------------------")
            # avoid not sending out more evaluations while evaluating initial
            # populations
            # algorithm.individuals_disbatched -= 1
            continue

        del completion  # clean up

        if algorithm.is_running():
            # request more evaluations if still running
            _request_evals(False)
        else:
            # otherwise, terminate
            break

    finish_time = time.time()
    print("Total evaluation time:", str(finish_time - start_time), "seconds")


def run(config, reload, lvl_size, model_path):
    """Read in toml config files and run the search.

    Args:
        config (toml): toml config path of current experiment
        reload (str): path to pickle file for reloading experiment
        lvl_size (tuple of int): Size of the level
        model_path (str): file path to the GAN model
    """
    # configs are same regardless of reload
    (experiment_config, algorithm_config, elite_map_config,
     agent_configs) = read_in_lsi_config(config)

    # algorithm, log_dir, and base_log_dir are reused if we are reloading
    if reload is None:
        log_dir, base_log_dir = init_logging_dir(config, experiment_config,
                                                 algorithm_config,
                                                 elite_map_config,
                                                 agent_configs)
        algorithm = create_algorithm(base_log_dir,
                                     experiment_config["num_simulations"],
                                     elite_map_config, agent_configs,
                                     algorithm_config, lvl_size)
    else:
        with open(reload, "rb") as file:
            data = pickle.load(file)
        log_dir = data["log_dir"]
        base_log_dir = data["base_log_dir"]
        algorithm = data["algorithm"]

    def reload_saver(algorithm_):
        """Saves the algorithm to reload.pkl in the logging directory."""
        with open(os.path.join(log_dir, "reload.pkl"), "wb") as file:
            pickle.dump(
                {
                    "algorithm": algorithm_,
                    "base_log_dir": base_log_dir,
                    "log_dir": log_dir,
                }, file)

    print("LOGGING DIRECTORY:", log_dir)

    search(
        init_dask(experiment_config, log_dir),
        experiment_config["num_simulations"],
        algorithm_config,
        elite_map_config,
        agent_configs,
        model_path,
        experiment_config["visualize"],
        experiment_config["num_cores"],
        lvl_size,
        reload_saver,
        algorithm,
    )


def retrieve_lvl_size(size_version):
    """Retrieves level size and path to the corresponding GAN."""
    if size_version == "small":
        return (6, 9), os.path.join(GAN_TRAINING_DIR,
                                    "netG_epoch_49999_999_small.pth")
    if size_version == "large":
        return (10, 15), os.path.join(GAN_TRAINING_DIR,
                                      "netG_epoch_49999_999_large.pth")
    raise NotImplementedError(f"Unrecognized size_version {size_version}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c',
        '--config',
        help='path of experiment config file',
        required=False,
        default=os.path.join(LSI_CONFIG_EXP_DIR, "MAPELITES_demo.tml"),
    )
    parser.add_argument(
        '-r',
        '--reload',
        help='path to pickle file for reloading experiment',
        required=False,
        default=None,
    )
    parser.add_argument(
        '-s',
        '--size_version',
        type=str,
        default="small",
        help=("Size of the level\n"
              '"small" for (6, 9),\n'
              '"large" for (10, 15)'),
    )
    # parser.add_argument('-m',
    #                     '--model_path',
    #                     help='path of the GAN trained',
    #                     required=False,
    #                     default=os.path.join(GAN_TRAINING_DIR,
    #                                          "netG_epoch_49999_999.pth"))
    opt = parser.parse_args()
    run(opt.config, opt.reload, *retrieve_lvl_size(opt.size_version))
