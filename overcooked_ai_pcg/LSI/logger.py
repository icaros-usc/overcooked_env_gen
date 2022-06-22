"""Contains logger classes."""
import csv
import os
from abc import ABC, abstractmethod

import numpy as np
from overcooked_ai_pcg import LSI_LOG_DIR


class LoggerBase(ABC):
    @abstractmethod
    def init_log(self, ):
        pass

    def _write_row(self, to_add):
        """Append a row to csv file"""
        with open(self._log_path, 'a+') as f:
            writer = csv.writer(f)
            writer.writerow(to_add)
            f.close()

    def _read_row(self):
        with open(self._log_path, 'r') as f:
            # Read all the data from the csv file
            allRows = list(csv.reader(f, delimiter=','))


class RunningIndividualLog(LoggerBase):
    """
    Logger that logs individuals to a csv file

    Args:
        log_path (string): filename of the log file
        elite_map_config: toml config object of the feature maps
        agent_configs (list): list of toml config object of agents
    """
    def __init__(self, log_path, elite_map_config, agent_configs):
        super().__init__()
        self._log_path = os.path.join(LSI_LOG_DIR, log_path)
        self._isInitialized = False
        self._elite_map_config = elite_map_config
        self._agent_configs = agent_configs
        self.init_log()

    def init_log(self, ):
        # remove the file if exists
        if os.path.exists(self._log_path):
            os.remove(self._log_path)

        # construct labels
        data_labels = ["ID", "fitness"]
        # We need to be told how many orders we have

        data_labels += [
            'fitness_{}_w_{}'.format(agent_config["Agent1"]["name"],
                                     agent_config["Agent2"]["name"])
            for agent_config in self._agent_configs
        ]
        data_labels += [
            'score_{}_w_{}'.format(agent_config["Agent1"]["name"],
                                   agent_config["Agent2"]["name"])
            for agent_config in self._agent_configs
        ]
        data_labels += [
            'order_delivered_{}_w_{}'.format(agent_config["Agent1"]["name"],
                                             agent_config["Agent2"]["name"])
            for agent_config in self._agent_configs
        ]
        data_labels += [
            'player_workload_{}_w_{}'.format(agent_config["Agent1"]["name"],
                                             agent_config["Agent2"]["name"])
            for agent_config in self._agent_configs
        ]

        for bc in self._elite_map_config["Map"]["Features"]:
            data_labels.append(bc["name"])
        data_labels += [
            "human_preference", "human_adaptiveness", "rand_seed", "lvl_str"
        ]
        data_labels += [
            'joint_action{}_w_{}'.format(agent_config["Agent1"]["name"],
                                         agent_config["Agent2"]["name"])
            for agent_config in self._agent_configs
        ]
        self._write_row(data_labels)

    def log_individual(self, ind):
        to_add = [
            ind.ID,
            ind.fitness,
            *ind.fitnesses,
            *ind.scores,
            *ind.checkpoints,
            *ind.player_workloads,
            *ind.features,
            ind.human_preference,
            ind.human_adaptiveness,
            ind.rand_seed,
            ind.level,
            *ind.joint_actions,
        ]
        self._write_row(to_add)

    def log_individual_multi_row(self, ind):
        ind.fitnesses = np.array(ind.fitnesses)
        ind.scores = np.array(ind.scores)
        ind.checkpoints = np.array(ind.checkpoints)
        ind.player_workloads = np.array(ind.player_workloads)
        ind.features = np.array(ind.features)
        ind.joint_actions = np.array(ind.joint_actions, dtype=object)

        to_add = [
            ind.ID,
            ind.fitness[-1],  #avgerage fitness
            *ind.fitnesses[:, -1],
            *ind.scores[:, -1],
            *ind.checkpoints[:, -1],
            *ind.player_workloads[:, -1],
            *ind.features[:, -1],
            '',
            '',
            ind.rand_seed,
            ind.level,
        ]
        self._write_row(to_add)

        for i in range(len(ind.fitness) - 1):
            to_add = [
                '',
                ind.fitness[i],
                *ind.fitnesses[:, i],
                *ind.scores[:, i],
                *ind.checkpoints[:, i],
                *ind.player_workloads[:, i],
                *ind.features[:, i],
                ind.human_preference,
                ind.human_adaptiveness,
                '',
                '',
                *ind.joint_actions[:, i],
            ]
            self._write_row(to_add)


class FrequentMapLog(LoggerBase):
    """
    Logger that logs compressed feature map information

    Args:
        log_path (string): filename of the log file
        num_features (int): number of behavior characteristics
    """
    def __init__(self, log_path, num_features):
        super().__init__()
        self._log_path = os.path.join(LSI_LOG_DIR, log_path)
        self.init_log(num_features)

    def init_log(self, num_features):
        # remove the file if exists
        if os.path.exists(self._log_path):
            os.remove(self._log_path)

        # construct label
        feature_label = ":".join(
            ["feature" + str(i + 1) for i in range(num_features)])
        dimension_label = ":".join(
            ["f" + str(i + 1) for i in range(num_features)])
        data_labels = [
            "Dimension",
            dimension_label + ":IndividualID:Fitness:" + feature_label
        ]
        self._write_row(data_labels)

    def log_map(self, feature_map):
        to_add = []
        to_add.append("x".join(str(num) for num in feature_map.resolutions), )
        for index in feature_map.elite_indices:
            ind = feature_map.elite_map[index]
            if isinstance(ind.fitness, list) and len(ind.fitness) > 1:
                curr = [
                    *index,
                    ind.ID,
                    ind.fitness[-1],
                    *ind.features[:, -1],
                ]
            else:
                curr = [
                    *index,
                    ind.ID,
                    ind.fitness,
                    *ind.features,
                ]
            to_add.append(":".join(str(ele) for ele in curr))
        self._write_row(to_add)


class MapSummaryLog(LoggerBase):
    """
    Logger that logs general feature map info to a csv file

    Args:
        log_path (string): filename of the log file
    """
    def __init__(self, log_path):
        super().__init__()
        self._log_path = os.path.join(LSI_LOG_DIR, log_path)
        self.init_log()

    def init_log(self, ):
        # remove the file if exists
        if os.path.exists(self._log_path):
            os.remove(self._log_path)

        data_labels = [
            "NumEvaluated",
            "QD-Score",
            "MeanNormFitness",
            "MedianNormFitness",
            "MaxNormFitness",
            "CellsOccupied",
            "PercentOccupied",
        ]
        self._write_row(data_labels)

    def log_summary(self, feature_map, num_evaluated):
        all_fitness = []
        for index in feature_map.elite_indices:
            ind = feature_map.elite_map[index]
            all_fitness.append(ind.fitness)
        cells_occupied = len(feature_map.elite_indices)
        QD_score = np.sum(all_fitness)
        mean_fitness = np.average(all_fitness)
        median_fitness = np.median(all_fitness)
        max_fitness = np.max(all_fitness)
        num_cell = np.prod(feature_map.resolutions)
        percent_occupied = 100 * cells_occupied / num_cell

        to_add = [
            num_evaluated,
            QD_score,
            mean_fitness,
            median_fitness,
            max_fitness,
            cells_occupied,
            percent_occupied,
        ]
        self._write_row(to_add)


class HumanExpLogger(LoggerBase):
    """
    Logger that logs individuals to a csv file

    Args:
        log_path (string): filename of the log file
        elite_map_config: toml config object of the feature maps
        agent_configs (list): list of toml config object of agents
    """
    def __init__(self, log_path, elite_map_config, agent_configs):
        super().__init__()
        self._log_path = os.path.join(LSI_LOG_DIR, log_path)
        self._isInitialized = False
        self._elite_map_config = elite_map_config
        self._agent_configs = agent_configs
        self.init_log()

    def init_log(self, ):
        # remove the file if exists
        if os.path.exists(self._log_path):
            os.remove(self._log_path)

        # construct labels
        data_labels = [
            "ID", "feature 1", "feature 2", "row", "column", "fitness"
        ]
        # We need to be told how many orders we have

        data_labels += [
            'score_{}_w_{}'.format(agent_config["Agent1"]["name"],
                                   agent_config["Agent2"]["name"])
            for agent_config in self._agent_configs
        ]
        data_labels += [
            'order_delivered_{}_w_{}'.format(agent_config["Agent1"]["name"],
                                             agent_config["Agent2"]["name"])
            for agent_config in self._agent_configs
        ]
        data_labels += [
            'player_workload_{}_w_{}'.format(agent_config["Agent1"]["name"],
                                             agent_config["Agent2"]["name"])
            for agent_config in self._agent_configs
        ]
        data_labels += [
            'joint_action{}_w_{}'.format(agent_config["Agent1"]["name"],
                                         agent_config["Agent2"]["name"])
            for agent_config in self._agent_configs
        ]

        for bc in self._elite_map_config["Map"]["Features"]:
            data_labels.append(bc["name"])
        data_labels += [
            "human_preference", "human_adaptiveness", "rand_seed", "lvl_str"
        ]
        self._write_row(data_labels)

    def log_human_exp(self, ind, f1, f2, row, col):
        to_add = [
            ind.ID,
            f1,
            f2,
            row,
            col,
            ind.fitness,
            *ind.scores,
            *ind.checkpoints,
            *ind.player_workloads,
            *ind.joint_actions,
            *ind.features,
            ind.human_preference,
            ind.human_adaptiveness,
            ind.rand_seed,
            ind.level,
        ]
        self._write_row(to_add)