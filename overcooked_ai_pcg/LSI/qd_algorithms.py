"""Quality-Diversity algorithms and their components."""
import dataclasses
from abc import ABC, abstractmethod

import math
import random
import copy
import numpy as np
from numpy.linalg import eig
from overcooked_ai_pcg.helper import gen_int_rnd_lvl, obj_types, num_obj_type


@dataclasses.dataclass
class Individual:
    """Data for a single individual in a QD algorithm."""
    features = None  # BC's
    param_vector = None  # genotype
    level = None  # an Overcooked game level (string version)
    unrepaired_lvl = None  # an Overcooked unrepaired game level (int version)
    ID = None  # ID of the individual after being inserted to the map
    fitness = None  # fitness in the level = score - timestep
    fitnesses: tuple = None  # fitness score of two runs the level
    scores: tuple = None  # raw score of two runs on the level
    # proportional to the number of soup delivered
    player_workloads: tuple = None  # list of list of dic that summarize workload
    # of all players
    checkpoints: tuple = None  # the timestep that the players deliver an order
    joint_actions: tuple = None  # action sequence of the players
    # human model params
    human_preference = None  # float from 0 to 1;
    # human's subtask preference
    human_adaptiveness = None  # float from 0 to 1;
    # human's flexability to adapte to gameplay
    rand_seed = 0  # random seed for agent's random action choice
    concurr_active: tuple = None  # times that both agents are moving
    stuck_time: tuple = None  # times when agents are stuck


class DecompMatrix:
    def __init__(self, dimension):
        self.C = np.eye(dimension, dtype=np.float_)
        self.eigenbasis = np.eye(dimension, dtype=np.float_)
        self.eigenvalues = np.ones((dimension, ), dtype=np.float_)
        self.condition_number = 1
        self.invsqrt = np.eye(dimension, dtype=np.float_)

    def update_eigensystem(self):
        for i in range(len(self.C)):
            for j in range(i):
                self.C[i, j] = self.C[j, i]

        self.eigenvalues, self.eigenbasis = eig(self.C)
        self.eigenvalues = np.real(self.eigenvalues)
        self.eigenbasis = np.real(self.eigenbasis)
        self.condition_number = max(self.eigenvalues) / min(self.eigenvalues)

        for i in range(len(self.C)):
            for j in range(i + 1):
                self.invsqrt[i, j] = self.invsqrt[j, i] = sum(
                    self.eigenbasis[i, k] * self.eigenbasis[j, k] /
                    self.eigenvalues[k]**0.5 for k in range(len(self.C)))


class FeatureMap:
    def __init__(self, max_individuals, feature_ranges, resolutions):
        self.max_individuals = max_individuals
        self.feature_ranges = feature_ranges
        self.resolutions = resolutions

        self.elite_map = {}
        self.elite_indices = []

        self.num_individuals_added = 0

    def get_feature_index(self, feature_id, feature):
        feature_range = self.feature_ranges[feature_id]
        if feature - 1e-9 <= feature_range[0]:
            return 0
        if feature_range[1] <= feature + 1e-9:
            return self.resolutions[feature_id] - 1

        gap = feature_range[1] - feature_range[0]
        pos = feature - feature_range[0]
        index = int((self.resolutions[feature_id] * pos + 1e-9) / gap)
        return index

    def get_index(self, cur):
        #if len(cur.features[0]) > 1:
        if isinstance(cur.features[0], list):  #need to check if it is a list
            tmp = np.array(cur.features)
            return tuple(
                self.get_feature_index(i, f) for i, f in enumerate(tmp[:, -1]))
        else:
            return tuple(
                self.get_feature_index(i, f)
                for i, f in enumerate(cur.features))

    def add_to_map(self, to_add):
        index = self.get_index(to_add)
        if isinstance(to_add.fitness, list) and len(to_add.fitness) > 1:
            replaced_elite = False
            if index not in self.elite_map:
                self.elite_indices.append(index)
                self.elite_map[index] = to_add
                replaced_elite = True
                to_add.delta = (1, to_add.fitness[-1])
            elif self.elite_map[index].fitness[-1] < to_add.fitness[-1]:
                to_add.delta = (0, to_add.fitness[-1] -
                                self.elite_map[index].fitness[-1])
                self.elite_map[index] = to_add
                replaced_elite = True
        else:
            replaced_elite = False
            if index not in self.elite_map:
                self.elite_indices.append(index)
                self.elite_map[index] = to_add
                replaced_elite = True
                to_add.delta = (1, to_add.fitness)
            elif self.elite_map[index].fitness < to_add.fitness:
                to_add.delta = (0,
                                to_add.fitness - self.elite_map[index].fitness)
                self.elite_map[index] = to_add
                replaced_elite = True

        return replaced_elite

    def add(self, to_add):
        self.num_individuals_added += 1
        replaced_elite = self.add_to_map(to_add)
        return replaced_elite

    def get_random_elite(self):
        pos = np.random.randint(0, len(self.elite_indices))
        index = self.elite_indices[pos]
        return self.elite_map[index]


class QDAlgorithmBase(ABC):
    """Base class for all QD algorithms.

    Args:
        feature_map (FeatureMap): A container for storing solutions.
        running_individual_log (RunningIndividualLog): Previously constructed
            logger.
        frequent_map_log (FrequentMapLog): Previously constructed logger.
        map_summary_log (MapSummaryLog): Previously constructed logger.
    """
    def __init__(self, feature_map, running_individual_log, frequent_map_log,
                 map_summary_log):
        self.feature_map = feature_map
        self.individuals_disbatched = 0
        self.individuals_evaluated = 0
        self.running_individual_log = running_individual_log
        self.frequent_map_log = frequent_map_log
        self.map_summary_log = map_summary_log
        self.bound_constraints = []

    @abstractmethod
    def is_running(self):
        pass

    @abstractmethod
    def is_blocking(self):
        pass

    def add_bound_constraint(self, index, bounds):

        if len(self.bound_constraints) <= index:
            new_bound_constraints = [
                self.bound_constraints[i] if i < len(self.bound_constraints) else None \
                for i in range(index+1)
            ]
            self.bound_constraints = new_bound_constraints

        self.bound_constraints[index] = bounds

    @abstractmethod
    def generate_individual(self):
        pass

    @abstractmethod
    def return_evaluated_individual(self, ind):
        pass

    def insert_if_still_running(self, ind):
        """
        Adds the individual to the algorithm container if the algorithm is still
        running and logs the results.

        Returns:
            running (bool): Whether the algorithm is still running.
            individuals_evaluated (int): The number of individuals that have
                been evaluated so far. None if the algorithm is no longer
                running.
        """
        if self.is_running():
            self.return_evaluated_individual(ind)
            if isinstance(ind.fitness, list):
                self.running_individual_log.log_individual_multi_row(
                    ind)  #for stochastic human agent models
            else:
                self.running_individual_log.log_individual(ind)
            self.frequent_map_log.log_map(self.feature_map)
            self.map_summary_log.log_summary(self.feature_map,
                                             self.individuals_evaluated)
            return True, self.individuals_evaluated
        return False, None


class MapElitesAlgorithm(QDAlgorithmBase):
    def __init__(self,
                 mutation_power,
                 initial_population,
                 num_to_evaluate,
                 feature_map,
                 running_individual_log,
                 frequent_map_log,
                 map_summary_log,
                 num_params=32):
        super().__init__(feature_map, running_individual_log, frequent_map_log,
                         map_summary_log)
        self.num_to_evaluate = num_to_evaluate
        self.initial_population = initial_population
        self.mutation_power = mutation_power
        self.num_params = num_params

    def is_running(self):
        return self.individuals_evaluated < self.num_to_evaluate

    def is_blocking(self):
        return (self.individuals_disbatched == self.initial_population
                and self.individuals_evaluated < self.initial_population / 2)

    def generate_individual(self):
        ind = Individual()
        if self.individuals_disbatched < self.initial_population:
            ind.param_vector = np.random.normal(0.0, 1.0, self.num_params)

            for i in range(len(self.bound_constraints)):
                if self.bound_constraints[i] != None:
                    min_val, max_val = self.bound_constraints[i]
                    ind.param_vector[i] = np.clip(ind.param_vector[i], min_val,
                                                  max_val)

        else:
            parent = self.feature_map.get_random_elite()
            ind.param_vector = parent.param_vector + np.random.normal(
                0.0, self.mutation_power, self.num_params)

            for i in range(len(self.bound_constraints)):
                if self.bound_constraints[i] != None:
                    min_val, max_val = self.bound_constraints[i]
                    ind.param_vector[i] = np.clip(ind.param_vector[i], min_val,
                                                  max_val)

        self.individuals_disbatched += 1
        return ind

    def return_evaluated_individual(self, ind):
        ind.ID = self.individuals_evaluated
        self.individuals_evaluated += 1
        self.feature_map.add(ind)


class MapElitesBaselineAlgorithm(QDAlgorithmBase):
    def __init__(self,
                 mutation_k,
                 mutation_power,
                 initial_population,
                 num_to_evaluate,
                 feature_map,
                 running_individual_log,
                 frequent_map_log,
                 map_summary_log,
                 num_params=32,
                 lvl_size=(10, 15)):
        super().__init__(feature_map, running_individual_log, frequent_map_log,
                         map_summary_log)
        self.num_to_evaluate = num_to_evaluate
        self.initial_population = initial_population
        self.mutation_k = mutation_k
        self.mutation_power = mutation_power
        self.num_params = num_params
        self.lvl_size = lvl_size
        assert np.product(self.lvl_size) >= self.mutation_k

    def is_running(self):
        return self.individuals_evaluated < self.num_to_evaluate

    def is_blocking(self):
        return (self.individuals_disbatched == self.initial_population
                and self.individuals_evaluated < self.initial_population / 2)

    def generate_individual(self):
        ind = Individual()
        if self.individuals_disbatched < self.initial_population:
            ind.unrepaired_lvl = gen_int_rnd_lvl(self.lvl_size)

            # genrate human params directly
            ind.human_preference, ind.human_adaptiveness = \
                np.random.normal(0.0, 1.0, 2)
        else:
            parent = self.feature_map.get_random_elite()
            ind.unrepaired_lvl = copy.deepcopy(parent.unrepaired_lvl)
            # select k spots randomly without replacement
            # and replace them with random objects types
            to_mutate = random.sample([(x, y) for x in range(self.lvl_size[0])
                                       for y in range(self.lvl_size[1])],
                                      self.mutation_k)
            for x, y in to_mutate:
                ind.unrepaired_lvl[x][y] = np.random.randint(num_obj_type)

            # mutate human params directly
            ind.human_preference = parent.human_preference + \
                np.random.normal(0.0, self.mutation_power)
            ind.human_adaptiveness = parent.human_adaptiveness + \
                np.random.normal(0.0, self.mutation_power)

        ind.human_preference = np.clip(ind.human_preference, 0.0, 1.0)
        ind.human_adaptiveness = np.clip(ind.human_adaptiveness, 0.0, 1.0)

        self.individuals_disbatched += 1
        return ind

    def return_evaluated_individual(self, ind):
        ind.ID = self.individuals_evaluated
        self.individuals_evaluated += 1
        self.feature_map.add(ind)


class RandomGenerator(QDAlgorithmBase):
    def __init__(self,
                 num_to_evaluate,
                 feature_map,
                 running_individual_log,
                 frequent_map_log,
                 map_summary_log,
                 num_params=32):
        super().__init__(feature_map, running_individual_log, frequent_map_log,
                         map_summary_log)
        self.num_to_evaluate = num_to_evaluate
        self.num_params = num_params

    def is_running(self):
        return self.individuals_evaluated < self.num_to_evaluate

    def is_blocking(self):
        return False

    def generate_individual(self):
        ind = Individual()
        unscaled_params = np.random.normal(0.0, 1.0, self.num_params)
        for i in range(len(self.bound_constraints)):
            if self.bound_constraints[i] != None:
                min_val, max_val = self.bound_constraints[i]
                unscaled_params[i] = np.clip(unscaled_params[i], min_val,
                                             max_val)

        ind.param_vector = unscaled_params
        self.individuals_disbatched += 1
        return ind

    def return_evaluated_individual(self, ind):
        ind.ID = self.individuals_evaluated
        self.individuals_evaluated += 1
        self.feature_map.add(ind)


class ImprovementEmitter:
    def __init__(self, mutation_power, population_size, feature_map,
                 num_params):
        self.population_size = population_size
        self.sigma = mutation_power
        self.individuals_disbatched = 0
        self.individuals_evaluated = 0
        self.individuals_released = 0
        self.generation = 0
        self.num_params = num_params

        self.parents = []
        self.population = []
        self.feature_map = feature_map
        self.bound_constraints = []

        self.reset()

    def reset(self):
        self.mutation_power = self.sigma
        if len(self.feature_map.elite_map) == 0:
            self.mean = np.asarray([0.0] * self.num_params)
        else:
            self.mean = self.feature_map.get_random_elite().param_vector

        # Setup evolution path variables
        self.pc = np.zeros((self.num_params, ), dtype=np.float_)
        self.ps = np.zeros((self.num_params, ), dtype=np.float_)

        # Setup the covariance matrix
        self.C = DecompMatrix(self.num_params)

        # Reset the individuals evaluated
        self.individuals_evaluated = 0

    def check_stop(self, parents):
        if self.C.condition_number > 1e14:
            return True

        area = self.mutation_power * math.sqrt(max(self.C.eigenvalues))

        if area < 1e-11:
            return True
        if isinstance(parents[0].fitness, list):
            if abs(parents[0].fitness[-1] - parents[-1].fitness[-1]) < 1e-12:
                return True
        else:
            if abs(parents[0].fitness - parents[-1].fitness) < 1e-12:
                return True

        return False

    def is_blocking(self):
        return self.individuals_disbatched > self.population_size * 1.1

    def add_bound_constraint(self, index, bounds):

        if len(self.bound_constraints) <= index:
            new_bound_constraints = [
                self.bound_constraints[i] if i < len(self.bound_constraints) else None \
                for i in range(index+1)
            ]
            self.bound_constraints = new_bound_constraints

        self.bound_constraints[index] = bounds

    def generate_individual(self):

        # Resampling method for bound constraints
        while True:
            unscaled_params = np.random.normal(0.0, self.mutation_power, self.num_params) \
                            * np.sqrt(self.C.eigenvalues)
            unscaled_params = np.matmul(self.C.eigenbasis, unscaled_params)
            unscaled_params = self.mean + np.array(unscaled_params)

            is_within_bounds = True
            for i in range(len(self.bound_constraints)):
                if self.bound_constraints[i] != None:
                    min_val, max_val = self.bound_constraints[i]
                    if unscaled_params[i] < min_val or unscaled_params[
                            i] > max_val:
                        is_within_bounds = False
                        break

            if is_within_bounds:
                break

        ind = Individual()
        ind.param_vector = unscaled_params
        ind.generation = self.generation

        self.individuals_disbatched += 1
        self.individuals_released += 1

        return ind

    def return_evaluated_individual(self, ind):
        self.population.append(ind)
        self.individuals_evaluated += 1
        if self.feature_map.add(ind):
            self.parents.append(ind)
        if ind.generation != self.generation:
            return

        if len(self.population) < self.population_size:
            return

        # Only filter by this generation
        num_parents = len(self.parents)
        needs_restart = num_parents == 0

        # Only update if there are parents
        if num_parents > 0:
            parents = sorted(self.parents, key=lambda x: x.delta)[::-1]

            # Create fresh weights for the number of elites found
            weights = [math.log(num_parents + 0.5) \
                    - math.log(i+1) for i in range(num_parents)]
            total_weights = sum(weights)
            weights = np.array([w / total_weights for w in weights])

            # Dynamically update these parameters
            mueff = sum(weights)**2 / sum(weights**2)
            cc = (4 + mueff / self.num_params) / (self.num_params + 4 +
                                                  2 * mueff / self.num_params)
            cs = (mueff + 2) / (self.num_params + mueff + 5)
            c1 = 2 / ((self.num_params + 1.3)**2 + mueff)
            cmu = min(
                1 - c1, 2 * (mueff - 2 + 1 / mueff) /
                ((self.num_params + 2)**2 + mueff))
            damps = 1 + 2 * max(
                0,
                math.sqrt((mueff - 1) / (self.num_params + 1)) - 1) + cs
            chiN = self.num_params**0.5 * (1 - 1 / (4 * self.num_params) + 1. /
                                           (21 * self.num_params**2))

            # Recombination of the new mean
            old_mean = self.mean
            self.mean = sum(ind.param_vector * w
                            for ind, w in zip(parents, weights))

            # Update the evolution path
            y = self.mean - old_mean
            z = np.matmul(self.C.invsqrt, y)
            self.ps = (1-cs) * self.ps +\
                (math.sqrt(cs * (2 - cs) * mueff) / self.mutation_power) * z
            left = sum(x**2 for x in self.ps) / self.num_params \
                / (1-(1-cs)**(2*self.individuals_evaluated / self.population_size))
            right = 2 + 4. / (self.num_params + 1)
            hsig = 1 if left < right else 0

            self.pc = (1-cc) * self.pc + \
                hsig * math.sqrt(cc*(2-cc)*mueff) * y

            # Adapt the covariance matrix
            c1a = c1 * (1 - (1 - hsig**2) * cc * (2 - cc))
            self.C.C *= (1 - c1a - cmu)
            self.C.C += c1 * np.outer(self.pc, self.pc)
            for k, w in enumerate(weights):
                dv = parents[k].param_vector - old_mean
                self.C.C += w * cmu * np.outer(dv, dv) / (self.mutation_power**
                                                          2)

            # Update the covariance matrix decomposition and inverse
            if self.check_stop(parents):
                needs_restart = True
            else:
                self.C.update_eigensystem()

            # Update sigma
            cn, sum_square_ps = cs / damps, sum(x**2 for x in self.ps)
            self.mutation_power *= math.exp(
                min(1,
                    cn * (sum_square_ps / self.num_params - 1) / 2))

        if needs_restart:
            self.reset()

        # Reset the population
        self.individuals_disbatched = 0
        self.generation += 1
        self.population.clear()
        self.parents.clear()


class CMA_ME_Algorithm(QDAlgorithmBase):
    def __init__(self,
                 mutation_power,
                 num_to_evaluate,
                 population_size,
                 feature_map,
                 running_individual_log,
                 frequent_map_log,
                 map_summary_log,
                 num_params=32):

        super().__init__(feature_map, running_individual_log, frequent_map_log,
                         map_summary_log)

        self.num_to_evaluate = num_to_evaluate
        self.individuals_disbatched = 0
        self.individuals_evaluated = 0
        self.feature_map = feature_map
        self.mutation_power = mutation_power
        self.population_size = population_size

        self.emitters = [ImprovementEmitter(self.mutation_power, self.population_size, \
                         self.feature_map, num_params) for i in range(5)]

    def is_running(self):
        return self.individuals_evaluated < self.num_to_evaluate

    def is_blocking(self):
        # If any of our emitters are not blocking, we are not blocking
        for emitter in self.emitters:
            if not emitter.is_blocking():
                return False
        return True

    def add_bound_constraint(self, index, bounds):
        for emitter in self.emitters:
            emitter.add_bound_constraint(index, bounds)

    def generate_individual(self):
        pos = 0
        emitter = None
        for i in range(len(self.emitters)):
            if not self.emitters[i].is_blocking() and \
               (emitter == None or \
                emitter.individuals_released > self.emitters[i].individuals_released):
                emitter = self.emitters[i]
                pos = i

        ind = emitter.generate_individual()
        ind.emitter_id = pos

        self.individuals_disbatched += 1
        return ind

    def return_evaluated_individual(self, ind):
        ind.ID = self.individuals_evaluated
        self.individuals_evaluated += 1

        if ind.emitter_id == -1:
            self.feature_map.add(ind)
        else:
            self.emitters[ind.emitter_id].return_evaluated_individual(ind)
