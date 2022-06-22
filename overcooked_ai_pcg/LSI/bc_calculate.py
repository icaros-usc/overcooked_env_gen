import numpy as np
from overcooked_ai_pcg.helper import lvl_str2grid
from queue import Queue


def bc_demo1(ind):
    return np.random.rand() * 5


def bc_demo2(ind):
    return np.random.rand() * 5


def pot_onion_shortest_dist(ind):
    return shortest_dist('P', 'O', ind.level)


def pot_serve_shortest_dist(ind):
    return shortest_dist('P', 'S', ind.level)


def pot_dish_shortest_dist(ind):
    return shortest_dist('P', 'D', ind.level)


def onion_dish_shortest_dist(ind):
    return shortest_dist('O', 'D', ind.level)


def onion_serve_shortest_dist(ind):
    return shortest_dist('O', 'S', ind.level)


def dish_serve_shortest_dist(ind):
    return shortest_dist('D', 'S', ind.level)


def shortest_dist(terrain1, terrain2, lvl_str):
    """
    Use BFS to find the shortest distance between two specified
    terrain types in the level
    """
    shortest = np.inf
    dxs = (0, 0, 1, -1)
    dys = (1, -1, 0, 0)
    lvl_grid = lvl_str2grid(lvl_str)
    m = len(lvl_grid)
    n = len(lvl_grid[0])
    block_types = ['X', 'P', 'O', 'S', 'D']
    block_types.remove(terrain2)
    for i, row in enumerate(lvl_grid):
        for j, terrain in enumerate(row):
            if terrain == terrain1:
                q = Queue()
                seen = set()
                q.put((i, j))
                dist_matrix = np.full((m, n), np.inf)
                dist_matrix[i, j] = 0
                while not q.empty():
                    curr = q.get()
                    x, y = curr
                    if curr in seen:
                        continue
                    if lvl_grid[x][y] == terrain2 and dist_matrix[
                            x, y] < shortest:
                        shortest = dist_matrix[x, y]
                    seen.add(curr)
                    for dx, dy in zip(dxs, dys):
                        n_x = x + dx
                        n_y = y + dy
                        if n_x < m and n_x >= 0 and \
                           n_y < n and n_y >= 0 and \
                           lvl_grid[n_x][n_y] not in block_types:
                            q.put((n_x, n_y))
                            dist_matrix[n_x, n_y] = dist_matrix[x, y] + 1
    return shortest


def diff_num_ingre_held(ind):
    workloads = ind.player_workloads[-1]
    if len(workloads) > 2:
        workloads = np.array(workloads)
        workload_diff = []
        for w0, w1 in zip(workloads[:, 0], workloads[:, 1]):
            workload_diff.append(w0["num_ingre_held"] - w1["num_ingre_held"])

        #replace avg with median of workloads
        median_diff = np.median(workload_diff[:-1])
        workload_diff[-1] = round(median_diff)

        return workload_diff
    else:
        workloads = np.array(workloads[0])
        return workloads[0]["num_ingre_held"] - workloads[1]["num_ingre_held"]


def diff_num_plate_held(ind):
    workloads = ind.player_workloads[-1]
    if len(workloads) > 2:
        workloads = np.array(workloads)
        workload_diff = []
        for w0, w1 in zip(workloads[:, 0], workloads[:, 1]):
            workload_diff.append(w0["num_plate_held"] - w1["num_plate_held"])

        #replace avg with median of workloads
        median_diff = np.median(workload_diff[:-1])
        workload_diff[-1] = round(median_diff)
        return workload_diff
    else:
        workloads = np.array(workloads[0])
        return workloads[0]["num_plate_held"] - workloads[1]["num_plate_held"]


def diff_num_dish_served(ind):
    workloads = ind.player_workloads[-1]
    if len(workloads) > 2:
        workloads = np.array(workloads)
        workload_diff = []
        for w0, w1 in zip(workloads[:, 0], workloads[:, 1]):
            workload_diff.append(w0["num_served"] - w1["num_served"])

        #replace avg with median of workloads
        median_diff = np.median(workload_diff[:-1])
        workload_diff[-1] = round(median_diff)

        return workload_diff
    else:
        workloads = np.array(workloads[0])
        return workloads[0]["num_served"] - workloads[1]["num_served"]


# bc for human awared mdp agent
def human_adaptiveness(ind):
    # print('bc human_adaptiveness =', ind.human_adaptiveness)
    return ind.human_adaptiveness


def human_preference(ind):
    # print('bc human_preference =', ind.human_preference)
    return ind.human_preference


def cc_active(ind):
    return ind.concurr_actives[-1]


def stuck_time(ind):
    return ind.stuck_times[-1]
