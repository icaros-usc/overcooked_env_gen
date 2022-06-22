import os
import copy
import json
import shutil
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from queue import Queue
from pprint import pprint
from overcooked_ai_py import LAYOUTS_DIR, read_layout_dict
from overcooked_ai_pcg.helper import lvl_str2number, lvl_number2str, lvl_str2grid
from overcooked_ai_pcg.LSI.bc_calculate import shortest_dist


def count_items(lvl_str, items):
    total_cnt = 0
    for item in items:
        total_cnt += lvl_str.count(item)
    return total_cnt


def BFS(start_loc, lvl_graph):
    q = Queue()
    q.put(start_loc)
    seen = set()
    dxs = (0, 0, 1, -1)
    dys = (1, -1, 0, 0)
    m = len(lvl_graph)
    n = len(lvl_graph[0])
    dist_matrix = np.full((m, n), np.inf)
    dist_matrix[start_loc] = 0
    block_types = ['X', 'P', 'O', 'S', 'D']
    while not q.empty():
        curr = q.get()
        x, y = curr
        if curr in seen:
            continue
        if lvl_graph[x][y] == "B":
            shortest = dist_matrix[x, y]
            return shortest
        seen.add(curr)
        for dx, dy in zip(dxs, dys):
            n_x = x + dx
            n_y = y + dy
            if n_x < m and n_x >= 0 and \
                n_y < n and n_y >= 0 and \
                lvl_graph[n_x][n_y] not in block_types:
                q.put((n_x, n_y))
                dist_matrix[n_x, n_y] = dist_matrix[x, y] + 1
    return np.inf


def shortest_dist(terrain1, terrain2, lvl_str):
    start_locs = []
    end_locs = []
    lvl_matrix = [list(row) for row in lvl_str.strip().split("\n")]
    m = len(lvl_matrix)
    n = len(lvl_matrix[0])
    assert m == 10
    assert n == 15

    for i, row in enumerate(lvl_matrix):
        for j, tile in enumerate(row):
            if tile == terrain1:
                start_locs.append((i, j))
            elif tile == terrain2:
                end_locs.append((i, j))

    shortest_dist = np.inf
    for start_loc in start_locs:
        for end_loc in end_locs:
            lvl_graph = copy.deepcopy(lvl_matrix)
            lvl_graph[start_loc[0]][start_loc[1]] = "A"
            lvl_graph[end_loc[0]][end_loc[1]] = "B"
            curr_dist = BFS(start_loc, lvl_graph)
            if shortest_dist > curr_dist:
                shortest_dist = curr_dist
    return shortest_dist


def read_in_human_lvls(data_path, sub_dir=None):
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
            lvl_str = lvl_number2str(np_lvl)
            lvls.append(lvl_str)

    return lvls


if __name__ == '__main__':
    to_stats = {
        'milp_only': {
            'color': '#d5a49b',
            'edgecolor': '#d8a99f',
            'label': 'MILP (random)'
        },
        'human_lvls': {
            'color': '#5699c5',
            'edgecolor': '#2d7fb8',
            'label': "Human"
        },
        'gan_only': {
            'color': '#ff9f4a',
            'edgecolor': '#ff871d',
            'label': "GAN"
        },
        'gan_milp': {
            'color': '#60b761',
            'edgecolor': '#3aa539',
            'label': "GAN+MILP"
        },
    }

    to_analyze = {
        "floor": ([" "], "# Floor", (30, 100), (0, 0.08)),
        "counters": (["X"], "# Counters", (30, 100), (0, 0.08)),
        "key_items": (["P", "S", "O", "D"], "# Pots, Serves, Onions, Dishes",
                      (0, 15), (0, 1)),
        "path*PD":
        (["P", "D"], "Min path length from Pots to Dishes", (0, 35), (0, 0.2)),
        "path*PO":
        (["P", "O"], "Min path length from Pots to Onions", (0, 35), (0, 0.2)),
        "path*PS":
        (["P", "S"], "Min path length from Pots to Serves", (0, 35), (0, 0.2)),
    }

    # read in generated levels
    with open('all_lvl_strs.json') as all_lvl_strs_file:
        all_lvl_strs = json.load(all_lvl_strs_file)

    # read in human levels
    sub_dir = "train_gan_large"
    data_path = os.path.join(LAYOUTS_DIR, sub_dir)
    all_lvl_strs["human_lvls"] = read_in_human_lvls(data_path, sub_dir)

    for name, item in tqdm(to_analyze.items()):
        to_analy, label, xlim, ylim = item
        # create dir
        result_dir = f"lvl_dist_analysis_{name}"
        if os.path.isdir(result_dir):
            shutil.rmtree(result_dir)
        os.mkdir(result_dir)

        all_fig, all_ax = plt.subplots(figsize=(10, 6))

        for key, to_stat in to_stats.items():
            if name.split('*')[0] == "path" and key == "gan_only":
                continue

            one_fig, one_ax = plt.subplots(figsize=(10, 6))
            # print(len(all_lvl_strs[key]))
            lvl_metric = []
            for lvl_str in all_lvl_strs[key]:
                if name.split('*')[0] == "path":
                    lvl_metric.append(shortest_dist(*to_analy, lvl_str))
                else:
                    lvl_metric.append(count_items(lvl_str, to_analy))

            sorted_lvl_metric, counts = np.unique(lvl_metric,
                                                  return_counts=True)
            cnt_percentage = counts / len(all_lvl_strs[key])

            all_ax.bar(sorted_lvl_metric,
                       cnt_percentage,
                       color=to_stat["color"],
                       edgecolor=to_stat["edgecolor"],
                       alpha=0.7,
                       width=1,
                       label=to_stat["label"])
            all_ax.set_xlabel(label)

            one_ax.bar(sorted_lvl_metric,
                       cnt_percentage,
                       color=to_stat["color"],
                       edgecolor=to_stat["edgecolor"],
                       alpha=0.7,
                       width=1,
                       label=to_stat["label"])
            one_ax.set_xlabel(label)
            one_ax.legend()
            one_ax.set(xlim=xlim, ylim=ylim)
            one_fig.tight_layout()
            one_fig.savefig(os.path.join(result_dir, f"{name}_{key}.png"))
            plt.close()

        all_ax.legend()
        all_ax.set(xlim=xlim, ylim=ylim)
        all_fig.tight_layout()
        all_fig.savefig(os.path.join(result_dir, f"{name}_all.png"))
        plt.close()