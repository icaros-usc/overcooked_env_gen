import os
import ast
import numpy as np
import pandas
import warnings

warnings.simplefilter('always', UserWarning)
from pprint import pprint
from overcooked_ai_pcg import LSI_HUMAN_STUDY_RESULT_DIR

from overcooked_ai_pcg.LSI.human_study import (DETAILED_STUDY_TYPES,
                                               load_human_log_data,
                                               replay_with_joint_actions)

DEBUG = False

bc_column_names = [
    "total_sparse_reward",
    "checkpoints",
    "total_time_step",
    "workloads",
    "concurr_active",
    "stuck_time",
]

direct_mean_column_names = [
    "total_sparse_reward",
    "concurr_active",
    "stuck_time",
    "total_time_step",
]

column_names_to_keep = [
    "lvl_type",
    "workloads",
    "total_time_step",
    "concurr_active",
    "stuck_time",
    "joint_actions",
    "lvl_str",
]

diff = []
human_data_logs = []
for i, log_index in enumerate(sorted(os.listdir(LSI_HUMAN_STUDY_RESULT_DIR))):
    if not os.path.isdir(os.path.join(LSI_HUMAN_STUDY_RESULT_DIR, log_index)):
        continue
    _, human_log_data = load_human_log_data(log_index)
    human_log_data = human_log_data.sort_values(by=["lvl_type"]).reset_index(
        drop=True)

    # add a total timestep column
    total_time_steps = []

    # Rerun the level.
    # Due to some tricky bug that we cannot discover, we rerun the level
    # to make sure that the bc it is correct. If the value is different, we
    # use the value from the rerun and raise a warning.

    for index, row in human_log_data.iterrows():
        lvl_type = row["lvl_type"]
        lvl_str = row["lvl_str"]
        joint_actions = ast.literal_eval(row["joint_actions"])

        workloads, concurr_active, stuck_time, checkpoints, total_time_step = replay_with_joint_actions(
            lvl_str, joint_actions, plot=False)
        total_time_steps.append(total_time_step)

        if DEBUG:
            print(f"Replaying {log_index}, {lvl_type}")

            print("replayed concurr:", concurr_active)
            print("original concurr:", row["concurr_active"])

            print("replayed stuck time", stuck_time)
            print("original stuck time", row["stuck_time"])

            print("replayed workloads", workloads)
            print("original workloads", row["workloads"])
            print()

            print("replayed checkpoints", checkpoints)
            print("original checkpoints", row["checkpoints"])
            print()

        # raise a warning while the results are different.
        if ast.literal_eval(row["workloads"]) != workloads:
            human_log_data.at[index, "workloads"] = workloads
            warnings.warn(
                "Workloads are different for replayed and original level!")

        if concurr_active != row["concurr_active"]:
            human_log_data.at[index, "concurr_active"] = concurr_active
            warnings.warn(
                "Concurrent active times are different for replayed and original level!"
            )

        if stuck_time != row["stuck_time"]:
            human_log_data.at[index, "stuck_time"] = stuck_time
            warnings.warn(
                "Stuck time are different for replayed and original level!")

        if checkpoints != row["checkpoints"]:
            human_log_data.at[index, "checkpoints"] = checkpoints
            warnings.warn(
                "Checkpoints are different for replayed and original level!")

    # add total time step column
    human_log_data["total_time_step"] = total_time_steps

    human_log_data[column_names_to_keep].to_csv(os.path.join(
        LSI_HUMAN_STUDY_RESULT_DIR, log_index, "human_log_refined.csv"),
                                                index=False)

    human_data_logs.append((log_index, human_log_data))

merged_human_logs = pandas.DataFrame()

first_log_index = None
last_log_index = None
num_log_dir = 0
for bc_column_name in bc_column_names:
    first_log_index = None
    last_log_index = None
    num_log_dir = 0
    workloads_dicts = []
    for log_index, human_log_data in human_data_logs:
        # add lvl types
        if not "lvl_type" in merged_human_logs:
            merged_human_logs["lvl_type"] = human_log_data["lvl_type"]

        # add bcs
        # special case for workloads
        if bc_column_name == "workloads":
            workloads_dict = {}
            for index, item in human_log_data["workloads"].iteritems():
                curr_workloads = ast.literal_eval(str(item))
                for agent_i, agent_w in enumerate(curr_workloads):
                    for key in agent_w:
                        new_key = f"{key}-{agent_i}"
                        if new_key not in workloads_dict:
                            workloads_dict[new_key] = []
                        workloads_dict[new_key].append(agent_w[key])
            workloads_dicts.append((log_index, workloads_dict))

        else:
            merged_human_logs[
                f"user-{log_index}-{bc_column_name}"] = human_log_data[
                    bc_column_name]

        if num_log_dir == 0:
            first_log_index = log_index

        num_log_dir += 1
        last_log_index = log_index

    # add mean of the added bc as the new colunmn
    num_user = len(os.listdir(LSI_HUMAN_STUDY_RESULT_DIR))

    # directly calculate the mean for some columns
    if bc_column_name in direct_mean_column_names:
        sub_cols = merged_human_logs.loc[:,
                                         f"user-{first_log_index}-{bc_column_name}":
                                         f"user-{last_log_index}-{bc_column_name}"]
        merged_human_logs[f'{bc_column_name}_mean'] = sub_cols.mean(axis=1)

    # special treatment for workloads
    elif bc_column_name == "workloads":
        # take the mean
        num_lvl = merged_human_logs.shape[0]
        mean_workloads = {
            'num_ingre_held': np.zeros(num_lvl),
            'num_plate_held': np.zeros(num_lvl),
            'num_served': np.zeros(num_lvl),
        }
        for log_index, workloads_dict in workloads_dicts:
            if DEBUG:
                pprint(workloads_dict)
            for key in mean_workloads:
                mean_workloads[key] += np.abs(
                    np.subtract(workloads_dict[f"{key}-0"],
                                workloads_dict[f"{key}-1"]))
        num_user = len(workloads_dicts)
        for key in mean_workloads:
            mean_workloads[key] /= num_user

        if DEBUG:
            pprint(mean_workloads)

        # add workloads to result dataframe
        for key in workloads_dicts[0][1].keys():
            for log_index, workloads_dict in workloads_dicts:
                merged_human_logs[f"user-{log_index}-{key}"] = workloads_dict[
                    key]

merged_human_logs.to_csv(os.path.join(LSI_HUMAN_STUDY_RESULT_DIR,
                                      "merged_human_log.csv"),
                         index=False)
