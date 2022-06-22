"""Generates heat maps of the archives for the paper.
Pass in the path LOGDIR to a logging directory created by the search. This
script will read in data and configuration information from the logging
directory, and for each pair of features in the map, it will generate the
following files in LOGDIR/images:

- PDF called `map_final_{y_idx}_{x_idx}.pdf` showing the final heatmap. This is
  a PDF because PDF figures work better with Latex.
- AVI called `map_video_{y_idx}_{x_idx}.avi` showing the progress of the heatmap.

The {y_idx} and {x_idx} are the indices of the features used in the file in the
list `elite_map_config.Map.Features` in `config.toml`. {y_idx} is the index of
the feature along the y-axis, and {x_idx} is the index of the feature used along
the x-axis.

Usage:
    python gen_heat_map.py -l LOGDIR
"""
import argparse
import csv
import os
import shutil
from enum import Enum
from typing import Dict, List, Tuple, Union

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import toml
from alive_progress import alive_bar
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from overcooked_ai_pcg import LSI_CONFIG_ALGO_DIR, LSI_CONFIG_MAP_DIR

# Visualization settings.
REGULAR_FIGSIZE = (7, 6)
FPS = 10  # FPS for video.
NUM_TICKS = 5  # Number of ticks on plots.
COLORMAP = "viridis"  # Colormap for everything.

# Map settings.
FITNESS_MIN = 0
FITNESS_MAX = 40000
WORKLOAD_DIFFS_LOW = np.array([-6, -2, -2])
WORKLOAD_DIFFS_HIGH = np.array([6, 2, 2])

# Maps the raw feature names to a more human-readable name.
FEATURE_NAME = {
    "diff_num_ingre_held": "Difference in Ingredients (R-H)",
    "diff_num_plate_held": "Difference in Plates (R-H)",
    "diff_num_dish_served": "Difference in Orders (R-H)",
    "cc_active": "% concurrent motion",
    "stuck_time": "% time stuck",
}


class Mode(Enum):
    """Script mode, determines the type of plot to generate."""
    NORMAL = 0
    WORKLOADS_DIFF = 1
    WORKLOADS_DIFF_FINE = 2

    def use_3d(self):
        """Whether to enumerate one of the dimensions to display 3D BCs."""
        return self in [Mode.WORKLOADS_DIFF, Mode.WORKLOADS_DIFF_FINE]


def read_in_lsi_config(exp_config_file: str) -> Tuple[Dict, Dict, Dict]:
    experiment_config = toml.load(exp_config_file)
    algorithm_config = toml.load(
        os.path.join(
            LSI_CONFIG_ALGO_DIR,
            experiment_config["experiment_config"]["algorithm_config"]))
    elite_map_config = toml.load(
        os.path.join(
            LSI_CONFIG_MAP_DIR,
            experiment_config["experiment_config"]["elite_map_config"]))
    return experiment_config, algorithm_config, elite_map_config


def csv_data_to_pandas(
        data: List, y_feature_idx: int, x_feature_idx: int, num_features: int,
        mode: Mode) -> Union[Dict[int, pd.DataFrame], pd.DataFrame]:
    """Converts one row from elite_map.csv into a dataframe.

    (One row of the data contains a snapshot of the entire map at that
    iteration).

    If Mode.use_3d(), this function will instead return a dict of dataframes
    mapping values along one of the BCs to dataframes with the data of elites
    with that BC.
    """
    map_dims = tuple(map(int, data[0].split('x')))
    elites = data[1:]  # The rest of the row contains all the data.

    # Create 2D dataframe(s) to store the map data.

    # Index is descending to make the heatmap look better.
    index_labels = np.arange(map_dims[y_feature_idx] - 1, -1, -1)
    column_labels = np.arange(0, map_dims[x_feature_idx])

    if mode.use_3d():
        # For workload diff, create a dict of pandas dataframes; each one has y
        # as index and x as columns, and the dict is indexed by our 3rd BC.
        dataframes = {}

        # Index along which we will enumerate the BC. It is the one index not
        # covered by y_feature_idx and x_feature_idx.
        enumerate_idx = list(set(range(3)) - {y_feature_idx, x_feature_idx})[0]

        initial_data = np.full((len(index_labels), len(column_labels)), np.nan)
        for i in range(map_dims[enumerate_idx]):
            # Make sure to copy the initial data as pandas uses a reference.
            dataframes[i] = pd.DataFrame(np.copy(initial_data), index_labels,
                                         column_labels)
    else:
        # Create a pandas dataframe with our two BCs on the indices and columns.
        # Index is descending to make the heatmap look better.
        initial_data = np.full((len(index_labels), len(column_labels)), np.nan)
        dataframe = pd.DataFrame(initial_data, index_labels, column_labels)

    # Iterate through the entries in the map and insert them into the
    # appropriate dict.
    for elite in elites:
        tokens = elite.split(":")  # Each elite starts in string format.
        bc_indices = np.array(list(map(int, tokens[:num_features])))
        cell_y = bc_indices[y_feature_idx]
        cell_x = bc_indices[x_feature_idx]

        # Adjust fitness.
        fitness = float(tokens[num_features + 1])
        if fitness == 1:
            fitness = 0
        elif fitness < 200_000:
            fitness -= 100_000
        elif fitness < 300_000:
            fitness -= 190_000
        elif fitness < 400_000:
            fitness -= 280_000
        else:
            fitness -= 370_000

        #elif 200_000 <= fitness < 400_000:
        #    fitness -= 200_000
        #elif fitness >= 400_000:
        #    fitness -= 390_000

        # assert FITNESS_MIN == 0, \
        #     "Fitness min should be 0 to have proper normalization"
        fitness /= FITNESS_MAX  # Normalization - assumes min is 0.

        if mode.use_3d():
            # Insert into the correct dict. We keep all vals (none should be
            # intersecting).
            cell_enum = bc_indices[enumerate_idx]
            dataframes[cell_enum].loc[cell_y, cell_x] = fitness
        else:
            # Insert into the dataframe. Override with better vals.
            old_fitness = dataframe.loc[cell_y, cell_x]
            if np.isnan(old_fitness) or fitness > old_fitness:
                dataframe.loc[cell_y, cell_x] = fitness

    # Use correct BC values.
    if mode.use_3d():
        old_dataframes = dataframes
        dataframes = {}
        step = {Mode.WORKLOADS_DIFF: 1, Mode.WORKLOADS_DIFF_FINE: 0.1}[mode]
        enum_labels = np.arange(WORKLOAD_DIFFS_LOW[enumerate_idx],
                                WORKLOAD_DIFFS_HIGH[enumerate_idx] + step,
                                step)
        x_labels = np.arange(WORKLOAD_DIFFS_LOW[x_feature_idx],
                             WORKLOAD_DIFFS_HIGH[x_feature_idx] + step, step)
        y_labels = np.arange(WORKLOAD_DIFFS_HIGH[y_feature_idx],
                             WORKLOAD_DIFFS_LOW[y_feature_idx] - step, -step)

        if mode == Mode.WORKLOADS_DIFF_FINE:
            # Make decimal keys look pretty.
            enum_labels = list(map(lambda x: f"{x:.1f}", enum_labels))
            x_labels = list(map(lambda x: f"{x:.1f}", x_labels))
            y_labels = list(map(lambda x: f"{x:.1f}", y_labels))

        for idx, df in old_dataframes.items():
            df.rename(index=dict(zip(index_labels, y_labels)), inplace=True)
            df.rename(columns=dict(zip(column_labels, x_labels)), inplace=True)
            dataframes[enum_labels[idx]] = df

    return dataframes if mode.use_3d() else dataframe


def create_axes(
    mode: Mode, dataframe: pd.DataFrame, enumerate_name: str
) -> Tuple[mpl.figure.Figure, Union[mpl.axes.Axes, np.ndarray], mpl.axes.Axes]:
    """Creates a figure, axis/axes, and colorbar axis.

    If mode.use_3d() is True, the ax returned will be an array of axes rather
    than a single axis.

    enumerate_name only applies if mode.use_3d() is True.
    """

    if mode.use_3d():
        y_len = len(dataframe[list(dataframe)[0]].index)
        x_len = len(dataframe[list(dataframe)[0]].columns)
        is_vertical = y_len > x_len

        if is_vertical:
            # These are the main dims we use (we don't really plot horizontal
            # and square archives).
            if mode == Mode.WORKLOADS_DIFF:
                figsize = (9, 6)
            elif mode == Mode.WORKLOADS_DIFF_FINE:
                figsize = (72, 7)
        elif y_len == x_len:
            figsize = (18, 3)
        else:
            figsize = (15, 3)

        # third row is padding.
        height_ratios = ([0.03, 0.82, 0.08, 0.05]
                         if is_vertical else [0.05, 0.83, 0.01, 0.1])

        fig = plt.figure(figsize=figsize)
        num_plots = len(dataframe)  # dataframe is a dict in this case.
        spec = fig.add_gridspec(ncols=num_plots,
                                nrows=4,
                                hspace=0.0,
                                height_ratios=height_ratios)

        ax = np.array([fig.add_subplot(spec[1, i]) for i in range(num_plots)],
                      dtype=object)

        # Place title.
        title_ax = fig.add_subplot(spec[0,
                                        num_plots // 2 - 1:num_plots // 2 + 2])
        title_ax.set_axis_off()
        title_ax.text(0.5, 0, enumerate_name, ha="center", fontsize="medium")

        # Make the colorbar span the entire figure in vertical plots and only
        # the middle three plots in horizontal figures.
        cbar_ax = fig.add_subplot(
            spec[-1, :] if is_vertical and mode == Mode.WORKLOADS_DIFF else
            spec[-1, num_plots // 2 - 1:num_plots // 2 + 2], )

    else:
        fig, ax = plt.subplots(1, 1, figsize=REGULAR_FIGSIZE)
        ax_divider = make_axes_locatable(ax)
        cbar_ax = ax_divider.append_axes("right", size="7%", pad="10%")

    return fig, ax, cbar_ax


def set_spines_visible(ax: mpl.axis.Axis):
    for pos in ["top", "right", "bottom", "left"]:
        ax.spines[pos].set_visible(True)


def plot_heatmap(dataframe: Union[pd.DataFrame, Dict[int, pd.DataFrame]],
                 ax: Union[mpl.axes.Axes, np.ndarray], cbar_ax: mpl.axes.Axes,
                 y_name: str, x_name: str, mode: Mode):
    """Plots a heatmap of the given dataframe onto the given ax.

    A colorbar is created on cbar_ax.

    If  is True, ax should be an array of axes on which to plot
    the heatmap for each value of the enumerating BC. dataframe should then be a
    dict as described in csv_data_to_pandas().
    """
    if mode.use_3d():
        for idx, entry in enumerate(dataframe.items()):
            enum_bc, ind_dataframe = entry
            sns.heatmap(
                ind_dataframe,
                annot=False,
                cmap=COLORMAP,
                fmt=".0f",
                yticklabels=False,
                vmin=0,
                vmax=1,
                square=True,
                ax=ax[idx],
                cbar=idx == 0,  # Only plot cbar for first plot.
                cbar_ax=cbar_ax,
                cbar_kws={"orientation": "horizontal"})
            ax[idx].set_title(f"{enum_bc}", pad=8)
            if idx == 0:  # y-label on first plot.
                ax[idx].set_ylabel(y_name, labelpad=8)
            if idx == len(dataframe) // 2:  # x-label on center plot.
                ax[idx].set_xlabel(x_name, labelpad=6)

            # Hard-coded...
            if mode == Mode.WORKLOADS_DIFF:
                ax[idx].set_xticks([0.5, 2.5, 4.5])
                ax[idx].set_xticklabels([-2, 0, 2], rotation=0)
                if idx == 0:
                    ax[idx].set_yticks([0.5, 3.5, 6.5, 9.5, 12.5])
                    ax[idx].set_yticklabels([-6, -3, 0, 3, 6][::-1],
                                            rotation=0)
            elif mode == Mode.WORKLOADS_DIFF_FINE:
                ax[idx].set_xticks([0.5, 20.5, 40.5])
                ax[idx].set_xticklabels([-2, 0, 2], rotation=0)
                if idx == 0:
                    ax[idx].set_yticks([0.5, 30.5, 60.5, 90.5, 120.5])
                    ax[idx].set_yticklabels([-6, -3, 0, 3, 6][::-1],
                                            rotation=0)
        for a in ax.ravel():
            set_spines_visible(a)
        ax[0].figure.tight_layout()
    else:
        # Mainly specialized for the 2D plots in the paper.

        sns.heatmap(dataframe,
                    annot=False,
                    cmap=COLORMAP,
                    fmt=".0f",
                    vmin=0,
                    vmax=1,
                    square=True,
                    ax=ax,
                    cbar_ax=cbar_ax,
                    linecolor=(0, 0, 0))
        #ax.set_xticks([0.5, 20.5, 40.5, 60.5, 80.5, 100.5])
        #ax.set_yticks([0.5, 20.5, 40.5, 60.5, 80.5, 100.5])
        # ax.set_xticks([0.5,20.5,40.5])
        ax.set_xticks([0.5, 10.5, 20.5, 30.5, 40.5])
        ax.set_xticklabels([0, '', 20, '', 40], rotation=0)
        #ax.set_yticks([0.5, 20,])
        #ax.set_xticklabels([0, 20, 40], rotation=0)

        ax.set_yticks([0.5, 10.5, 20.5, 30.5, 40.5, 50.5, 60.5, 70.5])
        #ax.set_yticklabels([30,40,50,60,70,80,90,100][::-1])
        ax.set_yticklabels(['', 40, '', 60, '', 80, '', 100][::-1])

        #ax.set_yticklabels([0, 30, 60, 90][::-1])
        ax.set_ylabel(y_name, labelpad=12)
        ax.set_xlabel(x_name, labelpad=10)
        set_spines_visible(ax)
        ax.figure.tight_layout()
        #from IPython import embed
        #embed()


def save_video(img_paths: List[str], video_path: str):
    """Creates a video from the given images."""
    # pylint: disable = no-member

    # Grab the dimensions of the image.
    img = cv2.imread(img_paths[0])
    img_dims = img.shape[:2][::-1]

    # Create a video.
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(video_path, fourcc, FPS, img_dims)

    for img_path in img_paths:
        img = cv2.imread(img_path)
        video.write(img)

    video.release()


def main(opt):
    # Read in configurations.
    experiment_config, _, elite_map_config = read_in_lsi_config(
        os.path.join(opt.logdir, "config.tml"))
    features = elite_map_config['Map']['Features']

    mode = {
        "workloads_diff.tml": Mode.WORKLOADS_DIFF,
        "workloads_diff_finer_ver.tml": Mode.WORKLOADS_DIFF_FINE,
    }.get(
        experiment_config["experiment_config"]["elite_map_config"],
        Mode.NORMAL,
    )

    # Global plot settings.
    sns.set_theme(
        context="paper",
        style="ticks",
        font="Palatino Linotype",
        font_scale=2.4 if mode.use_3d() else 2.2,
        rc={
            # Refer to https://matplotlib.org/3.2.1/tutorials/introductory/customizing.html
            "axes.facecolor": "1",
            "xtick.bottom": True,
            "xtick.major.width": 0.8,
            "xtick.major.size": 3.0,
            "ytick.left": True,
            "ytick.major.width": 0.8,
            "ytick.major.size": 3.0,
        })

    # Create image directory and clear out previous images.
    img_dir = os.path.join(opt.logdir, "images/")
    if os.path.exists(img_dir):
        shutil.rmtree(img_dir)
    os.mkdir(img_dir)

    # Retrieve elite map. This file contains data about the _entire_ map after
    # each evaluation.
    with open(os.path.join(opt.logdir, "elite_map.csv"), "r") as csvfile:
        elite_map_data = list(csv.reader(csvfile, delimiter=','))
        elite_map_data = elite_map_data[1:]  # Exclude header.

    # Create outputs for every pair of features (allow reversing so we can get
    # all orientations of images).
    for y_feature_idx, y_feature in enumerate(features):
        for x_feature_idx, x_feature in enumerate(features):
            if y_feature_idx == x_feature_idx:
                continue

            y_name = FEATURE_NAME.get(y_feature["name"], y_feature["name"])
            x_name = FEATURE_NAME.get(x_feature["name"], x_feature["name"])
            if mode.use_3d():
                # The index of the feature along which to enumerate BCs.
                enumerate_idx = list(
                    set(range(3)) - {y_feature_idx, x_feature_idx})[0]
                enumerate_name = features[enumerate_idx]["name"]
                enumerate_name = FEATURE_NAME.get(enumerate_name,
                                                  enumerate_name)
            else:
                enumerate_name = None

            print("-------------------------\n"
                  "## Info ##\n"
                  f"y: Feature {y_feature_idx} ({y_name})\n"
                  f"x: Feature {x_feature_idx} ({x_name})\n"
                  "## Saving PDF of final map ##")
            dataframe = csv_data_to_pandas(elite_map_data[-1], y_feature_idx,
                                           x_feature_idx, len(features), mode)
            fig, ax, cbar_ax = create_axes(mode, dataframe, enumerate_name)
            plot_heatmap(dataframe, ax, cbar_ax, y_name, x_name, mode)
            fig.savefig(
                os.path.join(img_dir,
                             f"map_final_{y_feature_idx}_{x_feature_idx}.pdf"))

            if opt.video:
                print("## Generating video ##")
                video_img_paths = []
                frames = np.append(
                    np.arange(opt.step_size,
                              len(elite_map_data) + 1, opt.step_size),
                    np.full(5, len(elite_map_data)))
                with alive_bar(len(frames)) as progress:
                    for i, frame in enumerate(frames):
                        fig, ax, cbar_ax = create_axes(mode, dataframe,
                                                       enumerate_name)
                        dataframe = csv_data_to_pandas(
                            elite_map_data[frame - 1], y_feature_idx,
                            x_feature_idx, len(features), mode)
                        plot_heatmap(dataframe, ax, cbar_ax, y_name, x_name,
                                     mode)
                        video_img_paths.append(
                            os.path.join(
                                img_dir,
                                f"tmp_frame_{y_feature_idx}_{x_feature_idx}_{i*1000}.png"
                            ))
                        fig.savefig(video_img_paths[-1])
                        plt.close(fig)
                        progress()

                save_video(
                    video_img_paths,
                    os.path.join(
                        img_dir,
                        f"map_video_{y_feature_idx}_{x_feature_idx}.avi"))

                # for path in video_img_paths:
                #     os.remove(path)

            # Break early because we only want the plot for features 0 and 1 for
            # workload_diff
            if mode.use_3d():
                print("Breaking early for workload diff")
                break
        if mode.use_3d():
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-l",
        "--logdir",
        help=("Path to experiment logging directory. Images are"
              "also output here in the 'images' subdirectory"),
        required=True,
    )
    parser.add_argument(
        "-s",
        "--step_size",
        help="step size of the animation to generate",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--video",
        dest="video",
        action="store_true",
        default=True,
        help=("Whether to create the video (it may be useful to turn this off "
              "for debugging. Pass --no-video to disable."),
    )
    parser.add_argument("--no-video", dest="video", action="store_false")

    main(parser.parse_args())
