# Overcooked-AI-PCG

The Overcooked-AI-PCG procedural content generation project aims to create Overcooked game levels that result in diverse behaviors while a human and an AI agent play cooperatively in the generated environment.

This repository contains the source code for the paper:
"On the importance of environments in Human-Robot Coordination". Matthew Fontaine*, Ya-Chuan Hsu*, Yulun Zhang*, Bryon Tjanaka and Stefanos Nikolaidis. RSS 2021.


### Introduction

Overcooked-AI is a benchmark environment for fully cooperative multi-agent
performance, based on the wildly popular video game
[Overcooked](http://www.ghosttowngames.com/overcooked/).

The goal of the game is to deliver soups as fast as possible. Each soup requires
taking 3 items and placing them in a pot, waiting for the soup to cook, and then
having an agent pick up the soup and delivering it. The agents should split up
tasks on the fly and coordinate effectively in order to achieve high reward.

### Install Overcooked-AI

It is useful to setup a conda environment with Python 3.7 using
[Anaconda](https://www.anaconda.com/products/individual):

```
conda create -n overcooked_ai python=3.7
conda activate overcooked_ai
```

To complete the installation after cloning the repo, run the following commands:

```
cd overcooked_ai
pip install -e .
```

### Overcooked-AI Code Structure Overview

`overcooked_ai_py` contains:

`mdp/`:

- `overcooked_mdp.py`: main Overcooked game logic
- `overcooked_env.py`: environment classes built on top of the Overcooked mdp
- `layout_generator.py`: functions to generate random layouts programmatically
- `actions`: actions that agents can take
- `graphics`: render related functions

`agents/`:

- `agent.py`: location of agent classes
- `benchmarking.py`: sample trajectories of agents (both trained and planners)
  and load various models

`planning`:

- `planners.py`: near-optimal agent planning logic
- `search.py`: A\* search and shortest path logic

`run_tests.py`: script to run all tests

### Python Visualizations

To test the visualization mechanism of Overcooked-AI, please run the following:

```bash
cd overcooked_ai_py
python test_render.py
```

A pygame window should pop up and two agents should start performing random
actions in the environment.

## PCG for Overcooked-AI

### GAN Training

To train the GAN that generates Overcooked-AI levels, run the following:

```bash
cd overcooked_ai_pcg/GAN_training
python train_gan_vanilla.py --cuda
```

Note: We have 2 trained GANs in `overcooked_ai_pcg/GAN_training/data/training`.
One for large (10x15) levels and one for small (6x9) levels.

### Mixed Integer Linear Programming Solver

The solver is defined in `overcooked_ai_pcg/milp_repair.py`.

It uses [cplex optimizer of IBM](https://www.ibm.com/analytics/cplex-optimizer).
Please follow the step
[here](https://www.ibm.com/products/ilog-cplex-optimization-studio) to install
**IBM ILOG CPLEX Optimization Studio** and the python interface of it. Once you
have downloaded the installation file, this
[guide](https://www.ibm.com/support/knowledgecenter/SSSA5P_12.10.0/ilog.odms.studio.help/Optimization_Studio/topics/COS_installing.html)
may be helpful.

### Generate level using trained GAN and MILP solver

To use trained GAN and the MILP solver to generate Overcooked-AI levels, run the
following:

```bash
cd overcooked_ai_pcg/
python gen_lvl.py
```

The program will generate a level from random latent vector sampled from normal
distribution and then use MILP solver defined in
`overcooked_ai_pcg/milp_repair.py` to fix the level.

### Latent Space Illumination

The Overcooked experiments use [Dask](https://docs.dask.org) to run in a
distributed fashion. To begin, make sure you have the Conda environment set up
and your dependencies installed.

Next, change into the `LSI` directory:

```bash
cd overcooked_ai_pcg/LSI
```

Now run:

```bash
python run_search.py -c <exp_config_file_path> -s <level_size_version>
```

`exp_config_file_path` is the filepath to the experiment config file. It
defaults to `overcooked_ai_pcg/LSI/data/config/experiment/MAPELITES_demo.tml`

`level_size_version` is the version of the size of the levels. It can either be
`large` or `small`. `large` size refers to 10x15, `small` size refers to 6x9.

`run_search.py` will output log messages to the command line. Furthermore, visit
the Dask dashboard at http://localhost:8787 to see the status of the Dask
workers. See
[here](https://docs.dask.org/en/latest/diagnostics-distributed.html) for a
walkthrough of the dashboard.

For more info on running the search on a cluster, see the section
[Running on HPC](#running-on-hpc).

#### LSI config files

There are four kinds of config files, each configuring different components of
the LSI experiments. While running the experiments, the `experiment` config files
are the entry points for all the other config files.

##### `experiment` config files

They are under `overcooked_ai_pcg/LSI/data/config/experiment`.

An experiment config file contains the following required fields:

```
visualize (bool): to visualize the evaluations or not
num_cores (int): number of processes that runs the evaluations
num_simulations (int): total number of evaluations/simulations to run
algorithm_config (string): file name of the algorithm config file
elite_map_config (string): file name of the elite map config file
agent_config (list of string): file names of the agent config files
```

The experiment config files are the entry points each LSI experiments.

##### `algorithm` config files

They are under `overcooked_ai_pcg/LSI/data/config/algorithms`.

An algorithm config file contains the following required fields:

```
name (string): name of the algorithm used for deciding which
               algorithm instance to intialize at run time.
```

It also contains hyper params of the algorithm to run. For example, for
MAP-Elites, they are initial population and mutation power.

##### `elite_map` config files

They are under `overcooked_ai_pcg/LSI/data/config/elite_map`.

An elite map config file contains an array of behavior characteristics (bc).
Each bc contains the following required fields:

```
name (string): name of the bc
low (int/double): lower bound of the bc
high (int/double): upper bound of the bc
resolution (int): resolution (how many sections to divide) for the bc
```

Note that the name should match the name of the function to calculate the bc in
`overcooked_ai_pcg/LSI/bc_calculate.py`

##### `agent` config files

They are under `overcooked_ai_pcg/LSI/data/config/agents`.

A agent config file contains the two agents used for running overcooked game.
Each agent contains its own properties, which varies across different agent
type.

#### Experiments and Corresponding config files

##### LSI Experiments

Here we list the experiments in the paper along with their corresponding experiment
config files. We suggest setting `num_cores` parameters to the number of cores that
are available on your machine to fully utilize the compute you have. If you don't have a
powerful local machine to run the experiments, you may also refer to
[our instruction](#Running-on-HPC) to run experiments on the HPC.

| Experiment | Experiment Config file |
| ---------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| Workload Distributions with Centralized Planning                                               | overcooked_ai_pcg/LSI/data/config/experiment/CMAME_workloads_diff_fixed_plan.tml                  |
| Directly Searching for Environment using MAP-Elites                                            | overcooked_ai_pcg/LSI/data/config/experiment/MAPELITES-BASE_workloads_diff_fixed_plan.tml         |

##### Tile Distribution Analysis

We generated 1000 levels in `overcooked_ai_pcg/lvl_dist_analysis/all_lvl_strs.json`.
To recreate the tile distribution plots, run the following:

```
cd overcooked_ai_pcg/lvl_dist_analysis
python lvl_dist_analysis.py
```

### Making More GAN Training Data

You can make your own data and re-trained the GAN. The size of the training
levels is fixed to be 15(width) x 10(height) or 9(width) x 6(height).
The available tile types are:

```
'1': Player 1
'2': Player 2
'X': Wall
'S': Serve Point
'P': Pot
'O': Onion Dispenser
'D': Dish Dispenser
' ': Floor
```

Please make sure that the levels you make satisfy **ALL** of the following
constraints:

1. The level must be **rigidly surrounded**. i.e. the first and last row, and
   the first and last column can be anything except `‘1’`, `‘2’`, and `‘ ’`.

2. There are **exactly 2 players** at different positions. But they cannot be at
   the first and last row, and the first and last column.

3. There is **at least one** `‘O’`.

4. There is **at least one** `‘D’`.

5. There is **at least one** `‘P’`.

6. There is **at least one** `‘S’`.

7. `‘O’`, `‘D’`, `‘P’`, `‘S’` can be **anywhere**.

8. Both of the players must be able to reach at least one of `‘O’`, `‘D’`,
   `‘P’`, and `‘S’`.

9. The size is exactly **15(width) x 10(height)**

Please grab a version of `overcooked_ai_py/data/layouts/train_gan_large/base.layout`
or `overcooked_ai_py/data/layouts/train_gan_small/base.layout` to make the
levels and place them under `overcooked_ai_py/data/layouts/train_gan_large` or
`overcooked_ai_py/data/layouts/train_gan_small`. **Be sure to add
prefix `gen` to its file name to differentiate it from non-GAN-training
layouts.**

Note: These are also the constraints that the MILP solver is trying to satisfy.

### Reloading the Algorithm

Sometimes, `run_search.py` will crash in the middle of a run, perhaps due to the
HPC timing out or memory. Fortunately, we can continue running a crashed
experiment. `run_search.py` saves algorithm state to the logging directory in a
file called `reload.pkl`. In order to continue running from `reload.pkl`, you
will need to pass in the same config files and level size as before. Then, you
will need to pass in `reload.pkl` with the `-r` flag. Thus, your command should
look like:

```bash
python run_search.py -c CONFIG.tml -s SIZE_VERSION -r ..../reload.pkl
```

Note that:

- The same logging directory will be used (since we are continuing a run).
- The algorithm will not redispatch the individuals from the previous run; it
  will simply assume they failed.

### Running on HPC

The evaluations can take a long time. To run on USC's HPC, do the following:

1. SSH into HPC
   1. Make sure you have an HPC account. You may need to contact your PI about
      this.
   1. Run this command with your USCNetID:
      ```bash
      ssh USCNETID@discovery.usc.edu
      ```
1. Set up the environment.
   1. Clone the repo:
      ```bash
      git clone https://github.com/icaros-usc/overcooked_ai
      cd overcooked_ai
      ```
   1. Create a Conda environment:
      ```bash
      conda create --name overcooked_ai python=3.7
      ```
   1. Install this repo:
      ```bash
      pip install -e .
      ```
1. Install CPLEX.
   1. Get the free academic edition
      [here](https://www.ibm.com/products/ilog-cplex-optimization-studio).
   1. Download the installation file for Linux.
   1. Transfer the installation file to the cluster with `scp`. On your
      **local** machine, run
      ```bash
      scp INSTALLATION_FILE USCNETID@discovery.usc.edu:~
      ```
      where `INSTALLATION_FILE` is the location of the installation file and
      `USCNETID` is your USC net ID. This command will put the installation file
      in your home directory. Note: See
      [here](https://carc.usc.edu/user-information/user-guides/data-management/transferring-files-command-line)
      for more help transferring files to the USC HPC.
   1. Follow the instructions for installing CPLEX on Linux
      [here](https://www.ibm.com/support/knowledgecenter/SSSA5P_12.10.0/ilog.odms.studio.help/Optimization_Studio/topics/COS_installing.html).
      Basically:
      ```bash
      chmod u+x INSTALLATION_FILE
      ./INSTALLATION_FILE
      ```
      Make note of the installation directory you choose for CPLEX. Putting it
      in your home directory should work fine.
   1. The installation process will provide instructions on how to install the
      Python API for CPLEX, something like
      ```bash
      python <INSTALLATION_DIR>/python/setup.py install
      ```
      Once the installation is done, activate your Conda env
      (`conda activate overcooked_ai`) and install the API.
1. Run the script.
   1. Change into the `LSI` directory:
      ```bash
      cd overcooked_ai_pcg/LSI
      ```
   1. Edit the `python` command at the bottom of `hpc/run_search.slurm` to be
      whatever command you wish to run for `run_search.py`. Keep in mind that
      even though the script is in the `hpc` directory, you will run it relative
      to the `LSI` directory, so all filepaths should be relative to the `LSI`
      directory.
   1. If you are not in the `overcooked_ai_pcg/LSI` directory, change back into
      it.
   1. Start the script:
      ```bash
      sbatch hpc/run_search.slurm
      ```
1. View output.
   1. The `sbatch` command should output a job number. View the job output with
      ```bash
      cat logs/slurm-<JOBNUM>.out
      ```
      Even better, you can use
      ```bash
      tail -f logs/slurm-<JOBNUM>.out
      ```
      to continuously watch the script output. Note that this log file is only
      created when the job starts, so if you do not see the file, the job may
      not have started yet.
   1. The output above will only output how many simulations have finished
      running. The workers write the Overcooked games they generate to separate
      log files (since they are separate Slurm jobs). Instead of searching for
      these outputs, you can view the individuals log file with
      ```bash
      tail -f data/log/<LOGDIR>/individuals_log.csv
      ```
      where `<LOGDIR>` is the most recently created logging directory in the
      `data/log` directory (each directory has a date and time prepended to its
      name).
   1. To see what Slurm jobs are running (Dask will spawn several `dask-worker`
      jobs), run:
      ```bash
      squeue -u $USER
      ```
      You can replace `$USER` with your username / USCNetID if you would like.
      To continuously see what jobs are running, use:
      ```bash
      watch \"squeue -o '   %20i %.9P %.2t %.8p %.4D %.3C %.10M %20j %R' -u $USER\"
      ```
      We recommend using an alias in your `.bashrc` so that you do not have to
      remember this command.
   1. To view the Dask dashboard, you will need to open an SSH tunnel from your
      machine to HPC. To get a command for opening this tunnel from your local
      machine, run `hpc/dashboard_tunnel.sh logs/slurm-<JOBNUM>.out`. Then run
      the command on your local machine, and visit <http://localhost:8787>. See
      the video tutorial
      [here](https://docs.dask.org/en/latest/diagnostics-distributed.html) for
      more info about the Dask dashboard.
   1. The search will probably take a while to run. Go grab a coffee.

### Overcooked-AI-PCG Code structure Overview

`overcooked_ai_pcg/` contains:

- `milp_repair.py`: Mixed Integer Linear Programming solver to fix levels
  generated by GAN.

- `gen_lvl.py`: Script that generates a level from trained GAN and repair that
  level using MILP solver.

- `helper.py`: helper functions

- `GAN_training/`:

  - `dcgan.py`: Deep Convolutional Generative Adversarial Network Code
  - `train_gan.py`: GAN training script

- `LSI/`:
  - `bc_calculate.py`: Relevant functions to calculate behavior characteristics
  - `qd_algorithms.py`: Implementations of QD algorithms
  - `run_search.py`: Script to run LSI search
  - `evaluator.py`: Overcooked game evaluator
  - `logger.py`: LSI experiment data loggers
  - `data/`: config and log data of LSI experiment

## Credits

The `overcooked_ai_py` directory is adopted from [this project](https://github.com/HumanCompatibleAI/overcooked_ai) by the 
Center for Human-Compatible AI.


### Citing This Work
If you use this code for scholarly work, please kindly cite our work using the Bibtex snippet belw.
```
@inproceedings{fontaine:rss2021,
  title={On the Importance of Environments for Human-Robot Coordination},
  author={Fontaine, Matthew and Hsu, Ya-Chuan and Zhang, Yulun and  Nikolaidis, Stefanos},
  booktitle={Proceedings of Robotics: Science and Systems},
  year={2021}
}
```
