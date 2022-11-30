# Toward Multi-Agent Reinforcement Learning for Distributed Event-Triggered Control

This repository contains the official implementation of [Toward Multi-Agent Reinforcement Learning for Distributed Event-Triggered Control](https://sites.google.com/view/learning-distributed-etc/start). 
The code base is a modified version of the code in [Learning Event-triggered Control from Data through Joint Optimization](https://arxiv.org/pdf/2008.04712.pdf) by N. Funk, D. Baumann, V. Berenz and S. Trimpe. 

## Installation 

1. Clone the repo

2. Install the required packages.
   1. If conda has been installed, inside  *z_additionals* there is the environment.yml file which can be used to obtain all the required packages. Thus, execute 
      ```setup 
      conda env create -f environment.yml 
      ```
   2. Otherwise, make sure that the python environment you want to use contains the packages depicted in the yaml file

   We note that to install everything correctly, it might be necessary to
   install some of the packages manually.

4. Activate your Python environment. If conda has been used:
   ```setup 
   conda activate etc-nets
   ```

5. Install the components of this repository:
   ```setup 
   cd PATH_TO_THIS_REPO/baselines
   ```
   ```setup 
   pip install -e
   ```
   Alternatively: run the setup.py manually 

6. Install the modified Multiwalker environment
   ```setup 
   cd PATH_TO_PETTINGZOO/sisl/multiwalker/
   ```
   Replace the scripts with the modified versions in z_additionals/modified_multiwalker 

## Repo overview

Short overview over the repo:
* **baselines** folder mainly includes the original OpenAI baselines repository, which has been slightly modified such that our algorithm can be trained
* **train_results** is the folder where results are going to be saved
* **z_additionals** contains several files:
  * the exported conda environment
  * it also contains the modified Multiwalker environment

## Training agents

For starting a training run, first select the algorithm you want to train:

1. Move to baselines/baselines/ppo1

2. Choose either our_algorithm or PPO_random_skip

3. Copy run_mujoco.py, pposgd_simple.py and mlp_policy in the "ppo1" directory

To set parameters for the training process, change the input to the "learn" function in *run_mujoco*.

For training a model:

1. Navigate to baselines/baselines/ppo1 and open in terminal

2. Execute:

   ```setup
   python run_mujoco.py --seed 0 --app savename --num 3 --saves --wsaves --render
   ```

For loading a model:

1. Navigate to baselines/baselines/ppo1 and open in terminal

2. Execute 
   ```setup
   python run_mujoco.py --seed 0 --app existing_savename --num 3 --epoch 9999
   ```

   Seed, app and num have to be identical to the version you want to load

Every run of a model creates files in *train_results*. There we include the scripts that were used to run the environment in the first place, as well as the evaluations for each agent, and averaged quantities for the group of agents.
