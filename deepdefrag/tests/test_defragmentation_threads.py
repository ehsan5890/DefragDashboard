import gym
import os
import copy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.monitor import get_monitor_files
from optical_rl_gym.envs.rmsa_env import shortest_path_first_fit, shortest_available_path_first_fit, \
    least_loaded_path_first_fit, SimpleMatrixObservation, Fragmentation_alignment_aware_RMSA
from optical_rl_gym.utils import evaluate_heuristic, random_policy
import pandas as pd
from optical_rl_gym.utils import plot_spectrum_assignment, plot_spectrum_assignment_and_waste

import pickle
import logging
import numpy as np
from multiprocessing import Process

import matplotlib.pyplot as plt

logging.getLogger('rmsaenv').setLevel(logging.INFO)

seed = 20
episodes = 3000
episode_length = 200
incremental_traffic_percentage = 80
monitor_files = []
policies = []

# adding logging method
#log_dir = "./tmp/logrmsa-ppo/"
logging_dir = "../examples/stable_baselines3/results/"
figures_floder = f'{logging_dir}/figures-{incremental_traffic_percentage}/'
os.makedirs(logging_dir, exist_ok=True)
os.makedirs(figures_floder, exist_ok=True)
topology_name = 'gbn'
topology_name = 'nobel-us'
topology_name = 'germany50'
with open(f'../examples/topologies/nsfnet_chen_eon_5-paths.h5', 'rb') as f:
    topology = pickle.load(f)
#
# with open(f'../examples/topologies/Telia_5-paths.h5', 'rb') as f:
#      topology = pickle.load(f)

min_load = 100
max_load = 101
step_length = 8
steps = int((max_load - min_load)/step_length) +1
loads = np.zeros(steps)

def run_oldest(defrag_period, number_moves, env_args_f, num_eps, log_dir_f):
    env_args_f['defragmentation_period'] = defrag_period
    env_args_f['movable_connections'] = number_moves
    env_sp = gym.make('RMSA-v0', **env_args_f)
    env_sp = Monitor(env_sp, log_dir_f + f'df-oldest-{defrag_period}-{number_moves}', info_keywords=('episode_service_blocking_rate','episode_bit_rate_blocking_rate',
                                                      'number_arrivals',
 'bit_rate_blocking_rate', 'number_movements_episode',
 'number_defragmentation_procedure_episode','service_blocked_eopisode'))
    evaluate_heuristic(env_sp, shortest_path_first_fit, n_eval_episodes=num_eps)

processes = []

node_request_probabilities = np.array([0.01801802, 0.04004004, 0.05305305, 0.01901902, 0.04504505,
       0.02402402, 0.06706707, 0.08908909, 0.13813814, 0.12212212,
       0.07607608, 0.12012012, 0.01901902, 0.16916917])

for load_counter, load_traffic in enumerate(range(min_load,max_load,step_length)):
    log_dir = f'{logging_dir}heuristic5/'
    os.makedirs(log_dir, exist_ok=True)
    env_args = dict(topology=topology, seed=10, allow_rejection=True, load=load_traffic, mean_service_holding_time=25,
                    episode_length=episode_length, num_spectrum_resources=320, incremental_traffic_percentage = incremental_traffic_percentage, defragmentation_period = 34, movable_connections = 12, node_request_probabilities = node_request_probabilities)
    #
    # print('STR'.ljust(5), 'REW'.rjust(7), 'STD'.rjust(7))
    loads[load_counter] = load_traffic

    for i in [ (1,40),(5,20)]:
        #  for i in [(32,35)]:
        p = Process(target=run_oldest, args=(i[0], i[1], copy.deepcopy(env_args), episodes, log_dir))
        p.start()
        processes.append(p)

[p.join() for p in processes]  # wait for the completion of all processes