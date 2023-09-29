import gym
import copy
import os
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.monitor import get_monitor_files
from optical_rl_gym.envs.rmsa_env import shortest_path_first_fit, shortest_available_path_first_fit, \
 least_loaded_path_first_fit, SimpleMatrixObservation, Fragmentation_alignment_aware_RMSA
from optical_rl_gym.envs.defragmentation_env import choose_randomly, OldestFirst, assigning_path_without_defragmentation
from optical_rl_gym.utils import evaluate_heuristic, random_policy
import pandas as pd
from optical_rl_gym.utils import plot_spectrum_assignment, plot_spectrum_assignment_and_waste
from multiprocessing import Process
import pickle
import logging
import numpy as np


import matplotlib.pyplot as plt

logging.getLogger('rmsaenv').setLevel(logging.INFO)

seed = 20
episodes = 1500
episode_length = 200
incremental_traffic_percentage = 80
monitor_files = []
policies = []
logging_dir = "../examples/stable_baselines3/results/"
figures_floder = f'{logging_dir}/figures-{incremental_traffic_percentage}/'
os.makedirs(logging_dir, exist_ok=True)
os.makedirs(figures_floder, exist_ok=True)
topology_name = 'gbn'
topology_name = 'nobel-us'
topology_name = 'nsfnet'
# topology_name = 'Germany50'
# node_request_probabilities = np.array([0.01801802, 0.04004004, 0.05305305, 0.01901902, 0.04504505,
#        0.02402402, 0.06706707, 0.08908909, 0.13813814, 0.12212212,
#        0.07607608, 0.12012012, 0.01901902, 0.16916917])
with open(f'../examples/topologies/nsfnet_chen_eon_5-paths.h5', 'rb') as f:
 topology = pickle.load(f)

# with open(f'../examples/topologies/{topology_name}_5-paths.h5', 'rb') as f:
#   topology = pickle.load(f)
#
# with open(f'../examples/topologies/Telia_5-paths.h5', 'rb') as f:
# topology = pickle.load(f)

min_load = 175
max_load = 176
step_length = 8
k_paths = 5
steps = int((max_load - min_load)/step_length) +1
loads = np.zeros(steps)

def run_with_callback(callback, env_args_f, num_eps, log_dir_f):
 env_df = gym.make('Defragmentation-v0', **env_args_f)
 env_df = Monitor(env_df, log_dir_f, info_keywords=('episode_service_blocking_rate', 'service_blocking_rate',
 'reward', 'number_movements',
 'number_defragmentation_procedure', 'number_arrivals',
 'bit_rate_blocking_rate', 'number_movements_episode',
 'number_defragmentation_procedure_episode','service_blocked_eopisode', 'number_options', 'existing_options'))
 evaluate_heuristic(env_df, callback, n_eval_episodes=num_eps)

def run_oldest(defrag_period, number_moves, env_args_f, num_eps, log_dir_f, fragmented_consraint):
 oldest_scenario = OldestFirst(defrag_period, number_moves)
 env_df_oldest = gym.make('Defragmentation-v0', **env_args_f)
 env_df_oldest = Monitor(env_df_oldest, log_dir_f + f'df-oldest-{defrag_period}-{number_moves}-{fragmented_consraint}',
 info_keywords=('episode_service_blocking_rate', 'service_blocking_rate',
 'reward', 'number_movements',
 'number_defragmentation_procedure', 'number_arrivals',
 'bit_rate_blocking_rate', 'number_movements_episode',
 'number_defragmentation_procedure_episode','service_blocked_eopisode','number_options', 'existing_options'))
 evaluate_heuristic(env_df_oldest, oldest_scenario.choose_oldest_first,n_eval_episodes=num_eps)

# topologies = [ 'Germany50', 'Coronet']
loads = [175]

topologies = ['nsfnet_chen_eon']
if __name__ == '__main__':
    processes = []
     # for load_counter, load_traffic in enumerate(range(min_load,max_load,step_length)):
      # for topology_name in ['nsfnet_chen_eon', 'Germany50', 'Coronet']:
    for topology_name, load in zip(topologies, loads):
        with open(f'../examples/topologies/{topology_name}_{k_paths}-paths.h5', 'rb') as f:
            topology = pickle.load(f)
        log_dir = f'{logging_dir}heuristic12-{topology_name}/'
        os.makedirs(log_dir, exist_ok=True)
        env_args = dict(topology=topology, seed=10, allow_rejection=True, load=load, mean_service_holding_time=0.5,
        episode_length=episode_length, num_spectrum_resources=320, incremental_traffic_percentage=incremental_traffic_percentage, rmsa_function=shortest_available_path_first_fit,
                       )

        ############# No defragmentation#########

        p = Process(target=run_with_callback, args=(assigning_path_without_defragmentation, copy.deepcopy(env_args), episodes, log_dir +'df'))
        p.start()
        processes.append(p)

        # ############## Choose Randomly######
        #
        # p = Process(target=run_with_callback,
        # args=(choose_randomly, copy.deepcopy(env_args), episodes, log_dir+'rnd'))
        # p.start()
        # processes.append(p)

        ######Oldest first#####
        for i in [(10,10), (1,400)]:
        #  for i in [(32,35)]:
           fragmented_constraint = True
           env_args = dict(topology=topology, seed=10, allow_rejection=True, load=load, mean_service_holding_time=25,
                      episode_length=episode_length, num_spectrum_resources=320,
                      incremental_traffic_percentage=incremental_traffic_percentage,
                      rmsa_function=shortest_available_path_first_fit, fragmented_constraint=fragmented_constraint
                      )
           p = Process(target=run_oldest, args=(i[0], i[1], copy.deepcopy(env_args), episodes, log_dir, fragmented_constraint))
           p.start()
           processes.append(p)


        for i in [(10,10), (1,400)]:
        #  for i in [(32,35)]:
           fragmented_constraint = False
           env_args = dict(topology=topology, seed=10, allow_rejection=True, load=load, mean_service_holding_time=25,
                      episode_length=episode_length, num_spectrum_resources=320,
                      incremental_traffic_percentage=incremental_traffic_percentage,
                      rmsa_function=shortest_available_path_first_fit, fragmented_constraint=fragmented_constraint
                      )
           p = Process(target=run_oldest, args=(i[0], i[1], copy.deepcopy(env_args), episodes, log_dir, fragmented_constraint))
           p.start()
           processes.append(p)
     #
    [p.join() for p in processes] # wait for the completion of all processes

