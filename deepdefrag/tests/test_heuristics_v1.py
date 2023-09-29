import gym
import copy
import os
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.monitor import get_monitor_files
from optical_rl_gym.envs.rmsa_env import shortest_path_first_fit, shortest_available_path_first_fit, \
 least_loaded_path_first_fit, SimpleMatrixObservation, Fragmentation_alignment_aware_RMSA
from optical_rl_gym.envs.defragmentation_env import choose_randomly, OldestFirst, assigning_path_without_defragmentation, HighMetricFirst, HighCutMetricFirst
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
episodes = 2000
episode_length = 200
incremental_traffic_percentage = 80
monitor_files = []
policies = []
logging_dir = "../examples/stable_baselines3/results/"
figures_floder = f'{logging_dir}/figures-{incremental_traffic_percentage}/'
os.makedirs(logging_dir, exist_ok=True)
os.makedirs(figures_floder, exist_ok=True)
# topology_name = 'nsfnet'
topology_name = 'Germany50'
# node_request_probabilities = np.array([0.01801802, 0.04004004, 0.05305305, 0.01901902, 0.04504505,
#        0.02402402, 0.06706707, 0.08908909, 0.13813814, 0.12212212,
#        0.07607608, 0.12012012, 0.01901902, 0.16916917])
# with open(f'../examples/topologies/nsfnet_chen_eon_5-paths.h5', 'rb') as f:
#  topology = pickle.load(f)

with open(f'../examples/topologies/{topology_name}_5-paths.h5', 'rb') as f:
  topology = pickle.load(f)
#
# with open(f'../examples/topologies/Telia_5-paths.h5', 'rb') as f:
# topology = pickle.load(f)

min_load = 400
max_load =950
step_length = 100
steps = int((max_load - min_load)/step_length) +1
loads = np.zeros(steps)

def run_with_callback(callback, env_args_f, num_eps, log_dir_f, load_traffic):
 env_df = gym.make('Defragmentation-v0', **env_args_f)
 env_df = Monitor(env_df, log_dir_f+f"{load_traffic}", info_keywords=('episode_service_blocking_rate', 'service_blocking_rate',
 'reward', 'number_movements',
 'number_defragmentation_procedure', 'number_arrivals',
 'bit_rate_blocking_rate', 'number_movements_episode',
 'number_defragmentation_procedure_episode','service_blocked_eopisode', 'number_options', 'existing_options',
                                                                      'episode_service_blocking_rate_fragmentation','episode_service_blocking_rate_lack', 'episode_cuts','episode_frag_metric','episode_shanon_entropy'))
 evaluate_heuristic(env_df, callback, n_eval_episodes=num_eps)

def run_oldest(defrag_period, number_moves, env_args_f, num_eps, log_dir_f, fragmented_constraint, load_traffic):
 oldest_scenario = OldestFirst(defrag_period, number_moves)
 env_df_oldest = gym.make('Defragmentation-v0', **env_args_f)
 env_df_oldest = Monitor(env_df_oldest, log_dir_f + f'df-oldest-{defrag_period}-{number_moves}-{fragmented_constraint}-{load_traffic}',
 info_keywords=('episode_service_blocking_rate', 'service_blocking_rate',
 'reward', 'number_movements',
 'number_defragmentation_procedure', 'number_arrivals',
 'bit_rate_blocking_rate', 'number_movements_episode',
 'number_defragmentation_procedure_episode','service_blocked_eopisode','number_options', 'existing_options',
                'episode_service_blocking_rate_fragmentation', 'episode_service_blocking_rate_lack','episode_cuts','episode_frag_metric','episode_shanon_entropy'))
 evaluate_heuristic(env_df_oldest, oldest_scenario.choose_oldest_first,n_eval_episodes=num_eps)


def run_highest(defrag_period, number_moves, env_args_f, num_eps, log_dir_f, fragmented_constraint, load_traffic):
 highest_scenario = HighMetricFirst(defrag_period, number_moves)
 env_df_highest = gym.make('Defragmentation-v0', **env_args_f)
 env_df_highest = Monitor(env_df_highest, log_dir_f + f'df-highest-{defrag_period}-{number_moves}-{fragmented_constraint}-{load_traffic}',
 info_keywords=('episode_service_blocking_rate', 'service_blocking_rate',
 'reward', 'number_movements',
 'number_defragmentation_procedure', 'number_arrivals',
 'bit_rate_blocking_rate', 'number_movements_episode',
 'number_defragmentation_procedure_episode','service_blocked_eopisode','number_options', 'existing_options'
                ,'episode_service_blocking_rate_fragmentation','episode_service_blocking_rate_lack','episode_cuts','episode_frag_metric','episode_shanon_entropy'))
 evaluate_heuristic(env_df_highest, highest_scenario.choose_highest_difference_first,n_eval_episodes=num_eps)



def run_highest_cut(defrag_period, number_moves, env_args_f, num_eps, log_dir_f, fragmented_constraint, load_traffic):
 highest_scenario = HighCutMetricFirst(defrag_period, number_moves)
 env_df_highest = gym.make('Defragmentation-v0', **env_args_f)
 env_df_highest = Monitor(env_df_highest, log_dir_f + f'df-highest-cut-{defrag_period}-{number_moves}-{fragmented_constraint}-{load_traffic}',
 info_keywords=('episode_service_blocking_rate', 'service_blocking_rate',
 'reward', 'number_movements',
 'number_defragmentation_procedure', 'number_arrivals',
 'bit_rate_blocking_rate', 'number_movements_episode',
 'number_defragmentation_procedure_episode','service_blocked_eopisode','number_options', 'existing_options','episode_service_blocking_rate_fragmentation','episode_service_blocking_rate_lack','episode_cuts','episode_frag_metric','episode_shanon_entropy'))
 evaluate_heuristic(env_df_highest, highest_scenario.choose_highest_cut_difference_first,n_eval_episodes=num_eps)


if __name__ == '__main__':
 processes = []
 for load_counter, load_traffic in enumerate(range(min_load,max_load,step_length)):
     for traffic_type in [1,2]:
      log_dir = f'{logging_dir}heuristic15-{topology_name}-{traffic_type}/'
      os.makedirs(log_dir, exist_ok=True)
      env_args = dict(topology=topology, seed=10, allow_rejection=True, load=load_traffic, mean_service_holding_time=0.5,
      episode_length=episode_length, num_spectrum_resources=320, incremental_traffic_percentage=incremental_traffic_percentage, rmsa_function=shortest_available_path_first_fit,
                      )

      ############ No defragmentation#########
      env_args = dict(topology=topology, seed=10, allow_rejection=True, load=load_traffic, mean_service_holding_time=0.5,
      episode_length=episode_length, num_spectrum_resources=320,
                      incremental_traffic_percentage=incremental_traffic_percentage,
                      rmsa_function=shortest_available_path_first_fit,
                      traffic_type = traffic_type
                      )
      p = Process(target=run_with_callback, args=(assigning_path_without_defragmentation, copy.deepcopy(env_args), episodes, log_dir +'df',load_traffic))
      p.start()
      processes.append(p)


      ############ Choose Randomly######

      # p = Process(target=run_with_callback,
      # args=(choose_randomly, copy.deepcopy(env_args), episodes, log_dir+'rnd'))
      # p.start()
      # processes.append(p)

      #####Oldest first#####
      for i in [(10,10),(1,400),]:
          fragmented_constraint = True
          env_args = dict(topology=topology, seed=10, allow_rejection=True, load=load_traffic, mean_service_holding_time=0.5,
                       episode_length=episode_length, num_spectrum_resources=320,
                       incremental_traffic_percentage=incremental_traffic_percentage,
                       rmsa_function=shortest_available_path_first_fit,
                       fragmented_constraint= fragmented_constraint,
                        traffic_type = traffic_type)
          p = Process(target=run_oldest, args=(i[0], i[1], copy.deepcopy(env_args), episodes, log_dir, fragmented_constraint, load_traffic))
          p.start()
          processes.append(p)

      # for i in [(1,400),(10,10)]:
      #     fragmented_constraint = False
      #     env_args = dict(topology=topology, seed=10, allow_rejection=True, load=load_traffic, mean_service_holding_time=0.5,
      #                  episode_length=episode_length, num_spectrum_resources=320,
      #                  incremental_traffic_percentage=incremental_traffic_percentage,
      #                  rmsa_function=shortest_available_path_first_fit,
      #                  node_request_probabilities=node_request_probabilities,
      #                  fragmented_service_constraint= fragmented_constraint)
      #     p = Process(target=run_oldest, args=(i[0], i[1], copy.deepcopy(env_args), episodes, log_dir, fragmented_constraint))
      #     p.start()
      #     processes.append(p)



      for i in [(10,10), (1,400) ]:
          fragmented_constraint = True
          env_args = dict(topology=topology, seed=10, allow_rejection=True, load=load_traffic,
                          mean_service_holding_time=0.5,
                          episode_length=episode_length, num_spectrum_resources=320,
                          incremental_traffic_percentage=incremental_traffic_percentage,
                          rmsa_function=shortest_available_path_first_fit,
                          fragmented_constraint=fragmented_constraint, traffic_type = traffic_type)
          p = Process(target=run_highest,
                      args=(i[0], i[1], copy.deepcopy(env_args), episodes, log_dir, fragmented_constraint, load_traffic))
          p.start()
          processes.append(p)



      # for i in [(10,9), (10,5), (5,10)]:
      #     fragmented_constraint = True
      #     env_args = dict(topology=topology, seed=10, allow_rejection=True, load=load_traffic,
      #                     mean_service_holding_time=0.5,
      #                     episode_length=episode_length, num_spectrum_resources=320,
      #                     incremental_traffic_percentage=incremental_traffic_percentage,
      #                     rmsa_function=shortest_available_path_first_fit,
      #                     fragmented_constraint=fragmented_constraint)
      #     p = Process(target=run_highest_cut,
      #                 args=(i[0], i[1], copy.deepcopy(env_args), episodes, log_dir, fragmented_constraint))
      #     p.start()
      #     processes.append(p)

      # for i in [(1, 400), (10, 10)]:
      #     fragmented_constraint = False
      #     env_args = dict(topology=topology, seed=10, allow_rejection=True, load=load_traffic,
      #                     mean_service_holding_time=0.5,
      #                     episode_length=episode_length, num_spectrum_resources=320,
      #                     incremental_traffic_percentage=incremental_traffic_percentage,
      #                     rmsa_function=shortest_available_path_first_fit,
      #                     node_request_probabilities=node_request_probabilities,
      #                     fragmented_service_constraint=fragmented_constraint)
      #     p = Process(target=run_highest,
      #                 args=(i[0], i[1], copy.deepcopy(env_args), episodes, log_dir, fragmented_constraint))
      #     p.start()
      #     processes.append(p)


 [p.join() for p in processes] # wait for the completion of all processes

