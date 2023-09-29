import gym
import os
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.monitor import get_monitor_files
from optical_rl_gym.envs.rmsa_env import shortest_path_first_fit, shortest_available_path_first_fit, \
    least_loaded_path_first_fit, SimpleMatrixObservation
from optical_rl_gym.utils import evaluate_heuristic, random_policy
import pandas as pd

import pickle
import logging
import numpy as np

import matplotlib.pyplot as plt


logging.getLogger('rmsaenv').setLevel(logging.INFO)

seed = 20
episodes = 1
episode_length = 70
incremental_traffic_percentage = 80

monitor_files = []
policies = []




# adding logging method
#log_dir = "./tmp/logrmsa-ppo/"
logging_dir = "./tmp/logrmsa-ppo-defragmentation/"
figures_floder = f'{logging_dir}/figures-{incremental_traffic_percentage}/'
os.makedirs(logging_dir, exist_ok=True)
os.makedirs(figures_floder, exist_ok=True)


min_load = 40
max_load = 41
step_length = 8
steps = int((max_load - min_load)/step_length) +1

movable_connections = [0,10,50]
m = len(movable_connections)
SP_SBR_load = np.zeros([m,steps])
SP_BBR_load = np.zeros([m,steps])
SP_EF_load = np.zeros([m,steps])

SP_EFD_load = np.zeros([m,steps])

SP_CF_load = np.zeros([m,steps])
SP_CFN_load = np.zeros([m,steps])
loads = np.zeros(steps)

for load_counter, load_traffic in enumerate(range(min_load,max_load,step_length)):
    for move in range(m):
        log_dir = f'{logging_dir}logs_{load_traffic}_{episode_length}_{incremental_traffic_percentage}/'
        os.makedirs(log_dir, exist_ok=True)
        loads[load_counter] = load_traffic
        #pi load results
        all_results = load_results(log_dir)
        # Defining monitor files based on the below command does not work well, so i defined it statistically
        #monitor_files = get_monitor_files(log_dir)
        monitor_files = ['SP.monitor.csv']
        SBR = dict()
        BBR= dict()
        EF= dict()
        CF= dict()
        CFN = dict()
        counter = 0
        for file_names in monitor_files:
            SBR[file_names] = all_results.loc[counter:counter+9,'episode_service_blocking_rate'].to_list()
            BBR[file_names] = all_results.loc[counter:counter + 9, 'episode_bit_rate_blocking_rate'].to_list()
            EF[file_names] = all_results.loc[counter:counter + 9, 'external_fragmentation_network_episode'].to_list()
            CF[file_names] = all_results.loc[counter:counter + 9, 'compactness_fragmentation_network_episode'].to_list()
            CFN[file_names] = all_results.loc[counter:counter + 9, 'compactness_network_fragmentation_network_episode'].to_list()
            counter = counter + 10

        for key, value in SBR.items():
            SP_SBR = value

        for key, value in BBR.items():
            SP_BBR = value

        for key, value in EF.items():
             SP_EF = value
        for key, value in CF.items():
            SP_CF = value

        for key, value in CFN.items():
            SP_CFN = value

        SP_SBR_load[move][load_counter] = np.mean(SP_SBR)
        SP_BBR_load[move][load_counter] = np.mean(SP_BBR)
        SP_EF_load[move][load_counter] = np.mean(SP_EF)
        SP_CF_load[move][load_counter] = np.mean(SP_CF)
        SP_CFN_load[move][load_counter] = np.mean(SP_CFN)

fig = plt.figure(figsize=[8.4, 4.8])

plt.plot(loads, SP_EF_load[0][:], '+-b', label='SP_EF')
plt.plot(loads, SP_EF_load[1][:], '+-r', label='SP_EF_10')
plt.plot(loads, SP_EF_load[2][:], '+-g', label='SP_EF_50')
plt.xlabel('load')
plt.ylabel('External Fragmentation')
plt.legend()
plt.savefig(f'{figures_floder}/External_fragmentation.pdf')
plt.savefig(f'{figures_floder}/External_fragmentation.svg')
plt.show()
plt.close()


fig = plt.figure(figsize=[8.4, 4.8])
plt.semilogy(loads, SP_SBR_load[0][:], '+-b', label = 'SP_SBR')
plt.semilogy(loads, SP_SBR_load[1][:], '+-r', label = 'SP_SBR_10')
plt.semilogy(loads, SP_SBR_load[2][:], '+-g', label = 'SP_SBR_50')
plt.xlabel('load')
plt.ylabel('Service Blocking Rate ')
plt.legend()
plt.savefig(f'{figures_floder}/service_blocking.svg')
plt.savefig(f'{figures_floder}/service_blocking.pdf')
plt.show()
plt.close()

fig = plt.figure(figsize=[8.4, 4.8])
plt.semilogy(loads, SP_BBR_load[0][:], '+-b', label = 'SP_BBR')
plt.semilogy(loads, SP_BBR_load[1][:], '+-r', label = 'SP_BBR_10')
plt.semilogy(loads, SP_BBR_load[2][:], '+-g', label = 'SP_BBR_50')
plt.xlabel('load')
plt.ylabel('BIt Blocking Rate ')
plt.legend()
plt.savefig(f'{figures_floder}/bit_blocking.svg')
plt.savefig(f'{figures_floder}/bit_blocking.pdf')
plt.show()
plt.close()

fig = plt.figure(figsize=[8.4, 4.8])
plt.plot(loads, SP_CF_load[0][:], '+-b', label='SP_CF')
plt.plot(loads, SP_CF_load[1][:], '+-r', label='SP_CF_10')
plt.plot(loads, SP_CF_load[2][:], '+-g', label='SP_CF_50')
plt.xlabel('load')
plt.ylabel('Compactness Fragmentation')
plt.legend()
plt.savefig(f'{figures_floder}/Compactness_fragmentation.pdf')
plt.savefig(f'{figures_floder}/Compactness_fragmentation.svg')
plt.show()
plt.close()
fig = plt.figure(figsize=[8.4, 4.8])
plt.plot(loads, SP_CFN_load[0][:], '+-b', label='SP_CFN')
plt.plot(loads, SP_CFN_load[1][:], '+-r', label='SP_CFN_10')
plt.plot(loads, SP_CFN_load[2][:], '+-g', label='SP_CFN_50')
plt.xlabel('load')
plt.ylabel('Compactness network Fragmentation')
plt.legend()
plt.savefig(f'{figures_floder}/Compactness_network_fragmentation.pdf')
plt.savefig(f'{figures_floder}/Compactness_network_fragmentation.svg')
plt.show()
plt.close()

