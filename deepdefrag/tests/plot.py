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
episodes = 10
episode_length =  40
incremental_traffic_percentage = 80

monitor_files = []
policies = []

# adding logging method
#log_dir = "./tmp/logrmsa-ppo/"
logging_dir = "./tmp/logrmsa-ppo/"
figures_floder = f'{logging_dir}/figures-{incremental_traffic_percentage}/'
os.makedirs(logging_dir, exist_ok=True)
os.makedirs(figures_floder, exist_ok=True)
topology_name = 'gbn'
topology_name = 'nobel-us'
topology_name = 'germany50'
with open(f'../examples/topologies/nsfnet_chen_eon_5-paths.h5', 'rb') as f:
    topology = pickle.load(f)
min_load = 18
max_load = 60
step_length = 8
steps = int((max_load - min_load)/step_length) +1
Random_SBR_load = np.zeros(steps)
SP_SBR_load = np.zeros(steps)
SPAFF_SBR_load = np.zeros(steps)
LLP_SBR_load = np.zeros(steps)
FAAR_SBR_load = np.zeros(steps)

Random_BBR_load = np.zeros(steps)
SP_BBR_load = np.zeros(steps)
SPAFF_BBR_load = np.zeros(steps)
LLP_BBR_load = np.zeros(steps)
FAAR_BBR_load = np.zeros(steps)

Random_EF_load = np.zeros(steps)
SP_EF_load = np.zeros(steps)
SPAFF_EF_load = np.zeros(steps)
LLP_EF_load = np.zeros(steps)
FAAR_EF_load = np.zeros(steps)

Random_EFD_load = np.zeros(steps)
SP_EFD_load = np.zeros(steps)
SPAFF_EFD_load = np.zeros(steps)
LLP_EFD_load = np.zeros(steps)
FAAR_EFD_load = np.zeros(steps)

Random_CF_load = np.zeros(steps)
SP_CF_load = np.zeros(steps)
SPAFF_CF_load = np.zeros(steps)
LLP_CF_load = np.zeros(steps)
FAAR_CF_load = np.zeros(steps)

Random_CFN_load = np.zeros(steps)
SP_CFN_load = np.zeros(steps)
SPAFF_CFN_load = np.zeros(steps)
LLP_CFN_load = np.zeros(steps)
FAAR_CFN_load = np.zeros(steps)

Random_DDA_load = np.zeros(steps)
SP_DDA_load = np.zeros(steps)
SPAFF_DDA_load = np.zeros(steps)
LLP_DDA_load = np.zeros(steps)
FAAR_DDA_load = np.zeros(steps)

Random_DDP_load = np.zeros(steps)
SP_DDP_load = np.zeros(steps)
SPAFF_DDP_load = np.zeros(steps)
LLP_DDP_load = np.zeros(steps)
FAAR_DDP_load = np.zeros(steps)

Random_BBRF_load = np.zeros(steps)
SP_BBRF_load = np.zeros(steps)
SPAFF_BBRF_load = np.zeros(steps)
LLP_BBRF_load = np.zeros(steps)
FAAR_BBRF_load = np.zeros(steps)


Random_SBRF_load = np.zeros(steps)
SP_SBRF_load = np.zeros(steps)
SPAFF_SBRF_load = np.zeros(steps)
LLP_SBRF_load = np.zeros(steps)
FAAR_SBRF_load = np.zeros(steps)

Random_CFD_load = np.zeros(steps)
SP_CFD_load = np.zeros(steps)
SPAFF_CFD_load = np.zeros(steps)
LLP_CFD_load = np.zeros(steps)
FAAR_CFD_load = np.zeros(steps)


Random_SBR100_load = np.zeros(steps)
SP_SBR100_load = np.zeros(steps)
SPAFF_SBR100_load = np.zeros(steps)
LLP_SBR100_load = np.zeros(steps)
FAAR_SBR100_load = np.zeros(steps)

Random_SBR200_load = np.zeros(steps)
SP_SBR200_load = np.zeros(steps)
SPAFF_SBR200_load = np.zeros(steps)
LLP_SBR200_load = np.zeros(steps)
FAAR_SBR200_load = np.zeros(steps)

Random_SBR400_load = np.zeros(steps)
SP_SBR400_load = np.zeros(steps)
SPAFF_SBR400_load = np.zeros(steps)
LLP_SBR400_load = np.zeros(steps)
FAAR_SBR400_load = np.zeros(steps)

loads = np.zeros(steps)

for load_counter, load_traffic in enumerate(range(min_load,max_load,step_length)):
    log_dir = f'{logging_dir}logs_{load_traffic}_{episode_length}_{incremental_traffic_percentage}/'
    os.makedirs(log_dir, exist_ok=True)

    loads[load_counter] = load_traffic
    # load results
    all_results = load_results(log_dir)
    # Defining monitor files based on the below command does not work well, so i defined it statistically
    #monitor_files = get_monitor_files(log_dir)
    monitor_files = ['./tmp/logrmsa-ppo/Random.monitor.csv','./tmp/logrmsa-ppo/SP.monitor.csv','./tmp/logrmsa-ppo/SP-AFF.monitor.csv', './tmp/logrmsa-ppo/LLP.monitor.csv','./tmp/logrmsa-ppo/FAAR.monitor.csv' ]
    SBR = dict()
    BBR= dict()
    EF= dict()
    CF= dict()
    CFN = dict()
    DDA = dict()
    DDP = dict()
    BBRF = dict()
    SBRF = dict()
    EFD = dict()
    CFD = dict()
    SBR100 = dict()
    SBR200 = dict()
    SBR400 = dict()
    counter = 0

    # the 10 and 9 that is used in these plotting functions are related to episode lengths.
    # these are because of all_result.loc() function
    for file_names in monitor_files:
        SBR [file_names] = all_results.loc[counter:counter+9,'episode_service_blocking_rate'].to_list()
        BBR[file_names] = all_results.loc[counter:counter + 9, 'episode_bit_rate_blocking_rate'].to_list()
        EF[file_names] = all_results.loc[counter:counter + 9, 'external_fragmentation_network_episode'].to_list()
        CF[file_names] = all_results.loc[counter:counter + 9, 'compactness_fragmentation_network_episode'].to_list()
        CFN[file_names] = all_results.loc[counter:counter + 9, 'compactness_network_fragmentation_network_episode'].to_list()
        DDA[file_names] = all_results.loc[counter:counter + 9,'delay_deviation_absolute'].to_list()
        DDP[file_names] = all_results.loc[counter:counter + 9, 'delay_deviation_percentage'].to_list()
        BBRF[file_names] = all_results.loc[counter:counter + 9, 'bit_rate_blocking_fragmentation'].to_list()
        SBRF[file_names] = all_results.loc[counter:counter + 9, 'service_blocking_rate_fragmentation'].to_list()
        EFD[file_names] = all_results.loc[counter:counter + 9, 'external_fragmentation_deviation'].to_list()
        CFD[file_names] = all_results.loc[counter:counter + 9, 'compactness_fragmentation_deviation'].to_list()
        SBR100[file_names] = all_results.loc[counter:counter + 9, 'service_blocking_rate_100'].to_list()
        SBR200[file_names] = all_results.loc[counter:counter + 9, 'service_blocking_rate_200'].to_list()
        SBR400[file_names] = all_results.loc[counter:counter + 9, 'service_blocking_rate_400'].to_list()
        counter = counter + 10

    for key, value in SBR.items():
        if 'LLP' in key:
            LLP_SBR = value
        elif 'Random' in key:
            Random_SBR = value
        elif 'SP-AFF' in key:
            SPAFF_SBR = value
        elif 'FAAR' in key:
            FAAR_SBR = value
        else:
            SP_SBR = value


    for key, value in BBR.items():
        if 'LLP' in key:
            LLP_BBR = value
        elif 'Random' in key:
            Random_BBR = value
        elif 'SP-AFF' in key:
            SPAFF_BBR = value
        elif 'FAAR' in key:
            FAAR_BBR = value
        else:
            SP_BBR = value

    for key, value in EF.items():
        if 'LLP' in key:
            LLP_EF = value
        elif 'Random' in key:
            Random_EF = value
        elif 'SP-AFF' in key:
            SPAFF_EF = value
        elif 'FAAR' in key:
            FAAR_EF = value
        else:
            SP_EF = value
    for key, value in CF.items():
        if 'LLP' in key:
            LLP_CF = value
        elif 'Random' in key:
            Random_CF = value
        elif 'SP-AFF' in key:
            SPAFF_CF = value
        elif 'FAAR' in key:
            FAAR_CF = value
        else:
            SP_CF = value

    for key, value in CFN.items():
        if 'LLP' in key:
            LLP_CFN = value
        elif 'Random' in key:
            Random_CFN = value
        elif 'SP-AFF' in key:
            SPAFF_CFN = value
        elif 'FAAR' in key:
            FAAR_CFN = value
        else:
            SP_CFN = value

    for key, value in DDA.items():
        if 'LLP' in key:
            LLP_DDA = value
        elif 'Random' in key:
            Random_DDA = value
        elif 'SP-AFF' in key:
            SPAFF_DDA = value
        elif 'FAAR' in key:
            FAAR_DDA = value
        else:
            SP_DDA = value
    for key, value in DDP.items():
        if 'LLP' in key:
            LLP_DDP = value
        elif 'Random' in key:
            Random_DDP = value
        elif 'SP-AFF' in key:
            SPAFF_DDP = value
        elif 'FAAR' in key:
            FAAR_DDP = value
        else:
            SP_DDP = value


    for key, value in BBR.items():
        if 'LLP' in key:
            LLP_BBR = value
        elif 'Random' in key:
            Random_BBR = value
        elif 'SP-AFF' in key:
            SPAFF_BBR = value
        elif 'FAAR' in key:
            FAAR_BBR = value
        else:
            SP_BBR = value


    for key, value in BBRF.items():
        if 'LLP' in key:
            LLP_BBRF = value
        elif 'Random' in key:
            Random_BBRF = value
        elif 'SP-AFF' in key:
            SPAFF_BBRF = value
        elif 'FAAR' in key:
            FAAR_BBRF = value
        else:
            SP_BBRF = value


    for key, value in SBRF.items():
        if 'LLP' in key:
            LLP_SBRF = value
        elif 'Random' in key:
            Random_SBRF = value
        elif 'SP-AFF' in key:
            SPAFF_SBRF = value
        elif 'FAAR' in key:
            FAAR_SBRF = value
        else:
            SP_SBRF = value


    for key, value in EFD.items():
        if 'LLP' in key:
            LLP_EFD = value
        elif 'Random' in key:
            Random_EFD = value
        elif 'SP-AFF' in key:
            SPAFF_EFD = value
        elif 'FAAR' in key:
            FAAR_EFD = value
        else:
            SP_EFD = value


    for key, value in CFD.items():
        if 'LLP' in key:
            LLP_CFD = value
        elif 'Random' in key:
            Random_CFD = value
        elif 'SP-AFF' in key:
            SPAFF_CFD = value
        elif 'FAAR' in key:
            FAAR_CFD = value
        else:
            SP_CFD = value


    for key, value in SBR100.items():
        if 'LLP' in key:
            LLP_SBR100 = value
        elif 'Random' in key:
            Random_SBR100 = value
        elif 'SP-AFF' in key:
            SPAFF_SBR100 = value
        elif 'FAAR' in key:
            FAAR_SBR100 = value
        else:
            SP_SBR100 = value


    for key, value in SBR200.items():
        if 'LLP' in key:
            LLP_SBR200 = value
        elif 'Random' in key:
            Random_SBR200 = value
        elif 'SP-AFF' in key:
            SPAFF_SBR200 = value
        elif 'FAAR' in key:
            FAAR_SBR200 = value
        else:
            SP_SBR200 = value


    for key, value in SBR400.items():
        if 'LLP' in key:
            LLP_SBR400 = value
        elif 'Random' in key:
            Random_SBR400 = value
        elif 'SP-AFF' in key:
            SPAFF_SBR400 = value
        elif 'FAAR' in key:
            FAAR_SBR400 = value
        else:
            SP_SBR400 = value


    Random_SBR_load[load_counter] = np.mean(Random_SBR)
    SP_SBR_load[load_counter] = np.mean(SP_SBR)
    SPAFF_SBR_load[load_counter] = np.mean(SPAFF_SBR)
    LLP_SBR_load[load_counter] = np.mean(LLP_SBR)
    FAAR_SBR_load[load_counter] = np.mean(FAAR_SBR)

    Random_BBR_load[load_counter] = np.mean(Random_BBR)
    SP_BBR_load[load_counter] = np.mean(SP_BBR)
    SPAFF_BBR_load[load_counter] = np.mean(SPAFF_BBR)
    LLP_BBR_load[load_counter] = np.mean(LLP_BBR)
    FAAR_BBR_load[load_counter] = np.mean(FAAR_BBR)



    Random_EF_load[load_counter] = np.mean(Random_EF)
    SP_EF_load[load_counter] = np.mean(SP_EF)
    SPAFF_EF_load[load_counter] = np.mean(SPAFF_EF)
    LLP_EF_load[load_counter] = np.mean(LLP_EF)
    FAAR_EF_load[load_counter] = np.mean(FAAR_EF)

    Random_CF_load[load_counter] = np.mean(Random_CF)
    SP_CF_load[load_counter] = np.mean(SP_CF)
    SPAFF_CF_load[load_counter] = np.mean(SPAFF_CF)
    LLP_CF_load[load_counter] = np.mean(LLP_CF)
    FAAR_CF_load[load_counter] = np.mean(FAAR_CF)

    Random_CFN_load[load_counter] = np.mean(Random_CFN)
    SP_CFN_load[load_counter] = np.mean(SP_CFN)
    SPAFF_CFN_load[load_counter] = np.mean(SPAFF_CFN)
    LLP_CFN_load[load_counter] = np.mean(LLP_CFN)
    FAAR_CFN_load[load_counter] = np.mean(FAAR_CFN)


    # delay calculation
    Random_DDA_load[load_counter] = np.mean(Random_DDA)
    SP_DDA_load[load_counter] = np.mean(SP_DDA)
    SPAFF_DDA_load[load_counter] = np.mean(SPAFF_DDA)
    LLP_DDA_load[load_counter] = np.mean(LLP_DDA)
    FAAR_DDA_load[load_counter] = np.mean(FAAR_DDA)

    Random_DDP_load[load_counter] = np.mean(Random_DDP)
    SP_DDP_load[load_counter] = np.mean(SP_DDP)
    SPAFF_DDP_load[load_counter] = np.mean(SPAFF_DDP)
    LLP_DDP_load[load_counter] = np.mean(LLP_DDP)
    FAAR_DDP_load[load_counter] = np.mean(FAAR_DDP)



    Random_BBRF_load[load_counter] = np.mean(Random_BBRF)
    SP_BBRF_load[load_counter] = np.mean(SP_BBRF)
    SPAFF_BBRF_load[load_counter] = np.mean(SPAFF_BBRF)
    LLP_BBRF_load[load_counter] = np.mean(LLP_BBRF)
    FAAR_BBRF_load[load_counter] = np.mean(FAAR_BBRF)

    Random_SBRF_load[load_counter] = np.mean(Random_SBRF)
    SP_SBRF_load[load_counter] = np.mean(SP_SBRF)
    SPAFF_SBRF_load[load_counter] = np.mean(SPAFF_SBRF)
    LLP_SBRF_load[load_counter] = np.mean(LLP_SBRF)
    FAAR_SBRF_load[load_counter] = np.mean(FAAR_SBRF)


    Random_EFD_load[load_counter] = np.mean(Random_EFD)
    SP_EFD_load[load_counter] = np.mean(SP_EFD)
    SPAFF_EFD_load[load_counter] = np.mean(SPAFF_EFD)
    LLP_EFD_load[load_counter] = np.mean(LLP_EFD)
    FAAR_EFD_load[load_counter] = np.mean(FAAR_EFD)


    Random_CFD_load[load_counter] = np.mean(Random_CFD)
    SP_CFD_load[load_counter] = np.mean(SP_CFD)
    SPAFF_CFD_load[load_counter] = np.mean(SPAFF_CFD)
    LLP_CFD_load[load_counter] = np.mean(LLP_CFD)
    FAAR_CFD_load[load_counter] = np.mean(FAAR_CFD)


    Random_SBR100_load[load_counter] = np.mean(Random_SBR100)
    SP_SBR100_load[load_counter] = np.mean(SP_SBR100)
    SPAFF_SBR100_load[load_counter] = np.mean(SPAFF_SBR100)
    LLP_SBR100_load[load_counter] = np.mean(LLP_SBR100)
    FAAR_SBR100_load[load_counter] = np.mean(FAAR_SBR100)


    Random_SBR200_load[load_counter] = np.mean(Random_SBR200)
    SP_SBR200_load[load_counter] = np.mean(SP_SBR200)
    SPAFF_SBR200_load[load_counter] = np.mean(SPAFF_SBR200)
    LLP_SBR200_load[load_counter] = np.mean(LLP_SBR200)
    FAAR_SBR200_load[load_counter] = np.mean(FAAR_SBR200)

    Random_SBR400_load[load_counter] = np.mean(Random_SBR400)
    SP_SBR400_load[load_counter] = np.mean(SP_SBR400)
    SPAFF_SBR400_load[load_counter] = np.mean(SPAFF_SBR400)
    LLP_SBR400_load[load_counter] = np.mean(LLP_SBR400)
    FAAR_SBR400_load[load_counter] = np.mean(FAAR_SBR400)



# plotting figure for each frag mentation metric
fig = plt.figure(figsize=[8.4, 4.8])
#plt.plot(loads, Random_EF_load, '+-r', label = 'Random_EF')
plt.plot(loads, SP_EF_load, '+-k', label = 'SP_EF')
#plt.plot(loads, SPAFF_EF_load, '+-g', label = 'SAPFF_EF')
#plt.plot(loads, LLP_EF_load, '+-b', label = 'LLP_EF')
plt.plot(loads, FAAR_EF_load, '+-y', label = 'FAAR_EF')

#plt.plot(loads, Random_EFD_load, '--r', label = 'Random_EFD')
plt.plot(loads, SP_EFD_load, '--k', label = 'SP_EFD')
#plt.plot(loads, SPAFF_EFD_load, '--g', label = 'SAPFF_EFD')
#plt.plot(loads, LLP_EFD_load, '--b', label = 'LLP_EFD')
plt.plot(loads, FAAR_EFD_load, '--y', label = 'FAAR_EFD')
plt.xlabel('load')
plt.ylabel('External Fragmentation')
plt.legend()
plt.savefig(f'{figures_floder}/External_fragmentation.pdf')
plt.savefig(f'{figures_floder}/External_fragmentation.svg')
plt.show()
plt.close()

fig = plt.figure(figsize=[8.4, 4.8])
#plt.plot(loads, Random_CF_load, '+-r', label = 'Random_CF')
plt.plot(loads, SP_CF_load, '+-k', label = 'SP_CF')
#plt.plot(loads, SPAFF_CF_load, '+-g', label = 'SAPFF_CF')
#plt.plot(loads, LLP_CF_load, '+-b', label = 'LLP_CF')
plt.plot(loads, FAAR_CF_load, '+-y', label = 'FAAR_CF')

#plt.plot(loads, Random_CFD_load, '--r', label = 'Random_CFD')
plt.plot(loads, SP_CFD_load, '--k', label = 'SP_CFD')
#plt.plot(loads, SPAFF_CFD_load, '--g', label = 'SAPFF_CFD')
#plt.plot(loads, LLP_CFD_load, '--b', label = 'LLP_CFD')
plt.plot(loads, FAAR_CFD_load, '--y', label = 'FAAR_CFD')


plt.xlabel('load')
plt.ylabel('Average on Link Compactness Fragmentation')
plt.legend()
plt.savefig(f'{figures_floder}/Compactness_link_fragmentation.pdf')
plt.savefig(f'{figures_floder}/Compactness_link_fragmentation.svg')
plt.show()
plt.close()

fig = plt.figure(figsize=[8.4, 4.8])
#plt.plot(loads, Random_CFN_load, '+-r', label = 'Random_CFN')
plt.plot(loads, SP_CFN_load, '+-k', label = 'SP_CFN')
#plt.plot(loads, SPAFF_CFN_load, '+-g', label = 'SAPFF_CFN')
#plt.plot(loads, LLP_CFN_load, '+-b', label = 'LLP_CFN')
plt.plot(loads, FAAR_CFN_load, '+-y', label = 'FAAR_CFN')
plt.xlabel('load')
plt.ylabel(' Network Compactness Fragmentation')
plt.legend()
plt.savefig(f'{figures_floder}/Compactness_network_fragmentation.pdf')
plt.savefig(f'{figures_floder}/Compactness_network_fragmentation.svg')
plt.show()
plt.close()

#plotting blocking rate


fig = plt.figure(figsize=[8.4, 4.8])
#plt.semilogy(loads, Random_SBR_load, '+-r', label = 'Random_SBR')
plt.semilogy(loads, SP_SBR_load, '+-k', label = 'SP_SBR')
#plt.semilogy(loads, SPAFF_SBR_load, '+-g', label = 'SAPFF_SBR')
#plt.semilogy(loads, LLP_SBR_load, '+-b', label = 'LLP_SBR')
plt.semilogy(loads, FAAR_SBR_load, '+-y', label = 'FAAR_SBR')
plt.xlabel('load')
plt.ylabel('Service Blocking Rate ')
plt.legend()
plt.savefig(f'{figures_floder}/service_blocking.svg')
plt.savefig(f'{figures_floder}/service_blocking.pdf')
plt.show()
plt.close()

fig = plt.figure(figsize=[8.4, 4.8])
#plt.semilogy(loads, Random_BBR_load, '+-r', label = 'Random_BBR')
plt.semilogy(loads, SP_BBR_load, '+-k', label = 'SP_BBR')
#plt.semilogy(loads, SPAFF_BBR_load, '+-g', label = 'SAPFF_BBR')
#plt.semilogy(loads, LLP_BBR_load, '+-b', label = 'LLP_BBR')
plt.semilogy(loads, FAAR_BBR_load, '+-y', label = 'FAAR_BBR')
plt.xlabel('load')
plt.ylabel('Bit Blocking Rate')
plt.legend()
plt.savefig(f'{figures_floder}/bit_blocking.pdf')
plt.savefig(f'{figures_floder}/bit_blocking.svg')
plt.show()
plt.close()


# plotting percentage delay dviation
fig = plt.figure(figsize=[8.4, 4.8])
#plt.plot(loads, Random_DDA_load, '+-r', label = 'Random_DDA')
plt.plot(loads, SP_DDA_load, '+-k', label = 'SP_DDA')
#plt.plot(loads, SPAFF_DDA_load, '+-g', label = 'SAPFF_DDA')
#plt.plot(loads, LLP_DDA_load, '+-b', label = 'LLP_DDA')
plt.plot(loads, FAAR_DDA_load, '+-y', label = 'FAAR_DDA')
plt.xlabel('load')
plt.ylabel('Delay Deviation Absolute')
plt.legend()
plt.savefig(f'{figures_floder}/Delay_Deviation_Absolute.pdf')
plt.savefig(f'{figures_floder}/Delay_Deviation_Absolute.svg')
plt.show()
plt.close()


fig = plt.figure(figsize=[8.4, 4.8])
#plt.plot(loads, Random_DDP_load, '+-r', label = 'Random_DDP')
plt.plot(loads, SP_DDP_load, '+-k', label = 'SP_DDP')
#plt.plot(loads, SPAFF_DDP_load, '+-g', label = 'SAPFF_DDP')
#plt.plot(loads, LLP_DDP_load, '+-b', label = 'LLP_DDP')
plt.plot(loads, FAAR_DDP_load, '+-y', label = 'FAAR_DDP')
plt.xlabel('load')
plt.ylabel('Delay Deviation Percentage')
plt.legend()
plt.savefig(f'{figures_floder}/Delay_Deviation_Percentage.pdf')
plt.savefig(f'{figures_floder}/Delay_Deviation_Percentage.svg')
plt.show()
plt.close()



# plotting the bit rate blocking due to fragmentation


fig = plt.figure(figsize=[8.4, 4.8])
#plt.semilogy(loads, Random_BBRF_load, '+-r', label = 'Random_BBRF')
plt.semilogy(loads, SP_BBRF_load, '+-k', label = 'SP_BBRF')
#plt.semilogy(loads, SPAFF_BBRF_load, '+-g', label = 'SAPFF_BBRF')
#plt.semilogy(loads, LLP_BBRF_load, '+-b', label = 'LLP_BBRF')
plt.semilogy(loads, FAAR_BBRF_load, '+-y', label = 'FAAR_BBRF')
plt.xlabel('load')
plt.ylabel('Bit Blocking Rate due to fragmentation')
plt.legend()
plt.savefig(f'{figures_floder}/bit_blocking_fragmentation.pdf')
plt.savefig(f'{figures_floder}/bit_blocking_fragmentation.svg')
plt.show()
plt.close()


fig = plt.figure(figsize=[8.4, 4.8])
#plt.semilogy(loads, Random_SBRF_load, '+-r', label = 'Random_SBRF')
plt.semilogy(loads, SP_SBRF_load, '+-k', label = 'SP_SBRF')
#plt.semilogy(loads, SPAFF_SBRF_load, '+-g', label = 'SAPFF_SBRF')
#plt.semilogy(loads, LLP_SBRF_load, '+-b', label = 'LLP_SBRF')
plt.semilogy(loads, FAAR_SBRF_load, '+-y', label = 'FAAR_SBRF')
plt.xlabel('load')
plt.ylabel('Service Blocking Rate due to fragmentation')
plt.legend()
plt.savefig(f'{figures_floder}/service_blocking_fragmentation.pdf')
plt.savefig(f'{figures_floder}/service_blocking_fragmentation.svg')
plt.show()
plt.close()


fig = plt.figure(figsize=[8.4, 4.8])
plt.semilogy(loads, SPAFF_SBR200_load, '--g', label = 'SAPFF_SBR200')
plt.semilogy(loads, SPAFF_SBR100_load, '+-g', label = 'SAPFF_SBR100')
plt.semilogy(loads, SPAFF_SBR400_load, 's-g', label = 'SAPFF_SBR400')


plt.xlabel('load')
plt.ylabel('Service Blocking Rate ')
plt.legend()
plt.savefig(f'{figures_floder}/service_blockingSAPFF.svg')
plt.savefig(f'{figures_floder}/service_blockingSAPFF.pdf')
plt.show()
plt.close()


fig = plt.figure(figsize=[8.4, 4.8])

plt.semilogy(loads, FAAR_SBR100_load, '+-y', label = 'FAAR_SBR100')
plt.semilogy(loads, FAAR_SBR200_load, '--y', label = 'FAAR_SBR200')
plt.semilogy(loads, FAAR_SBR400_load, 's-y', label = 'FAAR_SBR400')


plt.xlabel('load')
plt.ylabel('Service Blocking Rate ')
plt.legend()
plt.savefig(f'{figures_floder}/service_blockingFAAR.svg')
plt.savefig(f'{figures_floder}/service_blockingFAAR.pdf')
plt.show()
plt.close()



fig = plt.figure(figsize=[8.4, 4.8])

plt.semilogy(loads, LLP_SBR200_load, '--b', label = 'LLP_SBR200')
plt.semilogy(loads, LLP_SBR100_load, '+-b', label = 'LLP_SBR100')
plt.semilogy(loads, LLP_SBR400_load, 's-b', label = 'LLP_SBR400')


plt.xlabel('load')
plt.ylabel('Service Blocking Rate ')
plt.legend()
plt.savefig(f'{figures_floder}/service_blockingLLP.svg')
plt.savefig(f'{figures_floder}/service_blockingLLP.pdf')
plt.show()
plt.close()