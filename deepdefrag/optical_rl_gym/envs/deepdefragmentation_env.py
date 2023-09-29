import gym
import copy
import math
import heapq
import logging
import functools
import random
import numpy as np
from optical_rl_gym.utils import Path, Service, DefragmentationOption
import os
import random
import os
from .rmsa_env import RMSAEnv
from .defragmentation_env import DefragmentationEnv


class DeepDefragmentationEnv(DefragmentationEnv):
    def __init__(self, topology=None,
                 episode_length=1000,
                 load=100,
                 mean_service_holding_time=10800.0,
                 num_spectrum_resources=100,
                 node_request_probabilities=None,
                 seed=None,
                 k_paths=5,
                 allow_rejection=False,
                 incremental_traffic_percentage=80,
                 rmsa_function=None,
                 number_options=7,
                 penalty_cycle=-0.25,
                 penalty_movement=-3,
                 fragmented_constraint = False,
                 only_FF = True):
        super().__init__(topology,
                         episode_length=episode_length,
                         load=load,
                         mean_service_holding_time=mean_service_holding_time,
                         num_spectrum_resources=num_spectrum_resources,
                         node_request_probabilities=node_request_probabilities,
                         seed=seed, allow_rejection=allow_rejection,
                         k_paths=k_paths,
                         incremental_traffic_percentage=incremental_traffic_percentage,
                         rmsa_function=rmsa_function,
                         number_options = number_options,
                         penalty_cycle=penalty_cycle,
                        penalty_movement=penalty_movement,
                         fragmented_constraint = fragmented_constraint,
                         only_FF = only_FF,
                 )

        shape = (1 + 2 * self.topology.number_of_nodes() + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 +1 +1 + 1 + 1 + 1  )*self.number_options + 1
        # shape = (1 + 2 * self.topology.number_of_nodes() + 1 + 1 + 1 + 1 + 1  )*self.number_options + 1
        # shape = (1  + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 )*self.number_options + 1
        self.action_space = gym.spaces.Discrete(self.number_options)
        self.observation_space = gym.spaces.Box(low=0, high=1, dtype=np.float64, shape=(shape,))
        self.action_space.seed(self.rand_seed) # why?
        self.observation_space.seed(self.rand_seed) #why
        self.reset(only_counters=False)# why

    def step(self, action: int):

        if action >= self.number_options:
            raise Exception(" the action should be within the options")
            # this is to penalize the agent when choose non available option
        if action >= len(self.defragmentation_options_available):
            # TODO: do not serve traffic
            # obs, reward, flag, info = super().step(self.defragmentation_options_available[0])
            return self.observation(), -1, 0, super().get_info()
        return super().step(self.defragmentation_options_available[action])

    def observation(self):
        source_destination_tau = np.zeros((self.number_options, 2, self.topology.number_of_nodes()))
        options_obs = np.full((self.number_options, 1+1+1+1+1+1+1+1+1+1+1+1 +1 +1 +1  ), fill_value=-1.)
        # options_obs = np.full((self.number_options, 1+1+1+1+1+1), fill_value=-1.)
        index_option = 0
        all_indexes =[]
        options_found= 0
        selected_defragmentation_options=self.defragmentation_options

        #TODO : make it intelligence, and develop random allocation, and also 10:400

        # if len(selected_defragmentation_options) > 5:
        #     for count in range(1, len(selected_defragmentation_options)):
        #         if selected_defragmentation_options[1].service.number_slots < selected_defragmentation_options[count].service.number_slots: ## bigger size
        #             selected_defragmentation_options[1] = selected_defragmentation_options[count]
        #             selected_defragmentation_options[count] = selected_defragmentation_options[1]
        #         elif selected_defragmentation_options[2].service.number_slots > selected_defragmentation_options[count].service.number_slots: ## smaller size
        #             selected_defragmentation_options[2] = selected_defragmentation_options[count]
        #             selected_defragmentation_options[count] = selected_defragmentation_options[2] ## switvhing the situation
        #         elif selected_defragmentation_options[3].service.route.hops < selected_defragmentation_options[count].service.route.hops:
        #             selected_defragmentation_options[3] = selected_defragmentation_options[count]
        #             self.defragmentation_options[count] = selected_defragmentation_options[3]
        #         elif  selected_defragmentation_options[4].service.arrival_time + selected_defragmentation_options[4].service.holding_time\
        #             < self.defragmentation_options[count].service.arrival_time +selected_defragmentation_options[count].service.holding_time:
        #             selected_defragmentation_options[4] = selected_defragmentation_options[count]
        #             self.defragmentation_options[count] = selected_defragmentation_options[4]

        self.defragmentation_options_available = [DefragmentationOption(0, -1, 0, 0,0,0,0,0)]
        # while len(self.defragmentation_options_available) - 1 < self.number_options and len(self.defragmentation_options) > 0:
        while len(self.defragmentation_options_available) - 1 < self.number_options and len(selected_defragmentation_options) > 0:
            service_to_defrag = selected_defragmentation_options[index_option].service
            # new_index_option = random.randint(0, len(self.defragmentation_options)-1)
            # service_to_defrag = self.defragmentation_options[new_index_option].service
            # if service_to_defrag not in self.defragmented_services and\
            #  self.defragmentation_options[index_option].starting_slot < service_to_defrag.initial_slot and\
            #      self.defragmentation_options[index_option].start_of_block:
            # if new_index_option not in all_indexes:
            #     all_indexes.append(new_index_option)
            if service_to_defrag not in self.defragmented_services:
                options_found = len(self.defragmentation_options_available) - 1
                self.defragmentation_options_available.append(selected_defragmentation_options[index_option])
                # self.defragmentation_options_available.append(self.defragmentation_options[new_index_option])
                min_node = min(service_to_defrag.source_id, service_to_defrag.destination_id)
                max_node = max(service_to_defrag.source_id, service_to_defrag.destination_id)
                source_destination_tau[options_found, 0, min_node] = 1
                source_destination_tau[options_found, 1, max_node] = 1
                options_obs[options_found, 0] = len(service_to_defrag.route.node_list) # the number of links
                options_obs[options_found, 1] = (service_to_defrag.arrival_time/self.current_time)# time of arrival could be normalized to the current time,
                # or arrival.time- current time divided by current time
                options_obs[options_found, 2] = (service_to_defrag.number_slots - 5.5)/3.5  # shall be normalized?
                # options_obs[options_found, 3] = (service_to_defrag.arrival_time + service_to_defrag.holding_time - self.current_time)\
                #                         / self.mean_service_holding_time  # check normalization
                options_obs[options_found, 3] = 2 * (service_to_defrag.initial_slot - .5 * self.num_spectrum_resources) / self.num_spectrum_resources

                ## finding available slots
                available_slots = self.get_available_slots(service_to_defrag.route)
                options_obs[options_found, 4] = 2 * (np.sum(available_slots) - .5 * self.num_spectrum_resources) / self.num_spectrum_resources # total number available FSs
                # options_obs[options_found, 6] = 2*(self.defragmentation_options[index_option].starting_slot  - .5 * self.num_spectrum_resources) / self.num_spectrum_resources # possible starting slot
                # options_obs[options_found, 7] = 2*(self.defragmentation_options[index_option].size_of_free_block  - .5 * self.num_spectrum_resources) / self.num_spectrum_resources # size of free block
                # options_obs[options_found, 8] = self.defragmentation_options[index_option].start_of_block  # start or end of free block

                options_obs[options_found, 5] = 2*(selected_defragmentation_options[index_option].starting_slot  - .5 * self.num_spectrum_resources) / self.num_spectrum_resources # possible starting slot
                # options_obs[options_found, 7] = 2*(selected_defragmentation_options[index_option].size_of_free_block  - .5 * self.num_spectrum_resources) / self.num_spectrum_resources # size of free block
                options_obs[options_found, 6] = (selected_defragmentation_options[index_option].size_of_free_block  - 5.5) / 3.5 # size of free block
                # options_obs[options_found, 8] = selected_defragmentation_options[index_option].start_of_block  # start or end of free block # if it is always the begining, so no need for this
                # options_obs[options_found, 8] = (selected_defragmentation_options[index_option].left_side_free_slots) -5.5 / 3.5
                # options_obs[options_found, 9] = (selected_defragmentation_options[index_option].right_side_free_slots - 5.5)/ 3.5
                options_obs[options_found, 7] = (selected_defragmentation_options[index_option].r_frag_before)
                options_obs[options_found, 8] = selected_defragmentation_options[index_option].r_frag_after
                options_obs[options_found, 9] = (selected_defragmentation_options[index_option].number_cut_before)/3
                options_obs[options_found, 10] = (selected_defragmentation_options[index_option].number_cut_after)/3
                options_obs[options_found, 11] = (selected_defragmentation_options[index_option].frag_size_before -5.5)/3.5
                options_obs[options_found, 12] = (selected_defragmentation_options[index_option].frag_size_after - 5.5) / 3.5
                options_obs[options_found, 13] = selected_defragmentation_options[index_option].shanon_entropy_before
                options_obs[options_found, 14] = selected_defragmentation_options[index_option].shanon_entropy_after

            index_option += 1
            if index_option >= len(selected_defragmentation_options):
                break
        self.number_existing_options = len(self.defragmentation_options_available)
        reshaped_option = options_obs.reshape((1, np.prod(options_obs.shape)))
        if self.fragmentation_flag:
            reshaped_option = np.append(reshaped_option, 1)
        else:
            reshaped_option = np.append(reshaped_option, 0)
        return np.concatenate((source_destination_tau.reshape((1, np.prod(source_destination_tau.shape))),
                               reshaped_option.reshape((1, np.prod(reshaped_option.shape)))), axis=1)\
            .reshape(self.observation_space.shape)
        # return reshaped_option.reshape((1, np.prod(reshaped_option.shape)))

    def reset(self, only_counters=True):
        return super().reset(only_counters=only_counters)


def populate_network(env:DeepDefragmentationEnv) -> int:
    return 0
