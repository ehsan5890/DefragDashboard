import gym
import copy
import math
import heapq
import logging
import functools
import random
import numpy as np
from optical_rl_gym.utils import Path, Service
import os
import random
import pandas as pd

from optical_rl_gym.utils import plot_spectrum_assignment, plot_spectrum_assignment_and_waste, DefragmentationOption
from optical_rl_gym.envs.rmsa_env import shortest_path_first_fit, shortest_available_path_first_fit, \
    least_loaded_path_first_fit, SimpleMatrixObservation, Fragmentation_alignment_aware_RMSA

from itertools import chain
from operator import attrgetter
from .rmsa_env  import RMSAEnv



class DefragmentationEnv(RMSAEnv):
    metadata = {
        'metrics': ['service_blocking_rate', 'episode_service_blocking_rate',
                    'bit_rate_blocking_rate', 'episode_bit_rate_blocking_rate']
    }

    def __init__(self, topology=None,
                 episode_length=200,
                 load=10,
                 mean_service_holding_time=10800.0,
                 num_spectrum_resources=100,
                 node_request_probabilities=None,
                 bit_rate_lower_bound=25,
                 bit_rate_higher_bound=100,
                 seed=None,
                 k_paths=5,
                 allow_rejection=False,
                 reset=True,
                 incremental_traffic_percentage=80,
                 defragmentation_period=4,
                 rmsa_function=None,
                 number_options= 10,
                 penalty_movement= -0.1,
                 penalty_cycle= -0.3,
                 fragmented_constraint = False,
                 traffic_type= 1,
                 only_FF = True):
        super().__init__(topology,
                         episode_length=episode_length,
                         load=load,
                         mean_service_holding_time=mean_service_holding_time,
                         num_spectrum_resources=num_spectrum_resources,
                         node_request_probabilities=node_request_probabilities,
                         seed=seed, allow_rejection=allow_rejection,
                         k_paths=k_paths,
                         incremental_traffic_percentage=incremental_traffic_percentage, traffic_type = traffic_type,
                         reset=False)
        assert 'modulations' in self.topology.graph
        self.number_options = number_options
        # specific attributes for elastic optical networks
        self.defragmentation_period = defragmentation_period
        self.rmsa_function = rmsa_function
        self.service_to_defrag = None
        self.last_service_to_defrag = None
        self.last_new_initial_slot = 0
        self.last_old_initial_slot = 0
        self.defragmentation_options = []
        self.fragmented_service_constraint = fragmented_constraint
        # A counter to calculate the number of movements, and number of defragmentation procedure
        self.episode_num_moves = 0
        self.num_moves = 0
        self.num_moves_list = []


        self.episode_defragmentation_procedure = 0
        self.defragmentation_procedure = 0
        self.defragmentation_procedure_list = []
        # This flag is defined to figure out the fragmentation movement within steps.
        # It is used in the reward definition
        self.fragmentation_flag = False
        self.only_FF = only_FF
        self.rewards = []
        self.cumulative_rewards = []
        self.theta_list = []
        self.rfrag_before_list=[]
        self.rfrag_after_list = []
        self.r_frag_difference_list=[]
        self.num_cut_list_before=[]
        self.num_cut_list_after=[]
        self.shanon_entrophy_before_list = []
        self.shanon_entrophy_after_list= []
        self.frag_size_before_list = []
        self.frag_size_after_list = []
        self.sbr_list=[]
        self.counting_episode= 0
        self.previous_service_accepted = False
        self.previous_service = None


        self.calcualte_highest_metric_difference = True
        self.calcualte_highest_cut = True
        self.defragmented_services = []
        # a penalty for doing defragmentation, this should be tuned.
        self.fragmentation_penalty_cycle = penalty_cycle
        self.fragmentation_penalty_movement = penalty_movement
        # for implementing oldest first strategy, we need to define two auxiliary variable
        self.defragmentation_period_count = 0
        self.defragmentation_movement_period = 0
        self.number_existing_options = 0
        self.number_cuts = 0
        self.fragmentation_metric = 0
        self.total_shanon_entropy = 0

        self.blocked_services = []
        self.last_spectrum_slot_allocation = None

        # defining the reward function
        self.reward_cumulative = 0
        self.logger = logging.getLogger('rmsaenv')
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.warning(
                'Logging is enabled for DEBUG which generates a large number of messages. '
                'Set it to INFO if DEBUG is not necessary.')
        self._new_service = False
        if reset:
            self.reset(only_counters=False)

    def step(self, action):
        # If we want determine the final slot instead of moving directions
        service_to_defrag, new_initial_slot = action.service, action.starting_slot
        reward = 0
        # self.rfrag_before_list.append(action.r_frag_before)
        # self.rfrag_after_list.append(action.r_frag_after)
        # self.r_frag_difference_list.append(action.r_frag_after-action.r_frag_before)
        # self.num_cut_list_before.append(action.number_cut_before)
        # self.num_cut_list_after.append(action.number_cut_after)
        # self.shanon_entrophy_before_list.append(action.shanon_entropy_before)
        # self.shanon_entrophy_after_list.append(action.shanon_entropy_after)
        # self.frag_size_before_list.append(action.frag_size_before)
        # self.frag_size_after_list.append(action.frag_size_after)

        self.fragmentation_metric = 0
        self.number_cuts = 0

        self.last_spectrum_slot_allocation = copy.deepcopy(self.spectrum_slots_allocation)

        # self.fragmentation_metric += self._calculate_r_frag()
        # self.number_cuts += self._calculate_total_cuts()
        # self.rfrag_before_list.append(self.fragmentation_metric)
        # self.num_cut_list_before.append(self.number_cuts)
        #

        # doing defragmentation
        if new_initial_slot != -1:

            ### theta hints :  2 for end of cycle, 0 for simple reallocation, and 1 for serving traffic
            self.theta_list.append(0)
            self.episode_num_moves = self.episode_num_moves + 1
            self.num_moves = self.num_moves + 1
            self.service_to_defrag = service_to_defrag
            self.last_old_initial_slot = service_to_defrag.initial_slot
            self.last_service_to_defrag = service_to_defrag
            self.last_new_initial_slot = new_initial_slot

            self.defragmented_services.append(service_to_defrag)
            # r_frag_before = self._calculate_r_frag()
            self._move_path(new_initial_slot)
            # adding reward at the beginning of the cycle
            # if self.fragmentation_flag:
            #     reward = reward + self.fragmentation_penalty_movement
            # else:
            #     reward = reward + self.fragmentation_penalty_cycle + self.fragmentation_penalty_movement
            if action.r_frag_diff > 0 :
                reward = reward + 1 - np.log(action.r_frag_diff)/np.log(0.001) - self.fragmentation_penalty_movement
            elif action.r_frag_diff < 0:
                reward = reward - 1 + np.log(-action.r_frag_diff) / np.log(0.001) - self.fragmentation_penalty_movement
            else:
                reward = reward + np.log(row["SBR"] + 0.001)/np.log(0.001) - self.fragmentation_penalty_movement
            # reward = reward + self.fragmentation_penalty_movement + action.r_frag_after - action.r_frag_before
            self.fragmentation_flag = True
            # r_frag_after = self._calculate_r_frag()
            # r_frag_differences = r_frag_after - r_frag_before
            # reward = reward + 4*r_frag_differences
        else:
            path, initial_slot = self.rmsa_function(self)[0], self.rmsa_function(self)[1]
            if path < self.k_paths and initial_slot < self.num_spectrum_resources:  # action is for assigning a path
                slots = super().get_number_slots(self.k_shortest_paths[self.service.source, self.service.destination][path])
                self.logger.debug(
                    '{} processing action {} path {} and initial slot {} for {} slots'.format(self.service.service_id,
                                                                                              action, path, initial_slot,
                                                                                              slots))
                if super().is_path_free(self.k_shortest_paths[self.service.source, self.service.destination][path],
                                     initial_slot, slots):
                    super()._provision_path(self.k_shortest_paths[self.service.source, self.service.destination][path],
                                         initial_slot, slots)
                    self.service.accepted = True
                    self.previous_service_accepted = True
                    super()._add_release(self.service)

                else:

                    self.service.accepted = False
                    self.previous_service_accepted = False
            else:
                free_slots = np.sum(self.get_available_slots(self.k_shortest_paths[self.service.source, self.service.destination][0]))
                if free_slots < super().get_number_slots(self.k_shortest_paths[self.service.source, self.service.destination][0]):
                    self.episode_services_block_resource +=1
                else:
                    self.episode_services_block_frag += 1

                self.service.accepted = False
                self.previous_service_accepted = False
            self.services_processed += 1
            self.episode_services_processed += 1
            self.bit_rate_requested += self.service.bit_rate
            self.episode_bit_rate_requested += self.service.bit_rate
            self.topology.graph['services'].append(self.service)
            self.theta_list.append(1)
            if self.fragmentation_flag:
                self.theta_list.pop()
                self.theta_list.append(2)
                reward = reward + self.fragmentation_penalty_cycle
                self.fragmentation_flag = False
                self.defragmented_services = []
                self.episode_defragmentation_procedure += 1
                self.defragmentation_procedure += 1
            self._new_service = False

            episode_service_blocking_rate = (self.episode_services_processed - self.episode_services_accepted)/(self.episode_services_processed +1)
            reward = reward + np.log(episode_service_blocking_rate + 0.001)/np.log(0.001)
            # self._next_service()
            # the following line is added to have another type of rewards
            super()._next_service()

        # episode_bit_blocking_rate = (self.episode_bit_rate_requested - self.episode_bit_rate_provisioned) / (self.episode_bit_rate_requested + 1)
        episode_service_blocking_rate = (self.episode_services_processed - self.episode_services_accepted)/(self.episode_services_processed +1)
        # reward = reward + np.log(episode_service_blocking_rate + 0.001)/np.log(0.001)
        self.fragmentation_metric = 0

        self.number_cuts = 0
        self.fragmentation_metric += self._calculate_r_frag()
        self.number_cuts += self._calculate_total_cuts()
        self.rfrag_after_list.append(self.fragmentation_metric)
        self.num_cut_list_after.append(self.number_cuts)
        self.blocked_services.append(self.services_processed-self.services_accepted)
        self.num_moves_list.append(self.num_moves)
        self.defragmentation_procedure_list.append(self.defragmentation_procedure)

        #
        self.total_shanon_entropy = 0
        for n1, n2 in self.topology.edges():
            self.total_shanon_entropy += self._calculate_shanon_entropy(self.topology.graph['available_slots'][self.topology[n1][n2]['index'], :])
        self.total_shanon_entropy = self.total_shanon_entropy/len(self.topology.edges())

        self.shanon_entrophy_after_list.append(self.total_shanon_entropy)

        if len(self.shanon_entrophy_after_list) > 400:
            self.rfrag_after_list = self.rfrag_after_list[-400:]
            self.num_cut_list_after = self.num_cut_list_before[-400:]
            self.blocked_services = self.blocked_services[-400:]
            self.shanon_entrophy_after_list = self.shanon_entrophy_after_list[-400:]
            self.num_moves_list = self.num_moves_list[-400:]
            self.defragmentation_procedure_list = self.defragmentation_procedure_list[-400:]

        self.reward_cumulative += reward
        ## choosing the same options for the heuristic-nsfnet and reinforcment learning algorithm
        self.defragmentation_options = [] # Is this necessary?
        # self.defragmentation_options = [DefragmentationOption(0, -1, 0, 0, self.fragmentation_metric, 0, self.number_cuts, 0, 0, 0, self.total_shanon_entropy, 0)]
        available_candidates = 0
        self.fragmented_service_constraint = False
        for service in self.topology.graph['running_services']:
            if service not in self.defragmented_services :
                if len(self.defragmentation_options) > 2*self.number_options:
                    break
                # get available options
                self.get_available_options(service)
        self.fragmented_service_constraint = True
        chosen_services = []
        for i in range(len(self.defragmentation_options)):
            chosen_services.append(self.defragmentation_options[i].service)
        num_chosen_services = len(self.defragmentation_options)
        for service in self.topology.graph['running_services']:
            if service not in self.defragmented_services and service not in chosen_services:
                # if len(self.defragmentation_options) > 2*self.number_options:
                #     break
                # get available options
                self.get_available_options(service)

        temporory_options = self.defragmentation_options[num_chosen_services: len(self.defragmentation_options)]
        sorted_options = sorted(temporory_options, key = attrgetter('r_frag_diff'))
        sorted_options.reverse()
        reversed_sorted_options = sorted_options
        if reversed_sorted_options is not None and len(reversed_sorted_options) > 0:
            self.defragmentation_options[num_chosen_services: len(self.defragmentation_options)] = reversed_sorted_options

        self.rewards.append(reward)

        if len(self.rewards) > 400:
            self.rewards = self.rewards[-400:]
        self.sbr_list.append(episode_service_blocking_rate)
        # if len(self.cumulative_rewards) == 0:
        #     self.cumulative_rewards.append(reward)
        # else:
        #     self.cumulative_rewards.append(self.cumulative_rewards[len(self.cumulative_rewards)-1]*0.99 + reward)


        # this is for cumulative rewards, but we do not need it.
        # if self.episode_services_processed + self.episode_num_moves == self.episode_length:
        #     self.cumulative_rewards = np.zeros(self.episode_length)
        #     self.cumulative_rewards[self.episode_length-1] = self.rewards[self.episode_length-1]
        #     for i in range(1, self.episode_length):
        #         self.cumulative_rewards[self.episode_length - 1-i]= self.rewards[self.episode_length-1-i] \
        #                                                             + 0.99*self.cumulative_rewards[self.episode_length -i]
        #
        #     episodes_data = { 'rfrag_before': self.rfrag_before_list,
        #                      'rfrag_after': self.rfrag_after_list,  'num_cut_before':self.num_cut_list_before,
        #                      'num_cut_after':self.num_cut_list_after,
        #                       'Theta': self.theta_list }
        #     self.counting_episode +=1
        #     if self.counting_episode % 2 == 0:
        #         df = pd.DataFrame(episodes_data)
        #         df.to_csv(f"/cephyr/users/ehsanet/Vera/Desktop/Fragmentation_EON/examples/stable_baselines3/results-episode/episode_data-{self.counting_episode}.csv ")
        if self.episode_services_processed + self.episode_num_moves == self.episode_length:
            a = 1

        if self.episode_services_processed + self.episode_num_moves > self.episode_length:
            a = 2
        return self.observation(), reward, self.episode_services_processed + self.episode_num_moves == self.episode_length, self.get_info()

    def reset(self, only_counters=True):
        self.episode_bit_rate_requested = 0
        self.episode_bit_rate_provisioned = 0
        self.episode_services_processed = 0
        self.episode_services_accepted = 0
        self.episode_num_moves = 0
        self.episode_defragmentation_procedure = 0
        self.episode_services_block_frag = 0
        self.episode_services_block_resource = 0
        self.reward_cumulative = 0
        self.number_cuts = 0
        self.fragmentation_metric = 0
        self.total_shanon_entropy = 0
        # self.rewards = []
        self.cumulative_rewards = []
        self.theta_list = []
        self.rfrag_before_list=[]
        # self.rfrag_after_list = []
        self.r_frag_difference_list=[]
        self.num_cut_list_before=[]
        # self.num_cut_list_after=[]
        self.shanpn_entrophy_before_list = []
        # self.shanon_entrophy_after_list= []
        self.frag_size_before_list = []
        self.frag_size_after_list = []
        self.sbr_list = []


        if only_counters:
            return self.observation()

        return super().reset(only_counters = only_counters)


    def render(self, mode='human'):
        return

    def _move_path(self, new_initial_slot):

        # firstly, release the path, then provision the new path
        super()._release_path(self.service_to_defrag)
        new_path = self.service_to_defrag.route # the new path is the same path, we do not change paths.
        self.service_to_defrag.initial_slot = new_initial_slot
        for i in range(len(new_path.node_list) - 1):
            self.topology.graph['available_slots'][self.topology[new_path.node_list[i]][new_path.node_list[i + 1]]['index'],
            new_initial_slot:new_initial_slot + self.service_to_defrag.number_slots] = 0
            self.spectrum_slots_allocation[self.topology[new_path.node_list[i]][new_path.node_list[i + 1]]['index'],
            new_initial_slot:new_initial_slot + self.service_to_defrag.number_slots] = self.service_to_defrag.service_id
            self.topology[new_path.node_list[i]][new_path.node_list[i + 1]]['services'].append(self.service_to_defrag)
            self.topology[new_path.node_list[i]][new_path.node_list[i + 1]]['running_services'].append(self.service_to_defrag)
            self._update_link_stats(new_path.node_list[i], new_path.node_list[i + 1])

        self.topology.graph['running_services'].append(self.service_to_defrag)
        self._update_network_stats()


    # def _next_service(self):
    #     super()._next_service()
    #     ## choosing the same options for the heuristic and reinforcment learning algorithm
    #     self.defragmentation_options = [] # Is this necessary?
    #     # self.defragmentation_options.append((0, -1))  # -1 means no defragmentation. The first option is always do nothing.
    #     for service in self.topology.graph['running_services']:
    #         # the following line are commented since we want to choose the options differently
    #         #if super().is_path_free(service.route,
    #         #                    max(0,service.initial_slot-1),
    #         #                    1):
    #         #   self.defragmentation_options.append((service.service_id,1))
    #         # elif super().is_path_free(service.route,
    #         #                      service.initial_slot + service.number_slots,
    #         #                      1):
    #         #     self.defragmentation_options.append((service.service_id, 2))
    #         # get available options
    #         self.get_available_options(service)

    def _update_network_stats(self):
        super()._update_network_stats()



    def _calculate_r_frag(self):
        r_spectral = 0
        r_spatial = 0
        for n1, n2 in self.topology.edges():
            initial_indices, values, lengths = \
                RMSAEnv.rle(self.topology.graph['available_slots'][self.topology[n1][n2]['index'], :])
            unused_blocks = [i for i, x in enumerate(values) if x == 1]
            r_spectral += np.sqrt(np.sum(lengths[unused_blocks]**2))/(np.sum(lengths[unused_blocks])+1)
        for count in range(self.num_spectrum_resources):
            initial_indices, values, lengths = \
                RMSAEnv.rle(self.topology.graph['available_slots'][:, count])
            unused_blocks = [i for i, x in enumerate(values) if x == 1]
            r_spatial += np.sqrt(np.sum(lengths[unused_blocks] ** 2)) /(np.sum(lengths[unused_blocks])+1) # added to one to avoid infinity? #TODO : this should be checked. (add +1 is correct)
        return r_spatial/self.num_spectrum_resources + r_spectral/len(self.topology.edges())


    def _calculate_total_cuts(self):
        number_cut = 0
        for service in self.topology.graph['running_services']:
            number_cut_service = 0
            for i in range(len(service.route.node_list) - 1):

                available_slots_link = self.topology.graph['available_slots'][
                                       self.topology[service.route.node_list[i]][service.route.node_list[i + 1]]['index'],
                                       :]
                if available_slots_link[
                    service.initial_slot - 1] == 1:
                    number_cut_service += 1

                # if available_slots_link[
                #     service.initial_slot + service.number_slots] == 1:
                #     number_cut += 1
            number_cut += number_cut_service/len(service.route.node_list)
        return number_cut/(len(self.topology.graph['running_services']) +0.001)


    def _calculate_r_spectral(self, node_list):
        r_spectral = 0
        for i in range(len(node_list) - 1):
            initial_indices, values, lengths = \
                RMSAEnv.rle(self.topology.graph['available_slots'][self.topology[node_list[i]][node_list[i + 1]]['index'], :])
            unused_blocks = [i for i, x in enumerate(values) if x == 1]
            r_spectral += np.sqrt(np.sum(lengths[unused_blocks]**2))/(np.sum(lengths[unused_blocks])+1)
        return r_spectral/(len(node_list)+1)

    def _calculate_r_spatial(self, first_slot, last_slot):
        r_spatial = 0
        for count in range(first_slot,last_slot):
            initial_indices, values, lengths = \
                RMSAEnv.rle(self.topology.graph['available_slots'][:, count])
            unused_blocks = [i for i, x in enumerate(values) if x == 1]
            r_spatial += np.sqrt(np.sum(lengths[unused_blocks] ** 2)) /(np.sum(lengths[unused_blocks])+1) # added to one to avoid infinity?

        return r_spatial/(last_slot - first_slot +1)

    def _calculate_shanon_entropy(self, slots):
        initial_indices, values, lengths = \
            RMSAEnv.rle(slots)
        unused_blocks = [i for i, x in enumerate(values) if x == 1]
        shanon_entropy = - np.sum((lengths[unused_blocks]/self.num_spectrum_resources)*np.log(lengths[unused_blocks]/self.num_spectrum_resources))
        # a = lengths[unused_blocks]/self.num_spectrum_resources
        # b= np.log(lengths[unused_blocks]/self.num_spectrum_resources)
        return shanon_entropy




    def _calculate_r_frag_modified(self, node_list, initial_slot, num_slots, final_slot):
        r_spectral_before = 0
        r_spatial_before = 0
        r_spectral_after = 0
        r_spatial_after = 0
        concatenated = chain(range(initial_slot, initial_slot+num_slots), range(final_slot, final_slot + num_slots))
        # for n1, n2 in self.topology.edges():
        #     initial_indices, values, lengths = \
        #         RMSAEnv.rle(self.topology.graph['available_slots'][self.topology[n1][n2]['index'], :])
        #     unused_blocks = [i for i, x in enumerate(values) if x == 1]
        #     r_spectral += np.sqrt(np.sum(lengths[unused_blocks]**2))/(np.sum(lengths[unused_blocks])+1)
        # for count in range(self.num_spectrum_resources):
        #     initial_indices, values, lengths = \
        #         RMSAEnv.rle(self.topology.graph['available_slots'][:, count])
        #     unused_blocks = [i for i, x in enumerate(values) if x == 1]
        #     r_spatial += np.sqrt(np.sum(lengths[unused_blocks] ** 2)) /(np.sum(lengths[unused_blocks])+1) # added to one to avoid infinity? #TODO : this should be checked. (add +1 is correct)

        for i in range(len(node_list) - 1):
            initial_indices, values, lengths = \
                RMSAEnv.rle(self.topology.graph['available_slots'][self.topology[node_list[i]][node_list[i + 1]]['index'], :])
            unused_blocks = [i for i, x in enumerate(values) if x == 1]
            r_spectral_before += np.sqrt(np.sum(lengths[unused_blocks]**2))/(np.sum(lengths[unused_blocks])+1)


        for count in concatenated:
            initial_indices, values, lengths = \
                RMSAEnv.rle(self.topology.graph['available_slots'][:, count])
            unused_blocks = [i for i, x in enumerate(values) if x == 1]
            r_spatial_before += np.sqrt(np.sum(lengths[unused_blocks] ** 2)) /(np.sum(lengths[unused_blocks])+1) # added to one to avoid infinity?

        temporary_slots = self.topology.graph['available_slots']

        for i in range(len(node_list) - 1):
            temporary_slots[
            self.topology[node_list[i]][node_list[i + 1]]['index'],
            final_slot:final_slot + num_slots] = 0
            temporary_slots[
            self.topology[node_list[i]][node_list[i + 1]]['index'],
            initial_slot:initial_slot + num_slots] = 1

        for i in range(len(node_list) - 1):
            initial_indices, values, lengths = \
                RMSAEnv.rle(
                    temporary_slots[self.topology[node_list[i]][node_list[i + 1]]['index'], :])
            unused_blocks = [i for i, x in enumerate(values) if x == 1]
            r_spectral_after += np.sqrt(np.sum(lengths[unused_blocks] ** 2)) / (np.sum(lengths[unused_blocks]) + 1)

        concatenated = chain(range(initial_slot, initial_slot + num_slots),
                                 range(final_slot, final_slot + num_slots))
        for count in concatenated:
            initial_indices, values, lengths = \
                RMSAEnv.rle(temporary_slots[:, count])
            unused_blocks = [i for i, x in enumerate(values) if x == 1]
            r_spatial_after += np.sqrt(np.sum(lengths[unused_blocks] ** 2)) / (
                        np.sum(lengths[unused_blocks]) + 1)  # added to one to avoid infinity?



        return r_spatial_before + r_spectral_before, r_spatial_after+ + r_spectral_after






    def _update_link_stats(self, node1: str, node2: str):
        super()._update_link_stats(node1, node2)

    # Developing this function for finding all moving options in the blocks.

    def get_available_options(self, service):

        # get available slots across the whole path
        # 1 if slot is available across all the links
        # zero if not
        available_slots = self.get_available_slots(service.route)

        # getting the number of slots necessary for this service across this path
        slots = service.number_slots
        if (available_slots[service.initial_slot - 1] ==1 and\
           service.initial_slot + slots <= 319 and available_slots[service.initial_slot + slots] == 1 and service.initial_slot != 0) \
                or self.fragmented_service_constraint:

                    # This is other way to solve this problem more efficiently.
            temporarily_available_slots = self.get_available_slots(service.route)

            temporarily_available_slots[service.initial_slot:service.initial_slot+slots] = 1
            # getting the blocks
            initial_indices, values, lengths = RMSAEnv.rle(temporarily_available_slots)
            # selecting the indices where the block is available, i.e., equals to one
            available_indices = np.where(values == 1)
            # selecting the indices where the block has sufficient slots
            sufficient_indices = np.where(lengths >= slots)
            # getting the intersection, i.e., indices where the slots are available in sufficient quantity
        # and using only the J first indices
            final_indices = np.intersect1d(available_indices, sufficient_indices)
            if self.only_FF:
                eligible_options = 1
            else:
                eligible_options = final_indices.size

            for counter in range(eligible_options):
                    #Developing an if for checking free blocks
                if initial_indices[final_indices[counter]] != service.initial_slot:
                    #developing one new observation.
                    # left_side = 0
                    # right_side = 0
                    # right_counter = service.initial_slot + slots
                    # left_counter = service.initial_slot -1
                    #
                    #
                    # while left_counter >=0 and  available_slots[left_counter] ==1 :
                    #     left_side +=1
                    #     left_counter -=1
                    #
                    # while right_counter <=319 and available_slots[right_counter] ==1 :
                    #     right_side +=1
                    #     right_counter +=1
                    #
                    # if right_counter ==320: ## this is necessary since we are doing First-FF, and we need this for right handside . In
                    #     #other words, if we are neighbor to end of the spectrum slot, we are not fragmented.
                    #     right_side = 0

                    number_cut_before = 0
                    frag_size_before = 0
                    number_cut_after = 0
                    frag_size_after = 0
                    # if self.calcualte_highest_cut:
                    shanon_entropy_before = 0
                    shanon_entropy_after = 0
                    for i in range(len(service.route.node_list) - 1):

                        available_slots_link = self.topology.graph['available_slots'][
                        self.topology[service.route.node_list[i]][service.route.node_list[i + 1]]['index'],
                        :]
                        shanon_entropy_before += self._calculate_shanon_entropy(available_slots_link)
                        if available_slots_link[service.initial_slot -1] == 1:
                            number_cut_before += 1
                            left_counter = service.initial_slot - 1
                            left_frag = 0
                            while left_counter >=0 and  available_slots_link[left_counter] ==1 :
                                left_frag += 1
                                left_counter -= 1
                            frag_size_before += left_frag
                    shanon_entropy_before /= len(service.route.node_list)
                    number_cut_before /= len(service.route.node_list)
                    frag_size_before/= len(service.route.node_list)

                        # if  available_slots_link[
                        # service.initial_slot + slots] == 1:
                        #     number_cut += 1

                    # if self.calcualte_highest_metric_difference:
                    r_frag_before = self._calculate_r_spectral(service.route.node_list) + self._calculate_r_spatial(service.initial_slot, service.initial_slot +slots)
                    + self._calculate_r_spatial(initial_indices[final_indices[counter]], initial_indices[final_indices[counter]] + slots)

                    for i in range(len(service.route.node_list) - 1):
                        self.topology.graph['available_slots'][
                        self.topology[service.route.node_list[i]][service.route.node_list[i + 1]]['index'],
                        initial_indices[final_indices[counter]]:initial_indices[final_indices[counter]] + slots] = 0
                        self.topology.graph['available_slots'][
                        self.topology[service.route.node_list[i]][service.route.node_list[i + 1]]['index'],
                        service.initial_slot:service.initial_slot + slots] = 1


                    for i in range(len(service.route.node_list) - 1):

                        available_slots_link = self.topology.graph['available_slots'][
                        self.topology[service.route.node_list[i]][service.route.node_list[i + 1]]['index'],
                        :]

                        if  available_slots_link[
                        service.initial_slot -1] == 1:
                            number_cut_after += 1
                            left_counter = service.initial_slot - 1
                            left_frag = 0
                            while left_counter >=0 and available_slots_link[left_counter] == 1 :
                                left_frag += 1
                                left_counter -= 1
                            frag_size_after += left_frag
                        shanon_entropy_after += self._calculate_shanon_entropy(available_slots_link)

                    shanon_entropy_after /= len(service.route.node_list)
                    number_cut_after /= len(service.route.node_list)
                    frag_size_after /= len(service.route.node_list)



                    # r_frag_after = self._calculate_r_frag()

                    r_frag_after = self._calculate_r_spectral(service.route.node_list) + self._calculate_r_spatial(service.initial_slot, service.initial_slot +slots)
                    + self._calculate_r_spatial(initial_indices[final_indices[counter]], initial_indices[final_indices[counter]] + slots)
                    #
                    # if self.calcualte_highest_metric_difference:
                    #     r_frag_before, r_frag_after = self._calculate_r_frag_modified\
                    #         (service.route.node_list, service.initial_slot, slots, initial_indices[final_indices[counter]])



                    for i in range(len(service.route.node_list) - 1):
                        self.topology.graph['available_slots'][
                        self.topology[service.route.node_list[i]][service.route.node_list[i + 1]]['index'],
                        initial_indices[final_indices[counter]]:initial_indices[final_indices[counter]] + slots] = 1
                        self.topology.graph['available_slots'][
                        self.topology[service.route.node_list[i]][service.route.node_list[i + 1]]['index'],
                        service.initial_slot:service.initial_slot + slots] = 0


                    # for i in range(len(service.route.node_list) - 1):
                    #     self.topology.graph['available_slots'][
                    #     self.topology[service.route.node_list[i]][service.route.node_list[i + 1]]['index'],
                    #     initial_indices[final_indices[counter]] :initial_indices[final_indices[counter]] + slots] = 1
                    #     self.topology.graph['available_slots'][
                    #     self.topology[service.route.node_list[i]][service.route.node_list[i + 1]]['index'],
                    #     initial_indices[final_indices[counter]] +  lengths[final_indices[counter]] - slots :initial_indices[final_indices[counter]] +  lengths[final_indices[counter]]] = 0
                    #
                    # r_frag_after_right = self._calculate_r_spectral(service.route.node_list) + self._calculate_r_spatial(service.initial_slot, service.initial_slot +slots)
                    # + self._calculate_r_spatial(initial_indices[final_indices[counter]] + lengths[final_indices[counter]] - slots, initial_indices[final_indices[counter]]+ lengths[final_indices[counter]])
                    #
                    # for i in range(len(service.route.node_list) - 1):
                    #     self.topology.graph['available_slots'][
                    #     self.topology[service.route.node_list[i]][service.route.node_list[i + 1]]['index'],
                    #     initial_indices[final_indices[counter]] +  lengths[final_indices[counter]] - slots :initial_indices[final_indices[counter]] +  lengths[final_indices[counter]]] = 1
                    #     self.topology.graph['available_slots'][
                    #     self.topology[service.route.node_list[i]][service.route.node_list[i + 1]]['index'],

                    self.defragmentation_options.append(DefragmentationOption(service,
                                                                            initial_indices[final_indices[counter]],
                                                                            lengths[final_indices[counter]],
                                                                            True, r_frag_before, r_frag_after, r_frag_after-r_frag_before,
                                                                              number_cut_before,number_cut_after,  frag_size_before, frag_size_after, shanon_entropy_before, shanon_entropy_after))

                            # Other option is end of free block
                    # if initial_indices[final_indices[counter]] + lengths[final_indices[counter]] - slots != service.initial_slot:
                    #             # self.defragmentation_options.append((service, initial_indices[final_indices[counter]]
                    #             #                                     + lengths[final_indices[counter]] - slots))
                    #
                    #             self.defragmentation_options.append(DefragmentationOption(service,
                    #                                                                     initial_indices[final_indices[counter]]
                    #                                                                 + lengths[final_indices[counter]] - slots,
                    #                                                                     lengths[final_indices[counter]],
                    #                                                                     False, left_side, right_side))
                    #
                    # self.defragmentation_options.append(DefragmentationOption(service,
                    #                                                         initial_indices[final_indices[counter]]
                    #                                                     + lengths[final_indices[counter]] - slots,
                    #                                                         lengths[final_indices[counter]],
                    #                                                         False, left_side, right_side, r_frag_before, r_frag_after_right, number_cut, frag_size))

    def get_info(self):
        info = {
            'service_blocking_rate': (self.services_processed - self.services_accepted) / (self.services_processed+1),
            'episode_service_blocking_rate': (self.episode_services_processed - self.episode_services_accepted) / (self.episode_services_processed +1),
            'service_blocked_eopisode' : (self.episode_services_processed - self.episode_services_accepted),
            'bit_rate_blocking_rate': (self.bit_rate_requested - self.bit_rate_provisioned) /( self.bit_rate_requested +1),
            'reward': self.reward_cumulative,
            'number_movements_episode': self.episode_num_moves,
            'number_defragmentation_procedure_episode': self.episode_defragmentation_procedure,
            'number_movements': self.num_moves,
            'number_defragmentation_procedure': self.defragmentation_procedure,
            'number_arrivals': self.services_processed,
            'number_options' : len(self.defragmentation_options),
            'existing_options' : self.number_existing_options,
            'episode_service_blocking_rate_fragmentation': self.episode_services_block_frag/ (
                        self.episode_services_processed + 1),
            'episode_service_blocking_rate_lack': self.episode_services_block_resource / (
                    self.episode_services_processed + 1),
            'episode_cuts': self.number_cuts,
            'episode_frag_metric': self.fragmentation_metric,
            'episode_shanon_entropy' : self.total_shanon_entropy

        }
        return info

    def calculate_free_slots_neighbor(self):
        pass

class OldestFirst:
    def __init__(self, defragmentation_period: int = 10,  number_connection: int = 10) -> None:
        self.defragmentation_period = defragmentation_period
        self.number_connection = number_connection



    def choose_oldest_first(self, env: DefragmentationEnv):
        env.defragmentation_options_available = [DefragmentationOption(0, -1, 0, 0, 0, 0,0,0)]

        index_option = 0

        if env.defragmentation_period_count != self.defragmentation_period:
            env.defragmentation_period_count += 1
            return DefragmentationOption(0, -1, 0, 0, 0, 0,0,0)
        else:

            while len(env.defragmentation_options_available) - 1 < env.number_options and len(
                    env.defragmentation_options) > 0:
                service_to_defrag = env.defragmentation_options[index_option].service
                if service_to_defrag not in env.defragmented_services and \
                        env.defragmentation_options[index_option].starting_slot < service_to_defrag.initial_slot:
                    env.defragmentation_options_available.append(env.defragmentation_options[index_option])
                index_option += 1
                if index_option >= len(env.defragmentation_options):
                    break
            env.env.number_existing_options = len(env.defragmentation_options_available)
            env.defragmentation_movement_period += 1
            if env.defragmentation_movement_period == self.number_connection:
                env.defragmentation_movement_period = 0
                env.defragmentation_period_count = 0
            if len(env.defragmentation_options_available) > 1:
                return env.defragmentation_options_available[1]
            else:
                env.defragmentation_movement_period = 0
                env.defragmentation_period_count = 0
                return DefragmentationOption(0, -1, 0, 0, 0, 0,0,0)

        # if env.env.services_processed % self.defragmentation_period != 0:
        #     return DefragmentationOption(0, -1, 0, 0)
        # else:
        #     if len(env.defragmentation_options_available) > 1:
        #         return env.defragmentation_options_available[1]
        #     else:
        #         return DefragmentationOption(0, -1, 0, 0)




class HighMetricFirst:
    def __init__(self, defragmentation_period: int = 10,  number_connection: int = 10) -> None:
        self.defragmentation_period = defragmentation_period
        self.number_connection = number_connection



    def choose_highest_difference_first(self, env: DefragmentationEnv):


        ### this code is wrong

        env.env.calcualte_highest_metric_difference = True
        env.defragmentation_options_available = []
        if len(env.defragmentation_options) > 0:
            env.defragmentation_options_available.append(env.defragmentation_options[0])
            del env.defragmentation_options[0]
        else:
            env.defragmentation_options_available.append(DefragmentationOption(0, -1, 0, 0, 0, 0,0,0))


        if env.defragmentation_period_count != self.defragmentation_period:
            env.defragmentation_period_count += 1
            # return DefragmentationOption(0, -1, 0, 0, 0, 0,0,0)
            return env.defragmentation_options_available[0]
        else:
            # env.defragmentation_options_available = [DefragmentationOption(0, -1, 0, 0, 0, 0, 0, 0), ]
            index_option = 0
            best_difference = -200
            while len(env.defragmentation_options) > 0:
                service_to_defrag = env.defragmentation_options[index_option].service
                if service_to_defrag not in env.defragmented_services and \
                        env.defragmentation_options[index_option].starting_slot < service_to_defrag.initial_slot:
                    if (env.defragmentation_options[index_option].r_frag_after - env.defragmentation_options[
                        index_option].r_frag_before) > \
                            best_difference:
                        best_difference = env.defragmentation_options[index_option].r_frag_after - \
                                          env.defragmentation_options[index_option].r_frag_before
                        env.defragmentation_options_available.append(env.defragmentation_options[index_option])
                        env.defragmentation_options_available[1] = (env.defragmentation_options[index_option])
                index_option += 1
                if index_option >= len(env.defragmentation_options):
                    break
            env.env.number_existing_options = len(env.defragmentation_options_available)
            env.defragmentation_movement_period += 1
            if env.defragmentation_movement_period == self.number_connection:
                env.defragmentation_movement_period = 0
                env.defragmentation_period_count = 0
            if len(env.defragmentation_options_available) > 1:
                return env.defragmentation_options_available[1]
            else:
                env.defragmentation_movement_period = 0
                env.defragmentation_period_count = 0
                return DefragmentationOption(0, -1, 0, 0, 0, 0,0,0)



class HighCutMetricFirst:
    def __init__(self, defragmentation_period: int = 10,  number_connection: int = 10) -> None:
        self.defragmentation_period = defragmentation_period
        self.number_connection = number_connection



    def choose_highest_cut_difference_first(self, env: DefragmentationEnv):

        env.env.calcualte_highest_cut = True

        if env.defragmentation_period_count != self.defragmentation_period:
            env.defragmentation_period_count += 1
            return DefragmentationOption(0, -1, 0, 0, 0, 0,0,0)
        else:
            env.defragmentation_options_available = [DefragmentationOption(0, -1, 0, 0, 0, 0, 0, 0), ]
            index_option = 0
            best_cut = -200
            best_frag = -100
            best_difference = -200
            while len(env.defragmentation_options) > 0:
                service_to_defrag = env.defragmentation_options[index_option].service
                if service_to_defrag not in env.defragmented_services :
                        # and \
                        # env.defragmentation_options[index_option].starting_slot < service_to_defrag.initial_slot:
                    if (env.defragmentation_options[index_option].number_cut_before ) >  best_cut:
                        best_cut = env.defragmentation_options[index_option].number_cut_before
                        best_frag = env.defragmentation_options[index_option].frag_size_before
                        env.defragmentation_options_available.append(env.defragmentation_options[index_option])
                        env.defragmentation_options_available[1] = (env.defragmentation_options[index_option])
                        best_difference = env.defragmentation_options[index_option].r_frag_after - \
                                          env.defragmentation_options[index_option].r_frag_before
                    elif (env.defragmentation_options[index_option].number_cut_before) == best_cut:
                        if (env.defragmentation_options[index_option].frag_size_before ) >  best_frag:
                            best_frag = env.defragmentation_options[index_option].frag_size_before
                            env.defragmentation_options_available.append(env.defragmentation_options[index_option])
                            env.defragmentation_options_available[1] = (env.defragmentation_options[index_option])
                            best_difference = env.defragmentation_options[index_option].r_frag_after - \
                                              env.defragmentation_options[index_option].r_frag_before
                        elif (env.defragmentation_options[index_option].frag_size_before ) ==  best_frag:
                            if (env.defragmentation_options[index_option].r_frag_after - env.defragmentation_options[
                        index_option].r_frag_before) > \
                            best_difference:
                                env.defragmentation_options_available.append(env.defragmentation_options[index_option])
                                env.defragmentation_options_available[1] = (env.defragmentation_options[index_option])

                index_option += 1
                if index_option >= len(env.defragmentation_options):
                    break
            env.env.number_existing_options = len(env.defragmentation_options_available)
            env.defragmentation_movement_period += 1
            if env.defragmentation_movement_period == self.number_connection:
                env.defragmentation_movement_period = 0
                env.defragmentation_period_count = 0
            if len(env.defragmentation_options_available) > 1:
                return env.defragmentation_options_available[1]
            else:
                env.defragmentation_movement_period = 0
                env.defragmentation_period_count = 0
                return DefragmentationOption(0, -1, 0, 0, 0, 0,0,0)





def choose_randomly(env: DefragmentationEnv):
    env.defragmentation_options_available = [DefragmentationOption(0,-1,0,0,0,0,0,0)]
    index_option = 0
    while len(env.defragmentation_options_available) - 1 < env.number_options and len(env.defragmentation_options) > 0:
        service_to_defrag = env.defragmentation_options[index_option].service
        if service_to_defrag not in env.defragmented_services:
            env.defragmentation_options_available.append(env.defragmentation_options[index_option])
        index_option += 1
        if index_option >= len(env.defragmentation_options):
            break
    # if len(env.defragmentation_options) == 0:
    #     return [0, -1]
    # else:
    return env.defragmentation_options_available[random.randint(0,len(env.defragmentation_options_available)-1)]


def assigning_path_without_defragmentation(env: DefragmentationEnv):
    return DefragmentationOption(0,-1,0,0,0,0,0,0)
