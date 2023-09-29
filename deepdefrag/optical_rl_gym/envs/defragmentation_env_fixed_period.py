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

from optical_rl_gym.utils import plot_spectrum_assignment, plot_spectrum_assignment_and_waste, DefragmentationOption
from optical_rl_gym.envs.rmsa_env import shortest_path_first_fit, shortest_available_path_first_fit, \
    least_loaded_path_first_fit, SimpleMatrixObservation, Fragmentation_alignment_aware_RMSA

from .rmsa_env  import RMSAEnv



class DefragmentationEnv(RMSAEnv):
    metadata = {
        'metrics': ['service_blocking_rate', 'episode_service_blocking_rate',
                    'bit_rate_blocking_rate', 'episode_bit_rate_blocking_rate']
    }

    def __init__(self, topology=None,
                 episode_length=1000,
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
                 number_options = 5,
                 penalty_movement= -3,
                 penalty_cycle= -0.25
                 ):
        super().__init__(topology,
                         episode_length=episode_length,
                         load=load,
                         mean_service_holding_time=mean_service_holding_time,
                         num_spectrum_resources=num_spectrum_resources,
                         node_request_probabilities=node_request_probabilities,
                         seed=seed, allow_rejection=allow_rejection,
                         k_paths=k_paths,
                         incremental_traffic_percentage=incremental_traffic_percentage, reset=False)
        assert 'modulations' in self.topology.graph
        self.number_options = number_options
        # specific attributes for elastic optical networks
        self.defragmentation_period = defragmentation_period
        self.rmsa_function = rmsa_function
        self.service_to_defrag = None
        self.defragmentation_options = []
        # A counter to calculate the number of movements, and number of defragmentation procedure
        self.episode_num_moves = 0
        self.episode_number_moves = 0
        self.num_moves = 0
        self.number_moves = 0
        self.episode_defragmentation_procedure = 0
        self.defragmentation_procedure = 0
        # This flag is defined to figure out the fragmentation movement within steps.
        # It is used in the reward definition
        self.fragmentation_flag = False
        self.defragmented_services = []
        # a penalty for doing defragmentation, this should be tuned.
        self.fragmentation_penalty_cycle = penalty_cycle
        self.fragmentation_penalty_movement = penalty_movement
        # for implementing oldest first strategy, we need to define two auxiliary variable
        self.defragmentation_period_count = 0
        self.defragmentation_movement_period = 0
        self.number_existing_options = 0

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
        # doing defragmentation






        if new_initial_slot != -1:
            self.episode_num_moves = self.episode_num_moves + 1
            self.num_moves = self.num_moves + 1
            self.service_to_defrag = service_to_defrag
            self.defragmented_services.append(service_to_defrag)
            r_frag_before = self._calculate_r_frag()
            self._move_path(new_initial_slot)
            # adding reward at the beginning of the cycle
            # if self.fragmentation_flag:
            #     reward = reward + self.fragmentation_penalty_movement
            # else:
            #     reward = reward + self.fragmentation_penalty_cycle + self.fragmentation_penalty_movement

            # reward = reward + self.fragmentation_penalty_movement
            self.fragmentation_flag = True
            r_frag_after = self._calculate_r_frag()
            r_frag_difference = r_frag_after - r_frag_before
            reward = reward + 10*r_frag_difference
            # if r_frag_after > r_frag_before:
            #     reward = +1
            # elif r_frag_before > r_frag_after:
            #     reward = -1





        if  self.number_moves%10 == 0:
            if self.number_moves ==0:
                first_allocations = 400
            else:
                first_allocations = 10
            for i in range(first_allocations):
                path, initial_slot = self.rmsa_function(self)[0], self.rmsa_function(self)[1]
                if path < self.k_paths and initial_slot < self.num_spectrum_resources:  # action is for assigning a path
                    slots = super().get_number_slots(
                        self.k_shortest_paths[self.service.source, self.service.destination][path])
                    self.logger.debug(
                        '{} processing action {} path {} and initial slot {} for {} slots'.format(
                            self.service.service_id,
                            action, path, initial_slot,
                            slots))
                    if super().is_path_free(self.k_shortest_paths[self.service.source, self.service.destination][path],
                                            initial_slot, slots):
                        super()._provision_path(
                            self.k_shortest_paths[self.service.source, self.service.destination][path],
                            initial_slot, slots)
                        self.service.accepted = True
                        super()._add_release(self.service)

                    else:
                        self.service.accepted = False
                else:
                    self.service.accepted = False
                self.services_processed += 1
                self.episode_services_processed += 1
                self.bit_rate_requested += self.service.bit_rate
                self.episode_bit_rate_requested += self.service.bit_rate
                self.topology.graph['services'].append(self.service)
                if self.fragmentation_flag:
                    # reward = reward + self.fragmentation_penalty_cycle
                    self.fragmentation_flag = False
                    self.defragmented_services = []
                    self.episode_defragmentation_procedure += 1
                    self.defragmentation_procedure += 1
                self._new_service = False
                # self._next_service()
                # the following line is added to have another type of rewards
                super()._next_service()

        # else:
        #     path, initial_slot = self.rmsa_function(self)[0], self.rmsa_function(self)[1]
        #     if path < self.k_paths and initial_slot < self.num_spectrum_resources:  # action is for assigning a path
        #         slots = super().get_number_slots(self.k_shortest_paths[self.service.source, self.service.destination][path])
        #         self.logger.debug(
        #             '{} processing action {} path {} and initial slot {} for {} slots'.format(self.service.service_id,
        #                                                                                       action, path, initial_slot,
        #                                                                                       slots))
        #         if super().is_path_free(self.k_shortest_paths[self.service.source, self.service.destination][path],
        #                              initial_slot, slots):
        #             super()._provision_path(self.k_shortest_paths[self.service.source, self.service.destination][path],
        #                                  initial_slot, slots)
        #             self.service.accepted = True
        #             super()._add_release(self.service)
        #
        #         else:
        #             self.service.accepted = False
        #     else:
        #         self.service.accepted = False
        #     self.services_processed += 1
        #     self.episode_services_processed += 1
        #     self.bit_rate_requested += self.service.bit_rate
        #     self.episode_bit_rate_requested += self.service.bit_rate
        #     self.topology.graph['services'].append(self.service)
        #     if self.fragmentation_flag:
        #         reward = reward + self.fragmentation_penalty_cycle
        #         self.fragmentation_flag = False
        #         self.defragmented_services = []
        #         self.episode_defragmentation_procedure += 1
        #         self.defragmentation_procedure += 1
        #     self._new_service = False
        #     # self._next_service()
        #     # the following line is added to have another type of rewards
        #     super()._next_service()

        # episode_bit_blocking_rate = (self.episode_bit_rate_requested - self.episode_bit_rate_provisioned) / (self.episode_bit_rate_requested + 1)
        # reward = reward + 1 - episode_bit_blocking_rate
        # self.reward_cumulative += reward
        ## choosing the same options for the heuristic-nsfnet and reinforcment learning algorithm
        self.episode_number_moves = self.episode_number_moves + 1
        self.number_moves += 1
        self.defragmentation_options = [] # Is this necessary?
        for service in self.topology.graph['running_services']:
            # get available options
            self.get_available_options(service)
        return self.observation(), reward, self.episode_number_moves == self.episode_length, self.get_info()

    def reset(self, only_counters=True):
        self.episode_bit_rate_requested = 0
        self.episode_bit_rate_provisioned = 0
        self.episode_services_processed = 0
        self.episode_services_accepted = 0
        self.episode_num_moves = 0
        self.episode_defragmentation_procedure = 0
        self.episode_number_moves = 0
        self.reward_cumulative = 0
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
        return r_spatial + r_spectral

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
        # if available_slots[service.initial_slot - 1] ==1 and\
        #    service.initial_slot + slots <= 319 and available_slots[service.initial_slot + slots] == 1 and service.initial_slot != 0:

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
        for counter in range(final_indices.size):
            if counter ==0:
                #Developing an if for checking free blocks
                if initial_indices[final_indices[counter]] != service.initial_slot:
                    # self.defragmentation_options.append((service, initial_indices[final_indices[counter]]))
                    #developing one new observation.
                    left_side = 0
                    right_side = 0
                    right_counter = service.initial_slot + slots
                    left_counter = service.initial_slot -1
                    while left_counter >=0 and  available_slots[left_counter] ==1 :
                        left_side +=1
                        left_counter -=1

                    while right_counter <=319 and available_slots[right_counter] ==1 :
                        right_side +=1
                        right_counter +=1

                    if right_counter ==320: ## this is necessary since we are doing First-FF, and we need this for right handside . In
                        #other words, if we are neighbor to end of the spectrum slot, we are not fragmented.
                        right_side = 0

                    self.defragmentation_options.append(DefragmentationOption(service,
                                                                            initial_indices[final_indices[counter]],
                                                                            lengths[final_indices[counter]],
                                                                        True, left_side, right_side))

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
            'existing_options' : self.number_existing_options

        }
        return info

    def calculate_free_slots_neighbor(self):
        pass

class OldestFirst:
    def __init__(self, defragmentation_period: int = 10,  number_connection: int = 10) -> None:
        self.defragmentation_period = defragmentation_period
        self.number_connection = number_connection



    def choose_oldest_first(self, env: DefragmentationEnv):
        env.defragmentation_options_available = [DefragmentationOption(0, -1, 0, 0, 0, 0)]
        index_option = 0
        while len(env.defragmentation_options_available) - 1 < env.number_options and len(
                env.defragmentation_options) > 0:
            service_to_defrag = env.defragmentation_options[index_option].service
            if service_to_defrag not in env.defragmented_services and \
                env.defragmentation_options[index_option].starting_slot < service_to_defrag.initial_slot :
                env.defragmentation_options_available.append(env.defragmentation_options[index_option])
            index_option += 1
            if index_option >= len(env.defragmentation_options):
                break
        env.env.number_existing_options = len(env.defragmentation_options_available)
        if env.defragmentation_period_count != self.defragmentation_period:
            env.defragmentation_period_count += 1
            return DefragmentationOption(0, -1, 0, 0, 0, 0)
        else:
            env.defragmentation_movement_period += 1
            if env.defragmentation_movement_period == self.number_connection:
                env.defragmentation_movement_period = 0
                env.defragmentation_period_count = 0
            if len(env.defragmentation_options_available) > 1:
                return env.defragmentation_options_available[1]
            else:
                env.defragmentation_movement_period = 0
                env.defragmentation_period_count = 0
                return DefragmentationOption(0, -1, 0, 0, 0, 0)

        # if env.env.services_processed % self.defragmentation_period != 0:
        #     return DefragmentationOption(0, -1, 0, 0)
        # else:
        #     if len(env.defragmentation_options_available) > 1:
        #         return env.defragmentation_options_available[1]
        #     else:
        #         return DefragmentationOption(0, -1, 0, 0)








def choose_randomly(env: DefragmentationEnv):
    env.defragmentation_options_available = [DefragmentationOption(0,-1,0,0,0,0)]
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
    return DefragmentationOption(0,-1,0,0,0,0)
