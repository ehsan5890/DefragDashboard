import gym
import copy
from typing import Optional
import math
import heapq
import logging
import functools
import random
import numpy as np
from optical_rl_gym.utils import Path, Service
import os

from optical_rl_gym.utils import plot_spectrum_assignment, plot_spectrum_assignment_and_waste

from .optical_network_env import OpticalNetworkEnv


class RMSAEnv(OpticalNetworkEnv):
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
                 defragmentation_period=32,
                 movable_connections = 10,
                 traffic_type = 1):
        super().__init__(topology,
                         episode_length=episode_length,
                         load=load,
                         mean_service_holding_time=mean_service_holding_time,
                         num_spectrum_resources=num_spectrum_resources,
                         node_request_probabilities=node_request_probabilities,
                         seed=seed, allow_rejection=allow_rejection,
                         k_paths=k_paths,
                         incremental_traffic_percentage=incremental_traffic_percentage)
        assert 'modulations' in self.topology.graph
        # specific attributes for elastic optical networks
        self.bit_rate_requested = 0
        self.bit_rate_provisioned = 0
        self.bit_rate_blocked_fragmentation = 0
        self.bit_rate_blocked_lack_resource = 0
        self.episode_bit_rate_requested = 0
        self.episode_bit_rate_provisioned = 0
        self.average_delay_absolute = 0
        self.average_delay_percentage = 0
        self.bit_rate_lower_bound = bit_rate_lower_bound
        self.bit_rate_higher_bound = bit_rate_higher_bound
        self.block_due_to_fragmentation = False
        self.defragmentation_period = defragmentation_period
        self.movable_connections = movable_connections
        self.episode_defragmentation_procedure = 0
        self.episode_num_moves = 0
        self.episode_services_block_resource = 0
        self.episode_services_block_frag = 0
        self.traffic_type = traffic_type

        self.spectrum_slots_allocation = np.full((self.topology.number_of_edges(), self.num_spectrum_resources),
                                                 fill_value=-1, dtype=int)

        # do we allow proactive rejection or not?
        self.reject_action = 1 if allow_rejection else 0

        # defining the observation and action spaces
        self.actions_output = np.zeros((self.k_paths + 1,
                                        self.num_spectrum_resources + 1),
                                       dtype=int)
        self.episode_actions_output = np.zeros((self.k_paths + 1,
                                                self.num_spectrum_resources + 1),
                                               dtype=int)
        self.actions_taken = np.zeros((self.k_paths + 1,
                                       self.num_spectrum_resources + 1),
                                      dtype=int)
        self.episode_actions_taken = np.zeros((self.k_paths + 1,
                                               self.num_spectrum_resources + 1),
                                              dtype=int)
        self.action_space = gym.spaces.MultiDiscrete((self.k_paths + self.reject_action,
                                                      self.num_spectrum_resources + self.reject_action))
        self.observation_space = gym.spaces.Dict(
            {'topology': gym.spaces.Discrete(10),
             'current_service': gym.spaces.Discrete(10)}
        )
        self.action_space.seed(self.rand_seed)
        self.observation_space.seed(self.rand_seed)

        self.logger = logging.getLogger('rmsaenv')
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.warning(
                'Logging is enabled for DEBUG which generates a large number of messages. '
                'Set it to INFO if DEBUG is not necessary.')
        self._new_service = False
        if reset:
            self.reset(only_counters=False)

    def step(self, action):
        path, initial_slot = action[0], action[1]
        self.actions_output[path, initial_slot] += 1
        if path < self.k_paths and initial_slot < self.num_spectrum_resources:  # action is for assigning a path
            slots = self.get_number_slots(self.k_shortest_paths[self.service.source, self.service.destination][path])
            self.logger.debug(
                '{} processing action {} path {} and initial slot {} for {} slots'.format(self.service.service_id,
                                                                                          action, path, initial_slot,
                                                                                          slots))
            if self.is_path_free(self.k_shortest_paths[self.service.source, self.service.destination][path],
                                 initial_slot, slots):
                self._provision_path(self.k_shortest_paths[self.service.source, self.service.destination][path],
                                     initial_slot, slots)
                self.service.accepted = True
                self.actions_taken[path, initial_slot] += 1
                self._add_release(self.service)

                # updating delay
                delay_path = self.k_shortest_paths[self.service.source, self.service.destination][path].length / (
                        2 * (10 ** 2))
                delay_deviation_path_from_shortest_path_absolute = (self.k_shortest_paths[
                                                                        self.service.source, self.service.destination][
                                                                        path].length - \
                                                                    self.k_shortest_paths[
                                                                        self.service.source, self.service.destination][
                                                                        0].length) / (2 * (10 ** 2))
                delay_deviation_path_from_shortest_path_percentage = (
                                                                             delay_deviation_path_from_shortest_path_absolute / delay_path) * 100
                self.average_delay_absolute = ((self.services_accepted * self.average_delay_absolute) +
                                               delay_deviation_path_from_shortest_path_absolute) / (
                                                      self.services_accepted + 1)
                self.average_delay_percentage = ((self.services_accepted * self.average_delay_percentage) +
                                                 delay_deviation_path_from_shortest_path_percentage) / (
                                                        self.services_accepted + 1)
            else:
                self.service.accepted = False
        else:
            self.service.accepted = False
            if self.block_due_to_fragmentation:
                self.bit_rate_blocked_fragmentation += self.service.bit_rate
                self.service_blocked_due_fragmentation += 1
            else:
                self.bit_rate_blocked_lack_resource += self.service.bit_rate
                self.service_blocked_due_lack_resource += 1
        if not self.service.accepted:
            self.actions_taken[self.k_paths, self.num_spectrum_resources] += 1
        self.services_processed += 1
        self._new_service = False
        self._next_service()
        # Periodical defragmentation
        if self.services_processed % self.defragmentation_period == 0:
            # Oldest First Strategy
            self.episode_defragmentation_procedure += 1
            services_to_defragmentation = self.topology.graph['running_services']
            counterrr = 0
            for i in range(10000):
                if i < len(services_to_defragmentation):
                    service_to_defrag = services_to_defragmentation[i]
                    num_slots = service_to_defrag.number_slots
                    old_initial_slot = service_to_defrag.initial_slot
                    new_slot = 10000 # this is a big number
                    for initial_slot in range(0, self.topology.graph['num_spectrum_resources'] - num_slots):
                        if self.is_path_free(service_to_defrag.route,
                                            initial_slot,
                                            num_slots):
                            new_slot = initial_slot
                            break
                    # lets try to find best slot close to the old_initial_slot
                    if new_slot > old_initial_slot:
                        counter = num_slots
                        for initial_slot in range(old_initial_slot - num_slots, old_initial_slot):
                            if initial_slot >=0:
                                if self.is_path_free(
                                        service_to_defrag.route,
                                        initial_slot,
                                        counter):
                                    new_slot = initial_slot
                                    break
                            counter = counter -1
                    if new_slot < old_initial_slot:
                        self.episode_num_moves = self.episode_num_moves + 1
                        self._move_path(service_to_defrag.route, service_to_defrag, new_slot, old_initial_slot, num_slots)
                        counterrr +=1
                        if counterrr == self.movable_connections:
                            break

        if self.service.bit_rate in self.services_processed_bit_rate.keys():
            self.services_processed_bit_rate[self.service.bit_rate] +=1
        else:
            self.services_processed_bit_rate[self.service.bit_rate] = 1

        self.episode_services_processed += 1

        self.bit_rate_requested += self.service.bit_rate
        self.episode_bit_rate_requested += self.service.bit_rate
        figure_folder = './figures'

        self.topology.graph['services'].append(self.service)
        sum_fragmentation_network_external = 0
        sum_fragmentation_network_compactness = 0
        fragmentation_external_links = np.zeros(self.topology.number_of_edges())
        fragmentation_compactness_links = np.zeros(self.topology.number_of_edges())
        for x in self.topology.edges():
            sum_fragmentation_network_external += self.topology[x[0]][x[1]]['external_fragmentation']
            fragmentation_external_links[self.topology[x[0]][x[1]]['id']] = self.topology[x[0]][x[1]]['external_fragmentation']
        external_fragmentation_network = sum_fragmentation_network_external / self.topology.number_of_edges()
        external_fragmentation_deviation = np.std(fragmentation_external_links)

        for x in self.topology.edges():
            sum_fragmentation_network_compactness += self.topology[x[0]][x[1]]['compactness']
            fragmentation_compactness_links[self.topology[x[0]][x[1]]['id']] = self.topology[x[0]][x[1]]['compactness']
        #  print(self.topology[x[0]][x[1]]['compactness'])
        compactness_fragmentation_network = sum_fragmentation_network_compactness / self.topology.number_of_edges()
        compactness_fragmentation_deviation = np.std(fragmentation_compactness_links)

        aa = (self.services_processed - self.services_accepted) / self.services_processed
        reward = self.reward()

        # do we calculate the blocking due to fragmentation correctly?
        # print(self.bit_rate_requested - self.bit_rate_provisioned)
        # print(self.bit_rate_blocked_fragmentation + self.bit_rate_blocked_lack_resource)
        # print(self.services_processed - self.services_accepted)
        # print(self.service_blocked_due_fragmentation + self.service_blocked_due_lack_resource)

        info = {
            'service_blocking_rate': (self.services_processed - self.services_accepted) / self.services_processed,
            'episode_service_blocking_rate': (
                                                     self.episode_services_processed - self.episode_services_accepted) / self.episode_services_processed,
            'bit_rate_blocking_rate': (self.bit_rate_requested - self.bit_rate_provisioned) / self.bit_rate_requested,
            'episode_bit_rate_blocking_rate': (
                                                      self.episode_bit_rate_requested - self.episode_bit_rate_provisioned) / self.episode_bit_rate_requested,
            'external_fragmentation_network_episode': external_fragmentation_network,
            'compactness_fragmentation_network_episode': compactness_fragmentation_network,
            'compactness_network_fragmentation_network_episode': self.topology.graph['compactness'],
            'delay_deviation_absolute': self.average_delay_absolute,
            'delay_deviation_percentage': self.average_delay_percentage,
            'bit_rate_blocking_fragmentation': (self.bit_rate_blocked_fragmentation/self.bit_rate_requested),
            'service_blocking_rate_fragmentation': (self.service_blocked_due_fragmentation/self.services_processed),
            'compactness_fragmentation_deviation': compactness_fragmentation_deviation,
            'external_fragmentation_deviation': external_fragmentation_deviation,
            'service_blocking_rate_100': (self.services_processed_bit_rate[100] - self.services_accepted_bit_rate[100]) / self.services_processed_bit_rate[100],
            'service_blocking_rate_200': (self.services_processed_bit_rate[200] - self.services_accepted_bit_rate[200]) / self.services_processed_bit_rate[200],
            'service_blocking_rate_400': (self.services_processed_bit_rate[400] - self.services_accepted_bit_rate[
                400]) / self.services_processed_bit_rate[400],
            'service_blocked_eopisode': (self.episode_services_processed - self.episode_services_accepted),
            'number_movements_episode': self.episode_num_moves,
            'number_defragmentation_procedure_episode': self.episode_defragmentation_procedure,
            'number_arrivals': self.services_processed,

        }

        # self._new_service = False
        # self._next_service()
        cc = self.observation()
        return self.observation(), reward, self.episode_services_processed == self.episode_length, info

    def reset(self, only_counters=True):
        self.episode_bit_rate_requested = 0
        self.episode_bit_rate_provisioned = 0
        self.episode_services_processed = 0
        self.episode_services_accepted = 0
        self.episode_num_moves = 0
        self.episode_defragmentation_procedure = 0
        self.episode_actions_output = np.zeros((self.k_paths + self.reject_action,
                                                self.num_spectrum_resources + self.reject_action),
                                               dtype=int)
        self.episode_actions_taken = np.zeros((self.k_paths + self.reject_action,
                                               self.num_spectrum_resources + self.reject_action),
                                              dtype=int)

        if only_counters:
            return self.observation()

        super().reset()

        self.bit_rate_requested = 0
        self.bit_rate_provisioned = 0

        self.topology.graph["available_slots"] = np.ones((self.topology.number_of_edges(), self.num_spectrum_resources),
                                                         dtype=int)

        self.spectrum_slots_allocation = np.full((self.topology.number_of_edges(), self.num_spectrum_resources),
                                                 fill_value=-1, dtype=int)

        self.topology.graph["compactness"] = 0.
        self.topology.graph["throughput"] = 0.
        for idx, lnk in enumerate(self.topology.edges()):
            self.topology[lnk[0]][lnk[1]]['external_fragmentation'] = 0.
            self.topology[lnk[0]][lnk[1]]['compactness'] = 0.

        self._new_service = False
        self._next_service()
        return self.observation()

    def render(self, mode='human'):
        return

    def _provision_path(self, path: Path, initial_slot, number_slots):
        # usage
        if not self.is_path_free(path, initial_slot, number_slots):
            raise ValueError("Path {} has not enough capacity on slots {}-{}".format(path.node_list, path, initial_slot,
                                                                                     initial_slot + number_slots))

        self.logger.debug(
            '{} assigning path {} on initial slot {} for {} slots'.format(self.service.service_id, path.node_list,
                                                                          initial_slot, number_slots))
        for i in range(len(path.node_list) - 1):
            self.topology.graph['available_slots'][self.topology[path.node_list[i]][path.node_list[i + 1]]['index'],
            initial_slot:initial_slot + number_slots] = 0
            self.spectrum_slots_allocation[self.topology[path.node_list[i]][path.node_list[i + 1]]['index'],
            initial_slot:initial_slot + number_slots] = self.service.service_id
            self.topology[path.node_list[i]][path.node_list[i + 1]]['services'].append(self.service)
            self.topology[path.node_list[i]][path.node_list[i + 1]]['running_services'].append(self.service)
            self._update_link_stats(path.node_list[i], path.node_list[i + 1])
        self.topology.graph['running_services'].append(self.service)
        self.service.route = path
        self.service.initial_slot = initial_slot
        self.service.number_slots = number_slots
        self._update_network_stats()

        self.services_accepted += 1

        if self.service.bit_rate in self.services_accepted_bit_rate.keys():
            self.services_accepted_bit_rate[self.service.bit_rate] +=1
        else:
            self.services_accepted_bit_rate[self.service.bit_rate] = 1

        self.episode_services_accepted += 1
        self.bit_rate_provisioned += self.service.bit_rate
        self.episode_bit_rate_provisioned += self.service.bit_rate
        # calculating_delay

    def _release_path(self, service: Service):
        for i in range(len(service.route.node_list) - 1):
            self.topology.graph['available_slots'][
            self.topology[service.route.node_list[i]][service.route.node_list[i + 1]]['index'],
            service.initial_slot:service.initial_slot + service.number_slots] = 1
            self.spectrum_slots_allocation[
            self.topology[service.route.node_list[i]][service.route.node_list[i + 1]]['index'],
            service.initial_slot:service.initial_slot + service.number_slots] = -1
            self.topology[service.route.node_list[i]][service.route.node_list[i + 1]]['running_services'].remove(service)
            self._update_link_stats(service.route.node_list[i], service.route.node_list[i + 1])
        self.topology.graph['running_services'].remove(service)

    def _move_path(self, path: Path, service: Service, new_initial_slot, old_initial_slot, number_slots):
        # firstly, provision service, then release the service (make before break)
        # ehsan comment, these lines should be removed
        #if not self.is_path_free(path, new_initial_slot, number_slots):
        #    raise ValueError(
        #        "Path {} has not enough capacity on slots, moving is not possible {}-{}".format(path.node_list,
        #                                                                                       path, new_initial_slot,
        #                                                                                        new_initial_slot + number_slots))

        self.logger.debug(
            '{} moving path {} on initial slot {} for {} slots'.format(service.service_id, path.node_list,
                                                                       new_initial_slot, number_slots))
        for i in range(len(path.node_list) - 1):
            self.topology.graph['available_slots'][self.topology[path.node_list[i]][path.node_list[i + 1]]['index'],
            new_initial_slot:new_initial_slot + number_slots] = 0
            self.spectrum_slots_allocation[self.topology[path.node_list[i]][path.node_list[i + 1]]['index'],
            new_initial_slot:new_initial_slot + number_slots] = service.service_id
            self.topology[path.node_list[i]][path.node_list[i + 1]]['services'].append(service)
            self.topology[path.node_list[i]][path.node_list[i + 1]]['running_services'].append(service)
        service.initial_slot = new_initial_slot # Update service initial slot
        self.topology.graph['running_services'].append(service)

        # secondly, release the first path, moves from old slot to the new slot.

        for i in range(len(service.route.node_list) - 1):
            if old_initial_slot - new_initial_slot >= number_slots:
                self.topology.graph['available_slots'][
                self.topology[service.route.node_list[i]][service.route.node_list[i + 1]]['index'],
                old_initial_slot:old_initial_slot + service.number_slots] = 1
                self.spectrum_slots_allocation[
                self.topology[service.route.node_list[i]][service.route.node_list[i + 1]]['index'],
                old_initial_slot:old_initial_slot + service.number_slots] = -1
            else:
                self.topology.graph['available_slots'][
                self.topology[service.route.node_list[i]][service.route.node_list[i + 1]]['index'],
                new_initial_slot + number_slots:old_initial_slot + service.number_slots] = 1
                self.spectrum_slots_allocation[
                self.topology[service.route.node_list[i]][service.route.node_list[i + 1]]['index'],
                new_initial_slot + number_slots:old_initial_slot + service.number_slots] = -1
            self.topology[service.route.node_list[i]][service.route.node_list[i + 1]]['running_services'].remove(
                    service)
            self.topology[service.route.node_list[i]][service.route.node_list[i + 1]]['services'].remove(service)
            self._update_link_stats(service.route.node_list[i], service.route.node_list[i + 1])
        self.topology.graph['running_services'].remove(service)
        self._update_network_stats()

    def _update_network_stats(self):
        last_update = self.topology.graph['last_update']
        time_diff = self.current_time - last_update
        if self.current_time > 0:
            last_throughput = self.topology.graph['throughput']
            last_compactness = self.topology.graph['compactness']

            cur_throughput = 0.

            for service in self.topology.graph["running_services"]:
                cur_throughput += service.bit_rate

            throughput = ((last_throughput * last_update) + (cur_throughput * time_diff)) / self.current_time
            self.topology.graph['throughput'] = throughput

            compactness = ((last_compactness * last_update) + (self._get_network_compactness() * time_diff)) / \
                          self.current_time
            self.topology.graph['compactness'] = compactness

        self.topology.graph['last_update'] = self.current_time

    def _update_link_stats(self, node1: str, node2: str):
        last_update = self.topology[node1][node2]['last_update']
        time_diff = self.current_time - self.topology[node1][node2]['last_update']
        if self.current_time > 0:
            last_util = self.topology[node1][node2]['utilization']
            cur_util = (self.num_spectrum_resources - np.sum(
                self.topology.graph['available_slots'][self.topology[node1][node2]['index'], :])) / \
                       self.num_spectrum_resources
            utilization = ((last_util * last_update) + (cur_util * time_diff)) / self.current_time
            self.topology[node1][node2]['utilization'] = utilization

            slot_allocation = self.topology.graph['available_slots'][self.topology[node1][node2]['index'], :]

            # implementing fragmentation from https://ieeexplore.ieee.org/abstract/document/6421472
            last_external_fragmentation = self.topology[node1][node2]['external_fragmentation']
            last_compactness = self.topology[node1][node2]['compactness']

            cur_external_fragmentation = 0.
            cur_link_compactness = 0.
            if np.sum(slot_allocation) > 0:
                initial_indices, values, lengths = RMSAEnv.rle(slot_allocation)

                # computing external fragmentation from https://ieeexplore.ieee.org/abstract/document/6421472
                unused_blocks = [i for i, x in enumerate(values) if x == 1]
                max_empty = 0
                if len(unused_blocks) > 1 and unused_blocks != [0, len(values) - 1]:
                    max_empty = max(lengths[unused_blocks])
                cur_external_fragmentation = 1. - (float(max_empty) / float(np.sum(slot_allocation)))

                # computing link spectrum compactness from https://ieeexplore.ieee.org/abstract/document/6421472
                used_blocks = [i for i, x in enumerate(values) if x == 0]

                if len(used_blocks) > 1:
                    lambda_min = initial_indices[used_blocks[0]]
                    lambda_max = initial_indices[used_blocks[-1]] + lengths[used_blocks[-1]]

                    # evaluate again only the "used part" of the spectrum
                    internal_idx, internal_values, internal_lengths = RMSAEnv.rle(
                        slot_allocation[lambda_min:lambda_max])
                    unused_spectrum_slots = np.sum(1 - internal_values)

                    if unused_spectrum_slots > 0:
                        cur_link_compactness = ((lambda_max - lambda_min) / np.sum(1 - slot_allocation)) * (
                                1 / unused_spectrum_slots)
                    else:
                        cur_link_compactness = 1.
                else:
                    cur_link_compactness = 1.

            external_fragmentation = ((last_external_fragmentation * last_update) + (
                    cur_external_fragmentation * time_diff)) / self.current_time
            self.topology[node1][node2]['external_fragmentation'] = external_fragmentation

            link_compactness = ((last_compactness * last_update) + (
                    cur_link_compactness * time_diff)) / self.current_time
            self.topology[node1][node2]['compactness'] = link_compactness

        self.topology[node1][node2]['last_update'] = self.current_time

    def _next_service(self):
        if self._new_service:
            return
        # at = self.current_time + self.rng.expovariate(1 / self.mean_service_inter_arrival_time)
        # self.current_time = at
        # ht = self.rng.expovariate(1 / self.mean_service_holding_time)
        # deploying incremental traffic
        if random.random() < (self.incremental_traffic_percentage) / 100:
            at = self.current_time + self.rng.expovariate(1 / self.mean_service_inter_arrival_time)
            self.current_time = at
            ht = self.rng.expovariate(1 / self.mean_service_holding_time)
        else:
            at = self.current_time + self.rng.expovariate(1 / self.mean_service_inter_arrival_time_dynamic)
            self.current_time = at
            ht = self.rng.expovariate(
                1 / (self.mean_service_holding_time / self.incremental_dynamic_proportion_mean_time))

        src, src_id, dst, dst_id = self._get_node_pair()

        # bit_rate = self.rng.randint(self.bit_rate_lower_bound, self.bit_rate_higher_bound)
        # Developing bit-rate for Telia
        bit_rate_threshold = self.rng.random()
        if self.traffic_type ==1:
            if bit_rate_threshold < 0.5:
                bit_rate = 100
            elif 0.5 <= bit_rate_threshold < 0.8:
                bit_rate = 200
            else:
                bit_rate = 400
        elif self.traffic_type ==2:
            if bit_rate_threshold < 0.33:
                bit_rate = 100
            elif 0.33 <= bit_rate_threshold < 0.66:
                bit_rate = 200
            else:
                bit_rate = 400


        self.previous_service = self.service
        # release connections up to this point
        while len(self._events) > 0:
            (time, service_to_release) = heapq.heappop(self._events)
            if time <= self.current_time:
                self._release_path(service_to_release)
            else:  # release is not to be processed yet
                self._add_release(service_to_release)  # puts service back in the queue
                break  # breaks the loop
        ### this is not correct, because we reset the service_episode_counter, lets fix it,
        # self.service = Service(self.episode_services_processed, src, src_id,
        #                        destination=dst, destination_id=dst_id,
        #                        arrival_time=at, holding_time=ht, bit_rate=bit_rate)

        self.service = Service(self.services_processed, src, src_id,
                               destination=dst, destination_id=dst_id,
                               arrival_time=at, holding_time=ht, bit_rate=bit_rate)

        self._new_service = True

    def _get_path_slot_id(self, action: int):
        """
        Decodes the single action index into the path index and the slot index to be used.

        :param action: the single action index
        :return: path index and initial slot index encoded in the action
        """
        path = int(action / self.num_spectrum_resources)
        initial_slot = action % self.num_spectrum_resources
        return path, initial_slot

    def get_number_slots(self, path: Path) -> int:
        """
        Method that computes the number of spectrum slots necessary to accommodate the service request into the path.
        The method already adds the guardband.
        """
        return math.ceil(self.service.bit_rate / path.best_modulation['capacity']) + 1

    def is_path_free(self, path: Path, initial_slot: int, number_slots: int) -> bool:
        if initial_slot + number_slots > self.num_spectrum_resources:
            # logging.debug('error index' + env.parameters.rsa_algorithm)
            return False
        for i in range(len(path.node_list) - 1):
            if np.any(self.topology.graph['available_slots'][
                      self.topology[path.node_list[i]][path.node_list[i + 1]]['index'],
                      initial_slot:initial_slot + number_slots] == 0):
                return False
        return True

    def get_available_slots(self, path: Path):
        available_slots = functools.reduce(np.multiply,
                                           self.topology.graph["available_slots"][
                                           [self.topology[path.node_list[i]][path.node_list[i + 1]]['id']
                                            for i in range(len(path.node_list) - 1)], :])
        return available_slots

    def get_number_cuts(self, path: Path, initial_slot: int, number_slots: int):
        number_cuts = 0
        if initial_slot != 0:
            for i in range(len(path.node_list) - 1):
                number_cuts += self.topology.graph["available_slots"][
                                   self.topology[path.node_list[i]][path.node_list[i + 1]]['id'], initial_slot - 1] * \
                               self.topology.graph["available_slots"][
                                   self.topology[path.node_list[i]][path.node_list[i + 1]][
                                       'id'], initial_slot + number_slots]

        return number_cuts

    def get_number_alignment_factor(self, path: Path, initial_slot: int, number_slots: int):
        number_alignments = 0
        for i in range(len(path.node_list) - 1):
            for id, adj in enumerate(self.topology.adj[path.node_list[i]]):
                if adj not in path.node_list:
                    if i == 0:
                        # if we considers all slots seperately to calculate number of allignments.
                        # number_alignments += np.sum(
                        #     self.topology.graph["available_slots"][self.topology[adj][path.node_list[i]]['id'],
                        #     initial_slot:initial_slot + number_slots])
                        # If we consider all slots as block, and calculate number of free blocks.
                        number_alignments += functools.reduce(np.multiply,
                                                              self.topology.graph["available_slots"][
                                                              self.topology[adj][path.node_list[i]]['id'],
                                                              initial_slot:initial_slot + number_slots])
                    elif i == len(path.node_list) - 1:
                        # number_alignments += np.sum(
                        #     self.topology.graph["available_slots"][self.topology[path.node_list[i]][adj]['id'],
                        #     initial_slot:initial_slot + number_slots])
                        # If we consider all slots as block, and calculate number of free blocks.
                        number_alignments += functools.reduce(np.multiply,
                                                              self.topology.graph["available_slots"][
                                                              self.topology[path.node_list[i]][adj]['id'],
                                                              initial_slot:initial_slot + number_slots])
                    else:
                        # if we considers all slots seperately to calculate number of allignments.
                        # number_alignments += np.sum(
                        #     self.topology.graph["available_slots"][self.topology[path.node_list[i]][adj]['id'],
                        #     initial_slot:initial_slot + number_slots])
                        # number_alignments += np.sum(
                        #     self.topology.graph["available_slots"][self.topology[adj][path.node_list[i]]['id'],
                        #     initial_slot:initial_slot + number_slots])

                        # If we consider all slots as block, and calculate number of free blocks.
                        number_alignments += functools.reduce(np.multiply,
                                                              self.topology.graph["available_slots"][
                                                              self.topology[path.node_list[i]][adj]['id'],
                                                              initial_slot:initial_slot + number_slots])
                        number_alignments += functools.reduce(np.multiply,
                                                              self.topology.graph["available_slots"][
                                                              self.topology[adj][path.node_list[i]]['id'],
                                                              initial_slot:initial_slot + number_slots])
        return number_alignments

    def rle(inarray):
        """ run length encoding. Partial credit to R rle function.
            Multi datatype arrays catered for including non Numpy
            returns: tuple (runlengths, startpositions, values) """
        # from: https://stackoverflow.com/questions/1066758/find-length-of-sequences-of-identical-values-in-a-numpy-array-run-length-encodi
        ia = np.asarray(inarray)  # force numpy
        n = len(ia)
        if n == 0:
            return (None, None, None)
        else:
            y = np.array(ia[1:] != ia[:-1])  # pairwise unequal (string safe)
            i = np.append(np.where(y), n - 1)  # must include last element posi
            z = np.diff(np.append(-1, i))  # run lengths
            p = np.cumsum(np.append(0, z))[:-1]  # positions
            return p, ia[i], z

    def get_available_blocks(self, path):
        # get available slots across the whole path
        # 1 if slot is available across all the links
        # zero if not
        available_slots = self.get_available_slots(
            self.k_shortest_paths[self.service.source, self.service.destination][path])

        # getting the number of slots necessary for this service across this path
        slots = self.get_number_slots(self.k_shortest_paths[self.service.source, self.service.destination][path])

        # getting the blocks
        initial_indices, values, lengths = RMSAEnv.rle(available_slots)

        # selecting the indices where the block is available, i.e., equals to one
        available_indices = np.where(values == 1)

        # selecting the indices where the block has sufficient slots
        sufficient_indices = np.where(lengths >= slots)

        # getting the intersection, i.e., indices where the slots are available in sufficient quantity
        # and using only the J first indices
        final_indices = np.intersect1d(available_indices, sufficient_indices)[:self.j]

        return initial_indices[final_indices], lengths[final_indices]

    def _get_network_compactness(self):
        # implementing network spectrum compactness from https://ieeexplore.ieee.org/abstract/document/6476152

        sum_slots_paths = 0  # this accounts for the sum of all Bi * Hi

        for service in self.topology.graph["running_services"]:
            sum_slots_paths += service.number_slots * service.route.hops

        # this accounts for the sum of used blocks, i.e.,
        # \sum_{j=1}^{M} (\lambda_{max}^j - \lambda_{min}^j)
        sum_occupied = 0

        # this accounts for the number of unused blocks \sum_{j=1}^{M} K_j
        sum_unused_spectrum_blocks = 0

        for n1, n2 in self.topology.edges():
            # getting the blocks
            initial_indices, values, lengths = \
                RMSAEnv.rle(self.topology.graph['available_slots'][self.topology[n1][n2]['index'], :])
            used_blocks = [i for i, x in enumerate(values) if x == 0]
            if len(used_blocks) > 1:
                lambda_min = initial_indices[used_blocks[0]]
                lambda_max = initial_indices[used_blocks[-1]] + lengths[used_blocks[-1]]
                sum_occupied += lambda_max - lambda_min  # we do not put the "+1" because we use zero-indexed arrays

                # evaluate again only the "used part" of the spectrum
                internal_idx, internal_values, internal_lengths = RMSAEnv.rle(
                    self.topology.graph['available_slots'][self.topology[n1][n2]['index'], lambda_min:lambda_max])
                sum_unused_spectrum_blocks += np.sum(internal_values)

        if sum_unused_spectrum_blocks > 0:
            cur_spectrum_compactness = (sum_occupied / sum_slots_paths) * (self.topology.number_of_edges() /
                                                                           sum_unused_spectrum_blocks)
        else:
            cur_spectrum_compactness = 1.

        return cur_spectrum_compactness


def shortest_path_first_fit(env: RMSAEnv, service : Optional[Service] = None) -> int:
    if service == None:
        service = env.service
        num_slots = env.get_number_slots(env.k_shortest_paths[service.source, service.destination][0])
    else:
        num_slots = service.number_slots
    for initial_slot in range(0, env.topology.graph['num_spectrum_resources'] - num_slots):
        if env.is_path_free(env.k_shortest_paths[service.source, service.destination][0], initial_slot,
                            num_slots):
            return [0, initial_slot]

    free_slots = np.sum(env.get_available_slots(env.k_shortest_paths[service.source, service.destination][0]))
    if num_slots < free_slots:
        #env.env.block_due_to_fragmentation = True  # we wrote two env here, since the first env is for monitor and the second one is for RMSA env.
        env.block_due_to_fragmentation = True



def shortest_available_path_first_fit(env: RMSAEnv, ) -> int:
    for idp, path in enumerate(env.k_shortest_paths[env.service.source, env.service.destination]):
        num_slots = env.get_number_slots(path)
        for initial_slot in range(0, env.topology.graph['num_spectrum_resources'] - num_slots):
            if env.is_path_free(path, initial_slot, num_slots):
                return [idp, initial_slot]

    free_slots = np.sum(env.get_available_slots(path))
    if num_slots < free_slots:
        env.block_due_to_fragmentation = True  # we wrote two env here, since the first env is for monitor and the second one is for RMSA env.
    return [env.topology.graph['k_paths'], env.topology.graph['num_spectrum_resources']]


def least_loaded_path_first_fit(env: RMSAEnv) -> int:
    max_free_slots = 0
    flag_assigned_path = False
    action = [env.topology.graph['k_paths'], env.topology.graph['num_spectrum_resources']]
    for idp, path in enumerate(env.k_shortest_paths[env.service.source, env.service.destination]):
        num_slots = env.get_number_slots(path)
        for initial_slot in range(0, env.topology.graph['num_spectrum_resources'] - num_slots):
            if env.is_path_free(path, initial_slot, num_slots):
                free_slots = np.sum(env.get_available_slots(path))
                if free_slots > max_free_slots:
                    action = [idp, initial_slot]
                    max_free_slots = free_slots
                    flag_assigned_path = True
                break  # breaks the loop for the initial slot
    if not flag_assigned_path:
        free_slots = np.sum(env.get_available_slots(path))
        if num_slots < free_slots:
            env.env.block_due_to_fragmentation = True  # we wrote two env here, since the first env is for monitor and the second one is for RMSA env.

    return action


def Fragmentation_alignment_aware_RMSA(env: RMSAEnv) -> int:
    min_factor = math.inf
    flag_assigned_path = False
    action = [env.topology.graph['k_paths'], env.topology.graph['num_spectrum_resources']]
    # for idp, path in enumerate(env.k_shortest_paths[env.service.source, env.service.destination]):
    # If we want to simulate the shortest path for FAAR algorithm, we use the next line.
    for idp, path in enumerate([env.k_shortest_paths[env.service.source, env.service.destination][0]]):
        num_slots = env.get_number_slots(path)
        for initial_slot in range(0, env.topology.graph['num_spectrum_resources'] - num_slots):
            if env.is_path_free(path, initial_slot, num_slots):
                free_slots = np.sum(env.get_available_slots(path))
                number_cuts = env.get_number_cuts(path, initial_slot, num_slots)
                number_alignment_factor = env.get_number_alignment_factor(path, initial_slot, num_slots)
                fragment_alignment_factor = (path.hops * num_slots + number_cuts + number_alignment_factor) / free_slots
                # Calculating remaining time for the services.
                remaining_times = 0
                for i in range(len(path.node_list) - 1):
                    for cheking_slot in [initial_slot-1, initial_slot + num_slots]:
                        if env.topology.graph['available_slots'][
                                  env.topology[path.node_list[i]][path.node_list[i + 1]]['index'],
                                  cheking_slot] == 0:
                           #print(len(env.topology['1']['2']['services']))
                           service_id = env.spectrum_slots_allocation[env.topology[path.node_list[i]][path.node_list[i + 1]]['index'], cheking_slot]
                           if env.topology.graph['services'][service_id].arrival_time + env.topology.graph['services'][service_id].holding_time\
                                   - env.current_time > 0:
                               remaining_times += env.topology.graph['services'][service_id].arrival_time + env.topology.graph['services'][service_id].holding_time\
                                                  - env.current_time

                # fragment_alignment_factor = (path.hops * num_slots + number_cuts +
                #                              number_alignment_factor) / (free_slots + remaining_times)
                if fragment_alignment_factor < min_factor:
                    action = [idp, initial_slot]
                    min_factor = fragment_alignment_factor
                    flag_assigned_path = True
            # break # breaks the loop for the initial slot
    if not flag_assigned_path:
        free_slots = np.sum(env.get_available_slots(path))
        if num_slots < free_slots:
            env.env.block_due_to_fragmentation = True  # we wrote two env here, since the first env is for monitor and the second one is for RMSA env.

    return action


class SimpleMatrixObservation(gym.ObservationWrapper):

    def __init__(self, env: RMSAEnv):
        super().__init__(env)
        shape = self.env.topology.number_of_nodes() * 2 \
                + self.env.topology.number_of_edges() * self.env.num_spectrum_resources
        self.observation_space = gym.spaces.Box(low=0, high=1, dtype=np.uint8, shape=(shape,))
        self.action_space = env.action_space

    def observation(self, observation):
        source_destination_tau = np.zeros((2, self.env.topology.number_of_nodes()))
        min_node = min(self.env.service.source_id, self.env.service.destination_id)
        max_node = max(self.env.service.source_id, self.env.service.destination_id)
        source_destination_tau[0, min_node] = 1
        source_destination_tau[1, max_node] = 1
        spectrum_obs = copy.deepcopy(self.topology.graph["available_slots"])
        return np.concatenate((source_destination_tau.reshape((1, np.prod(source_destination_tau.shape))),
                               spectrum_obs.reshape((1, np.prod(spectrum_obs.shape)))), axis=1) \
            .reshape(self.observation_space.shape)


class PathOnlyFirstFitAction(gym.ActionWrapper):

    def __init__(self, env: RMSAEnv):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(self.env.k_paths + self.env.reject_action)
        self.observation_space = env.observation_space

    def action(self, action):
        if action < self.env.k_paths:
            num_slots = self.env.get_number_slots(self.env.k_shortest_paths[self.env.service.source,
                                                                            self.env.service.destination][action])
            for initial_slot in range(0, self.env.topology.graph['num_spectrum_resources'] - num_slots):
                if self.env.is_path_free(self.env.k_shortest_paths[self.env.service.source,
                                                                   self.env.service.destination][action],
                                         initial_slot, num_slots):
                    return [action, initial_slot]
        return [self.env.topology.graph['k_paths'], self.env.topology.graph['num_spectrum_resources']]

    def step(self, action):
        return self.env.step(self.action(action))

