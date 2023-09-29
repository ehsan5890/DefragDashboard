from itertools import islice
import networkx as nx
import numpy as np

from dataclasses import dataclass


class Path:

    def __init__(self, path_id, node_list, length, best_modulation=None):
        self.path_id = path_id
        self.node_list = node_list
        self.length = length
        self.best_modulation = best_modulation
        self.hops = len(node_list) - 1


class Service:

    def __init__(self, service_id, source, source_id, destination=None, destination_id=None, arrival_time=None,
                 holding_time=None, bit_rate=None, best_modulation=None, service_class=None, number_slots=None):
        self.service_id = service_id
        self.arrival_time = arrival_time
        self.holding_time = holding_time
        self.source = source
        self.source_id = source_id
        self.destination = destination
        self.destination_id = destination_id
        self.bit_rate = bit_rate
        self.service_class = service_class
        self.best_modulation = best_modulation
        self.number_slots = number_slots
        self.route = None
        self.initial_slot = None
        self.accepted = False

    def __str__(self):
        msg = '{'
        msg += '' if self.bit_rate is None else f'br: {self.bit_rate}, '
        msg += '' if self.service_class is None else f'cl: {self.service_class}, '
        return f'Serv. {self.service_id} ({self.source} -> {self.destination})' + msg


def start_environment(env, steps):
    done = True
    for i in range(steps):
        if done:
            env.reset()
        while not done:
            action = env.action_space.sample()
            _, _, done, _ = env.step(action)
    return env


def get_k_shortest_paths(G, source, target, k, weight=None):
    """
    Method from https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.simple_paths.shortest_simple_paths.html#networkx.algorithms.simple_paths.shortest_simple_paths
    """
    return list(islice(nx.shortest_simple_paths(G, source, target, weight=weight), k))


def get_path_weight(graph, path, weight='length'):
    return np.sum([graph[path[i]][path[i + 1]][weight] for i in range(len(path) - 1)])


def random_policy(env):
    random_slot_path = env.action_space.sample()
    # for telia topology, we always pick the first path, it is kind of cheating, since random allocation is not appropriate for telia topology.
    #random_slot_path[0] = 0

    return random_slot_path


def evaluate_heuristic(env, heuristic, n_eval_episodes=1,
                       render=False, callback=None, reward_threshold=None,
                       return_episode_rewards=False,):
    episode_rewards, episode_lengths = [], []
    for _ in range(n_eval_episodes):
        obs = env.reset()
        done, state = False, None
        episode_reward = 0.0
        episode_length = 0
        while not done:
            action = heuristic(env)
            obs, reward, done, _info = env.step(action)
            episode_reward += reward
            if callback is not None:
                callback(locals(), globals())
            episode_length += 1
            if render:
                env.render()
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    if reward_threshold is not None:
        assert mean_reward > reward_threshold, 'Mean reward below threshold: ' \
                                               '{:.2f} < {:.2f}'.format(mean_reward, reward_threshold)
    if return_episode_rewards:
        return episode_rewards, episode_lengths
    return mean_reward, std_reward


import copy
import itertools
import numpy as np
import matplotlib.pyplot as plt


def plot_spectrum_assignment(topology, vector, values=False, filename=None, show=True, figsize=(15, 10), title=None):
    plt.figure(figsize=figsize)
    cmap = copy.copy(plt.cm.viridis)
    cmap.set_under(color='white')

    cmap_reverse = plt.cm.viridis_r
    cmap_reverse.set_under(color='black')
    p = plt.pcolor(vector, cmap=cmap, vmin=-0.0001, edgecolors='gray')
    #     p.set_rasterized(False)

    if values:
        thresh = vector.max() / 2.
        for i, j in itertools.product(range(vector.shape[0]), range(vector.shape[1])):
            if vector[i, j] == -1:
                continue
            else:
                text = '{:.0f}'.format(vector[i, j])
            color = cmap_reverse(vector[i, j] / vector.max())
            diff_color = np.sum(np.array(color) - np.array(cmap(vector[i, j] / vector.max())))
            if np.abs(diff_color) < 0.1:
                red = max(color[0] + 0.5, 1.)
                green = max(color[1] - 0.5, 0.)
                blue = max(color[2] - 0.5, 0.)
                color = (red, blue, green)
            #             print(i, j, vector[i, j], diff_color)
            plt.text(j + 0.5, i + 0.5, text,
                     horizontalalignment="center", verticalalignment='center',
                     color=color)

    plt.xlabel('Frequency slot')
    plt.ylabel('Link')
    if title is not None:
        plt.title(title)
    #     plt.yticks([topology.edges[link]['id']+.5 for link in topology.edges()], [link[0] + '-' + link[1] for link in topology.edges()])
    plt.yticks([topology.edges[link]['id'] + .5 for link in topology.edges()],
               [f'{topology.edges[link]["id"]} ({link[0]}-{link[1]})' for link in topology.edges()])
    #     plt.colorbar()
    plt.tight_layout()
    plt.xticks([x + 0.5 for x in plt.xticks()[0][:-1]], [x for x in plt.xticks()[1][:-1]])
    if filename is not None:
        plt.savefig(filename)
    #if show:
        #plt.show()

    plt.close()


def plot_spectrum_assignment_and_waste(topology, vector, vector_wasted, values=False, filename=None, show=True,
                                       figsize=(15, 10), title=None):
    plt.figure(figsize=figsize)

    cmap = copy.deepcopy(plt.cm.viridis)
    cmap.set_under(color='white')

    cmap_reverse = copy.copy(plt.cm.viridis_r)
    cmap_reverse.set_under(color='black')
    p = plt.pcolor(vector, cmap=cmap, vmin=-0.0001, edgecolors='gray')
    #     p.set_rasterized(False)

    if values:
        fmt = 'd'
        thresh = vector.max() / 2.
        for i, j in itertools.product(range(vector.shape[0]), range(vector.shape[1])):
            if vector[i, j] != -1:
                text = format(vector[i, j], fmt)
                color = cmap_reverse(vector[i, j] / vector.max())
                plt.text(j + 0.5, i + 0.5, text,
                         horizontalalignment="center", verticalalignment='center',
                         color=color)

            if vector_wasted[i, j] != -1:
                text = format(vector_wasted[i, j], fmt)
                plt.text(j + 0.2, i + 0.5, text,
                         horizontalalignment="center", verticalalignment='center',
                         color='red')

    plt.xlabel('Frequency slot')
    plt.ylabel('Link')
    #     plt.colorbar()
    plt.yticks([topology.edges[link]['id'] + .5 for link in topology.edges()],
               [f'{topology.edges[link]["id"]} ({link[0]}-{link[1]})' for link in topology.edges()])

    plt.tight_layout()
    plt.xticks([x + 0.5 for x in plt.xticks()[0][:-1]], [x for x in plt.xticks()[1][:-1]])
    if filename is not None:
        plt.savefig(filename)
    #if show:
        #plt.show()

    plt.close()


@dataclass
class DefragmentationOption:
    service: Service
    starting_slot: int
    size_of_free_block: int
    start_of_block: bool
    r_frag_before :  float
    r_frag_after : float
    r_frag_diff : float = 0
    number_cut_before : int = 0
    number_cut_after: int = 0
    frag_size_before : int = 0
    frag_size_after: int = 0
    shanon_entropy_before : float = 0
    shanon_entropy_after: float = 0

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx