import argparse
import sys
from random import randint
import pickle

from PyQt6.QtWidgets import QApplication
from optical_rl_gym.envs.rmsa_env import shortest_available_path_first_fit
import gym
from gui import MainWindow
from tapi import TAPIClient
import networkx as nx
import matplotlib.pyplot as plt

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # TODO: create a parser for the arguments
    # - path to the environment pickle file, None creates a new environment
    # - path to the trained agent
    # example: https://github.com/carlosnatalino/python-simple-anycast-wdm-simulator/blob/8fda7f7b19aa092b15d46578f678d45e392b872a/run.py#L140C5-L140C39
    parser.add_argument('-tf', '--topology', default='nsfnet_chen_eon', help='Network topology file to be used')
    parser.add_argument('-s', '--seed', type=int, default=10,help='random seed value')
    parser.add_argument('-al', '--allow_rejection', default=True)
    parser.add_argument('-l', '--load', type=int,  default=60)
    parser.add_argument('-mh', '--mean_service_holding_time', type=float, default=25)
    parser.add_argument('-el', '--episode_length', type=int,  default=200)
    parser.add_argument('-n', '--num_spectrum_resources', type=int,  default=320)
    parser.add_argument('-ic', '--incremental_traffic_percentage', type=int, default=80)
    parser.add_argument('-rmsa', '--rmsa_function', default=shortest_available_path_first_fit)
    args = parser.parse_args()
    topology_name = 'nsfnet_chen_eon'
    with open(f'./defrag/examples/topologies/{topology_name}_5-paths.h5', 'rb') as f:
        topology = pickle.load(f)
    env_args = dict(topology= topology, seed=10, allow_rejection=True, load=60,
                    mean_service_holding_time=0.5,
                    episode_length=200, num_spectrum_resources=320,
                    incremental_traffic_percentage=320,
                    rmsa_function=shortest_available_path_first_fit,
                    )
    # TODO:
    # - create an Gym environment or load existing environment from pickle file
    # env = None  # from the assets folder
    env = gym.make('Defragmentation-v0', **env_args)

    # load the agent from the assets folder
    agent = None

    # create a TAPI client and make sure it connects
    tapi_client = TAPIClient()

    app = QApplication(sys.argv)
    w = MainWindow(env, agent, tapi_client)  # TODO: pass the environment to the main window
    w.show()
    app.exec()
