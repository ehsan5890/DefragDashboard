import os
import pickle
import numpy as np

from IPython.display import clear_output

# %matplotlib inline
# %config InlineBackend.figure_format = 'svg'

import stable_baselines3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.ppo.policies import MlpPolicy
# from stable_baselines3.dqn.policies import MlpPolicy
from stable_baselines3.common import results_plotter

from stable_baselines3.common.env_util import make_vec_env
from vec_monitor import VecMonitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure, Logger
stable_baselines3.__version__ # printing out stable_baselines version used
from stable_baselines3.common.utils import set_random_seed
from multiprocessing import Process, Pool
from itertools import product
import copy

import gym


# callback from https://stable-baselines.readthedocs.io/en/master/guide/examples.html#using-callback-monitoring-training
class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                 # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print("Num timesteps: {} - ".format(self.num_timesteps), end="")
                    print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))
                  # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print("Saving new best model to {}".format(self.save_path))
                        self.model.save(self.save_path)
                if self.verbose > 0:
                    clear_output(wait=True)

        return True




from optical_rl_gym.envs.rmsa_env import shortest_path_first_fit, shortest_available_path_first_fit
# loading the topology binary file containing the graph and the k-shortest paths
# if you want to generate your own binary topology file, check examples/create_topology_rmsa.py

def make_env(env_id: str, env_arg_f,  rank: int, seed: int = 0):


    def _init() -> gym.Env:
        env = gym.make(env_id, **env_arg_f)
        env.seed(seed + rank)
        return env

    set_random_seed(seed)
    return _init

def DRLDefrag( env_arg_f, log_dir_f, total_timesteps_f, algorithm, number_layer, ent_coef, number_neorons, discount_factor,
               learning_rate, penalty_cycle, penalty_movement, callback, load ):


    # env = gym.make('DeepDefragmentation-v0', **env_arg_f)
    # vec_env = make_vec_env('DeepDefragmentation-v0', **env_arg_f, n_envs=4)

    env_id = "DeepDefragmentation-v0"
    num_cpu = 4  # Number of processes to use
    # Create the vectorized environment
    envs = DummyVecEnv([make_env(env_id, env_arg_f,  i) for i in range(num_cpu)])
    envs = VecMonitor(envs, log_dir_f + f'training-dqn',
                  info_keywords=('episode_service_blocking_rate', 'service_blocking_rate',
                                 'reward', 'number_movements',
                                 'number_defragmentation_procedure', 'number_arrivals', 'bit_rate_blocking_rate',
                                 'number_movements_episode',
                                 'number_defragmentation_procedure_episode', 'service_blocked_eopisode',
                                 'number_options', 'existing_options'
                                 ))

    # env = Monitor(env, log_dir_f + f'training-dqn',
    #               info_keywords=('episode_service_blocking_rate', 'service_blocking_rate',
    #                              'reward', 'number_movements',
    #                              'number_defragmentation_procedure', 'number_arrivals', 'bit_rate_blocking_rate',
    #                              'number_movements_episode',
    #                              'number_defragmentation_procedure_episode', 'service_blocked_eopisode',
    #                              'number_options', 'existing_options'
    #                              ))
    tmp_path = "./tmp/sb3_log/"
    os.makedirs(tmp_path, exist_ok=True)
    # set up logger
    new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])

    policy_args = dict(net_arch=number_layer * [number_neorons])  # we use the elu activation functions

    if algorithm ==2 :

        agent = PPO(stable_baselines3.ppo.policies.MlpPolicy, envs, verbose=0, tensorboard_log=f"./tb/PPO-DeepDefrag-v0/{load}-{penalty_cycle}-{penalty_movement}-{algorithm}-{number_layer}-{ent_coef}-{number_neorons}-{discount_factor}-{learning_rate}",
                    policy_kwargs=policy_args,
                    gamma=discount_factor, learning_rate=learning_rate, ent_coef=ent_coef, )
    elif algorithm==1:
        agent = DQN(stable_baselines3.dqn.policies.MlpPolicy, envs, verbose=0, tensorboard_log=f"./tb/DQN-DeepDefrag-v0/{load}-{penalty_cycle}-{penalty_movement}-{algorithm}-{number_layer}-{ent_coef}-{number_neorons}-{discount_factor}-{learning_rate}",
                    policy_kwargs=policy_args, gamma=discount_factor, learning_rate=learning_rate, batch_size=200, seed = seed, exploration_fraction=0.3, )

    a = agent.learn(total_timesteps=total_timesteps_f, callback=callback)





k_paths = 5
number_options=7
seed = 18


if __name__ == '__main__':
 processes = []
 total_timesteps = 1000000

 # topologies = [ 'Germany50', 'Coronet']
 topologies =['nsfnet_chen_eon', ]
 loads = [80, ]

 # for topology_name in ['nsfnet_chen_eon', 'Germany50', 'Coronet']:
 for topology_name, load in zip(topologies, loads):
     with open(f'examples/topologies/{topology_name}_{k_paths}-paths.h5', 'rb') as f:
         topology = pickle.load(f)
     # number_layes = [5]
     # entropy_coefs = [0.1]
     # number_neorons = [384]
     # discount_factors = [0.96]
     # learning_rates = [10e-4]


     lists = []
     for penalty_cycle, penalty_movement, algorithm  in [(-0.3,-0.05,2) ]:
     # for pc,pm,al,nl,ec,nn,df,lr in product(penalty_cycles
     #     for algorithm in [1,2]:
         for number_layer in [5,]:
             for ent_coef in [0.2,]:
                 for number_neorons in [384]:
                     for discount_factor in [ 0.96,]:
                         for learning_rate in [1e-4]:
                             for fragment_cons in [False,]:
                                 for only_FF in [True]:

                                     env_args = dict(topology=topology, seed=10, load=load, num_spectrum_resources=320,
                                                     allow_rejection=False,  # the agent cannot proactively reject a request
                                                     mean_service_holding_time=25,
                                                     # value is not set as in the paper to achieve comparable reward values
                                                     episode_length=200,
                                                     rmsa_function=shortest_available_path_first_fit,
                                                     number_options=number_options,
                                                     penalty_cycle=penalty_cycle,
                                                     penalty_movement=penalty_movement,
                                                     fragmented_constraint = fragment_cons,
                                                     only_FF = only_FF)
                                     log_dir = f"./results/DeepDefrag-{topology_name}-{load}-{penalty_cycle}-{penalty_movement}-{number_options}-{seed}-" \
                                         f"{algorithm}-{number_layer}-{ent_coef}-{number_neorons}-{discount_factor}-{learning_rate}-{only_FF}-{fragment_cons}/"
                                     os.makedirs(log_dir, exist_ok=True)
                                     callback = SaveOnBestTrainingRewardCallback(check_freq=100, log_dir=log_dir)

                             # lists.append([env_args, log_dir, total_timesteps,  algorithm, number_layer, ent_coef, number_neorons, discount_factor, learning_rate])
                             #
                             #
                             # with Pool(processes=15) as p:
                             #     result_pool = p.map_async(DRLDefrag, lists)
                             #     p.close
                             #
                             #     done = False
                             #     while not done:
                             #         if result_pool.ready():
                             #             done = True


                                     p = Process(target=DRLDefrag, args=(env_args, log_dir, total_timesteps, algorithm, number_layer, ent_coef,
                                                                         number_neorons, discount_factor, learning_rate, penalty_cycle, penalty_movement, callback, load))

                                     p.start()
                                     processes.append(p)

 [p.join() for p in processes]  # wait for the completion of all processes










 # lists.append([env_args, log_dir, total_timesteps,  algorithm, number_layer, ent_coef, number_neorons, discount_factor, learning_rate])
 # def DRLDefrag( list:[]):
 #
 #
 #     env = gym.make('DeepDefragmentation-v0', **list[0])
 #     env = Monitor(env, list[1] + f'training-dqn',
 #                   info_keywords=('episode_service_blocking_rate', 'service_blocking_rate',
 #                                  'reward', 'number_movements',
 #                                  'number_defragmentation_procedure', 'number_arrivals', 'bit_rate_blocking_rate',
 #                                  'number_movements_episode',
 #                                  'number_defragmentation_procedure_episode', 'service_blocked_eopisode',
 #                                  'number_options', 'existing_options'
 #                                  ))
 #     tmp_path = "./tmp/sb3_log/"
 #     os.makedirs(tmp_path, exist_ok=True)
 #     # set up logger
 #     new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
 #
 #     policy_args = dict(net_arch=list[4] * [list[6]])  # we use the elu activation functions
 #
 #     if list[3] ==2 :
 #
 #         agent = PPO(stable_baselines3.ppo.policies.MlpPolicy, env, verbose=0, tensorboard_log=f"./tb/PPO-DeepDefrag-v0/{load}-{penalty_cycle}-{penalty_movement}-{algorithm}-{number_layer}-{ent_coef}-{number_neorons}-{discount_factor}-{learning_rate}",
 #                     policy_kwargs=policy_args,
 #                     gamma=list[7], learning_rate=list[8], ent_coef=list[5], )
 #     elif list[3]==1:
 #         agent = DQN(stable_baselines3.dqn.policies.MlpPolicy, env, verbose=0, tensorboard_log=f"./tb/DQN-DeepDefrag-v0/{load}-{penalty_cycle}-{penalty_movement}-{algorithm}-{number_layer}-{ent_coef}-{number_neorons}-{discount_factor}-{learning_rate}",
 #                     policy_kwargs=policy_args, gamma=list[7], learning_rate=list[8], batch_size=200, seed = seed, exploration_fraction=0.3, )
 #
 #     a = agent.learn(total_timesteps=list[2], callback=callback)


