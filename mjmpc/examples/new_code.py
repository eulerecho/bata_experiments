#!/usr/bin/env python
import sys
import os
import os.path as osp
sys.path.insert(0, os.path.abspath('..'))
import argparse
from copy import deepcopy
from datetime import datetime
import gym
from itertools import product
import json
import numpy as np
import pickle
import tqdm
import yaml
os.environ['KMP_DUPLICATE_LIB_OK']='True'
try:
    import mj_envs
except ImportError:
    print('mj_envs not found. Will not be able to run its configs')
import mjmpc.envs
from mjmpc.envs import GymEnvWrapper
from mjmpc.envs.vec_env import SubprocVecEnv
from mjmpc.utils import LoggerClass, timeit, helpers
from mjmpc.policies import MPCPolicy

gym.logger.set_level(40)
parser = argparse.ArgumentParser(description='Run MPC algorithm on given environment')
parser.add_argument('--config', type=str, help='yaml file with experiment parameters')
parser.add_argument('--dyn_randomize_config', type=str, help='yaml file with dynamics randomization parameters')
parser.add_argument('--save_dir', type=str, default='/tmp', help='folder to save data in')
parser.add_argument('--controller', type=str, default='mppi', help='controller to run')
parser.add_argument('--dump_vids', action='store_true', help='flag to dump video of episodes' )
parser.add_argument('--do_extra', action='store_true')
args = parser.parse_args()

#Load experiment parameters from config file
with open(args.config) as file:
    exp_params = yaml.load(file, Loader=yaml.FullLoader)
if args.dyn_randomize_config is not None:
    with open(args.dyn_randomize_config) as file:
        dynamics_rand_params = yaml.load(file, Loader=yaml.FullLoader)
else:
    dynamics_rand_params=None

controller_name = "mppi"
#Create the main environment
env_name  = "pretouch_mjrl-v0"
env = gym.make(env_name)
env = GymEnvWrapper(env)
env.real_env_step(False)

sim_env_name = env_name


# Function to create vectorized environments for controller simulations
def make_env():
    gym_env = gym.make(sim_env_name)
    rollout_env = GymEnvWrapper(gym_env)
    rollout_env.real_env_step(False)

    return rollout_env


#unpack params and create policy params
policy_params = exp_params[controller_name]
policy_params['d_obs'] = env.d_obs
policy_params['d_state'] = env.d_state
policy_params['d_action'] = env.d_action
policy_params['action_lows'] = env.action_space.low
policy_params['action_highs'] = env.action_space.high
print(policy_params)
if 'num_cpu' and 'particles_per_cpu' in policy_params:
    policy_params['num_particles'] = policy_params['num_cpu'] * policy_params['particles_per_cpu']

num_cpu = policy_params['num_cpu']
n_episodes = exp_params['n_episodes']
base_seed = exp_params['seed']
ep_length = exp_params['max_ep_length']

policy_params.pop('particles_per_cpu', None)
policy_params.pop('num_cpu', None)


#Create vectorized environments for MPC simulations
sim_env = SubprocVecEnv([make_env for i in range(num_cpu)])


def rollout_fn(num_particles, horizon, mean, noise, mode):
    """
    Given a batch of sequences of actions, rollout
    in sim envs and return sequence of costs. The controller is
    agnostic of how the rollouts are generated.
    """
    obs_vec, rew_vec, act_vec, done_vec, info_vec, next_obs_vec = sim_env.rollout(num_particles,
                                                                                  horizon,
                                                                                  mean.copy(),
                                                                                  noise,
                                                                                  mode)
    #we assume environment returns rewards, but controller needs costs
    sim_trajs = dict(
        observations=obs_vec.copy(),
        actions=act_vec.copy(),
        costs=-1.0*rew_vec.copy(),
        dones=done_vec.copy(),
        next_observations=next_obs_vec.copy(),
        infos=helpers.stack_tensor_dict_list(info_vec.copy())
    )

    return sim_trajs


#seeding to enforce consistent episodes
episode_seed = base_seed
policy_params['seed'] = episode_seed
obs = env.reset(seed=episode_seed)
sim_env.reset()

#create MPC policy and set appropriate functions
policy = MPCPolicy(controller_type=controller_name,
                    param_dict=policy_params, batch_size=1) #Only batch_size=1 is supported for now
policy.controller.set_sim_state_fn = sim_env.set_env_state
policy.controller.rollout_fn = rollout_fn
action=np.array([0,0])

for i in range(ep_length):

    curr_state = deepcopy(env.get_env_state())

    if (i%5)==0:
        action, value = policy.get_action(curr_state, calc_val=False)
        print(action)

    obs, reward, done, info = env.step(action)
    env.render()

    if done:
        break

sim_env.close() #Free up memory

env.close()
