
# 1) import everything ############################################################################################################
from __future__ import absolute_import, division, print_function

import base64
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
#import PIL.Image
import time
import shutil
from pathlib import Path

import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.policies import policy_saver
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

import custom_openai_frameworks

# 2) define our variables ############################################################################################################
env_name: str = 'GolfCardGame-v0'
model_number: str = '1600901469.939331' 

# 3) Setup & verify ############################################################################################################
policy_dir = "output/"+env_name+"/models/"+model_number+"/policy/"

# 4) Setup tensorflow ############################################################################################################
tf.compat.v1.enable_v2_behavior()

# 4) Eval with render func ############################################################################################################
eval_py_env = suite_gym.load(env_name)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

saved_policy = tf.compat.v2.saved_model.load(policy_dir)

t_s = eval_env.reset()
eval_py_env.render()
while not t_s.is_last():
    action_step = saved_policy.action(t_s)
    t_s = eval_env.step(action_step.action)
    eval_py_env.render()
eval_py_env.close()

