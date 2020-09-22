
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
eval_interval: int = 1000
num_iterations: int = eval_interval*10 # how long to train for. I recoment this be greater than eval_interval * 3
initial_collect_steps: int = 1000 
collect_steps_per_iteration: int = 1
replay_buffer_max_length: int = 100000 
batch_size: int = 64 
learning_rate: float = 1e-3
log_interval: int = 200
num_eval_episodes: int = 10
env_name: str = 'CountUp-v0'
#env_name: str = 'CustomCartPole-v0'
model_number: str = '1600745505.6067228' 

# 3) Setup & verify ############################################################################################################
checkpoint_dir = "output/"+env_name+"/models/"+model_number+"/checkpoint/"
policy_dir = "output/"+env_name+"/models/"+model_number+"/policy/"



# 4) Setup tensorflow ############################################################################################################
tf.compat.v1.enable_v2_behavior()

train_py_env = suite_gym.load(env_name)
eval_py_env = suite_gym.load(env_name)
train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

# 4.1) Define the network ############################################################################################################
fc_layer_params = (100,)
q_net = q_network.QNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    fc_layer_params=fc_layer_params)

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

train_step_counter = tf.Variable(0)

# 4.2) Define the agent ############################################################################################################
agent = dqn_agent.DqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step_counter)

agent.initialize()

eval_policy = agent.policy
collect_policy = agent.collect_policy

# 4.3) Define random agent ############################################################################################################
random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),train_env.action_spec())
example_environment = tf_py_environment.TFPyEnvironment(suite_gym.load(env_name))
time_step = example_environment.reset()
random_policy.action(time_step)

def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0
    for _ in range(num_episodes):
        t_s = environment.reset()
        episode_return = 0.0
        while not t_s.is_last():
            action_step = policy.action(t_s)
            t_s = environment.step(action_step.action)
            episode_return += t_s.reward
            #if t_s.reward > 0:
                #print('eval reward is greater than 0')
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]

print('compute_avg_return',compute_avg_return(eval_env, random_policy, num_eval_episodes))

# 4.4) Setup saving the model ############################################################################################################
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=replay_buffer_max_length)

train_checkpointer = common.Checkpointer(
    ckpt_dir=checkpoint_dir,
    max_to_keep=1,
    agent=agent,
    policy=agent.policy,
    replay_buffer=replay_buffer,
    global_step=train_step_counter
)
tf_policy_saver = policy_saver.PolicySaver(agent.policy)

# 4.5) Setup loading the model ############################################################################################################
train_checkpointer.initialize_or_restore()
train_step_counter = tf.compat.v1.train.get_global_step()
try:
    saved_policy = tf.compat.v2.saved_model.load(policy_dir)
except:
    pass

# 4.6) helpful functions ############################################################################################################
def collect_step(environment, policy, buffer):
    t_s = environment.current_time_step()
    action_step = policy.action(t_s)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(t_s, action_step, next_time_step)

    # Add trajectory to the replay buffer
    buffer.add_batch(traj)

def collect_data(env, policy, buffer, steps):
    for _ in range(steps):
        collect_step(env, policy, buffer)

collect_data(train_env, random_policy, replay_buffer, steps=100)

# 4.7) define dataset ############################################################################################################
dataset = replay_buffer.as_dataset(
    num_parallel_calls=3, 
    sample_batch_size=batch_size, 
    num_steps=2).prefetch(3)

iterator = iter(dataset)

# 4.8) Setup Training #########################################################################################################
# (Optional) Optimize by wrapping some of the code in a graph using TF function.
agent.train = common.function(agent.train)

# Reset the train step
agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
print('agent eval avg return',avg_return)
returns = [avg_return]


def create_policy_eval_video(policy):
    t_s = eval_env.reset()
    eval_py_env.render()
    while not t_s.is_last():
        action_step = policy.action(t_s)
        t_s = eval_env.step(action_step.action)
        eval_py_env.render()
    eval_py_env.close()
    

# 6) Capture data #########################################################################################################
create_policy_eval_video(saved_policy)
