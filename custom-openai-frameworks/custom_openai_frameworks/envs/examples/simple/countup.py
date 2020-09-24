import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import random


class CountUpEnv(gym.Env):

    def __init__(self):
        self.reward = None
        self.wrong_times = None
        self.mynumber = None
        self.state = None
        self.steps_beyond_done = None

        self.outputs: list(int) = [
            0,
            1,
            2,
            3
        ]
        self.inputs_low: list(float) = [self.outputs[0]]
        self.inputs_high: list(float) = [self.outputs[-1]]
        self.num_obervations: int = len(self.inputs_low)
        inputs_np_high = np.array(self.inputs_high, dtype=np.float32)
        inputs_np_low = np.array(self.inputs_low, dtype=np.float32)
        self.observation_space: spaces.Box = spaces.Box(inputs_np_low, inputs_np_high, dtype=np.float32)
        self.action_space: spaces.Discrete = spaces.Discrete(len(self.outputs))
        self.seed()

        self.is_inputing = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg
        
        if self.is_inputing is True:
            print('I think the next number should be',action)

        # 1) Set State
        if action == self.mynumber + 1:
            self.reward = 1
        elif action == 0 and self.mynumber == 3:
            self.reward = 1
        else:
            self.reward = 0
            self.wrong_times += 1

        self.mynumber = action
        if self.is_inputing is True:
            try:
                answer = input('What is your number?')
                self.mynumber = int(answer)
                print('Ok your number is',self.mynumber)
            except:
                pass

        self.state = (
            self.mynumber,
        )

        # 2) Get Done
        done = self.wrong_times > 4

        # 3) Get reward
        if not done:
            reward = self.reward
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = self.reward
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.mynumber = random.randint(self.outputs[0],self.outputs[-1])
        if self.is_inputing is True:
            answer = input('What is your number?')
            self.mynumber = int(answer)
        self.wrong_times = 0
        self.state = (
            self.mynumber,
        )
        self.reward = None
        self.steps_beyond_done = None
        return np.array(self.state)

    def render(self, mode='human'):

        if self.is_inputing is None:
            print('Hello')
            answer = input('Do you want to input? (y/n)')
            if answer == 'y':
                print('yay')
                self.is_inputing = True
            else:
                self.is_inputing = False

        if self.state is None:
            return None

        return None # this might cause problems

    def close(self):
        if self.is_inputing:
            print('bye')