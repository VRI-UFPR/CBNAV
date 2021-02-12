# -*- coding: utf-8 -*-

import numpy as np
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy, LinearAnnealedPolicy, SoftmaxPolicy, EpsGreedyQPolicy, GreedyQPolicy
from rl.memory import SequentialMemory, RingBuffer, EpisodeParameterMemory
from rl.core import Processor
from rl.core import Env
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

from environment.environment import Environment
from environment.environment_node_data import Mode

from keras.callbacks import Callback 
from keras.callbacks import CallbackList as KerasCallbackList

'''
Mapeamento dos dados discretos da ação
'''
import action_mapper 

DEBUG = True

class ManualProc(Processor):
	def __init__(self):
		print ('**********__init__**********')

	def process_step(self, observation, reward, done, info):
		return observation, reward, done, info

	def process_observation(self, observation):
		obs = observation[0]
		return obs

	def process_reward(self, reward):
		return reward

	def process_info(self, info):
		return info

	def process_action(self, action):
		return action

	def process_state_batch(self, batch):
		return batch[:, 0, :]

env = Environment("Simulation2d/svg/proto_1", 6)
env.use_observation_rotation_size(True)
env.set_observation_rotation_size(128)
env.set_mode(Mode.ALL_RANDOM)

processor = ManualProc()
states = env.observation_size()
actions = action_mapper.ACTION_SIZE

if DEBUG:
	print('states: {0}'.format(states))
	print('actions: {0}'.format(actions))

state, reward, done, _ = env.reset()
env.render()

done = False

while not done:
	value = input("Próxima ação: [0 - 6]: \n")
	action = int(value)
	state, reward, done, _ = env.step(action)
	#print ('reward: {0}'.format(reward))
	env.render()
