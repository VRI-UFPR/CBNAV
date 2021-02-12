import gym
import numpy as np

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
N_STEPS = 100_000

class DQNAgentProc(Processor):
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

env = Environment("Simulation2d/svg/proto_1_1", 1)
env.use_observation_rotation_size(True)
env.set_observation_rotation_size(128)
env.set_mode(Mode.ALL_RANDOM)

processor = DQNAgentProc()
states = env.observation_size()
actions = action_mapper.ACTION_SIZE

if DEBUG:
	print('states: {0}'.format(states))
	print('actions: {0}'.format(actions))

def build_callbacks(env_name):
    weights_filename = 'new_results/'+ env_name +'{step}.h5f'
    log_filename = 'new_log/{}.json'.format(env_name)
    callbacks = [ModelIntervalCheckpoint(weights_filename, interval=10000)]
    callbacks += [FileLogger(log_filename, interval=1000)]
    return callbacks

def build_model(states, actions):
	model = Sequential()
	model.add(Dense(2048, input_dim=states, activation='relu'))
	model.add(Dense(512, activation='relu'))
	model.add(Dense(256, activation='relu'))
	model.add(Dense(actions, activation='linear'))
	model.summary()
	return model

def build_agent(model, actions):
	policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), 
		attr='eps', value_max=1., 
		value_min=.05, value_test=.05, 
		nb_steps=80_000)
	#policy = EpsGreedyQPolicy(eps=.05)
	#policy = GreedyQPolicy()

	#policy = BoltzmannQPolicy()
	memory = SequentialMemory(limit=30000, window_length=1)
	dqn = DQNAgent(model=model, memory=memory, policy=policy, processor=processor,
		nb_actions=actions, nb_steps_warmup=500, target_model_update=1e-3,
		enable_double_dqn=True,
		enable_dueling_network=True, dueling_type='avg',
		batch_size=64, gamma=.95)
	return dqn

model = build_model(states, actions)
dqn = build_agent(model, actions)
callbacks = build_callbacks('t_1')

dqn.compile(Adam(lr=1e-3), metrics=['mse'])
dqn.fit(env, nb_steps=N_STEPS, visualize=False, verbose=2, callbacks=callbacks)
scores = dqn.test(env, nb_episodes=10, visualize=True)
print (np.mean(scores.history['episode_reward']))

