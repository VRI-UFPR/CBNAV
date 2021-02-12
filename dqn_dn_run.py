import gym
import numpy as np
import sys

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
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
N_STEPS = 200_000

class DQNAgentProc(Processor):
	def __init__(self):
		print ('**********__init__**********')
		self.gewonnen = 0

	def get_gewonnen(self):
		return self.gewonnen
	
	def set_gewonnen(self, gewonnen):
		self.gewonnen = gewonnen;
	

	def process_step(self, observation, reward, done, info):
		if reward == 50 or reward == 20:
			self.gewonnen += 1
			print ('Wir haben gewonnen: {0} zeit'.format(self.gewonnen ))
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

def build_name(env_name):
    weights_filename = 'new_results/'+ env_name + str(N_STEPS) +'.h5f'
    return weights_filename

def build_model(states, actions):
	model = Sequential()
	model.add(Dense(2048, input_dim=states, activation='relu'))
	model.add(Dense(512, activation='relu'))
	model.add(Dense(256, activation='relu'))
	model.add(Dense(actions, activation='linear'))
	model.summary()
	return model

def build_agent(model, actions, processor):
	policy = GreedyQPolicy()
	memory = SequentialMemory(limit=30000, window_length=1)
	dqn = DQNAgent(model=model, memory=memory, policy=policy, processor=processor,
		nb_actions=actions, nb_steps_warmup=500, target_model_update=1e-3,
		enable_double_dqn=True,
		enable_dueling_network=True, dueling_type='avg',
		batch_size=64, gamma=.95)
	return dqn

def main():

    print("python dqn_dq_run.py [Landkarte] [Keras model]")

    if len(sys.argv) == 4:
    	landkarte = str(sys.argv[1])
    	kerasmodel = str(sys.argv[2])
    	model_number = sys.argv[3]

    	if DEBUG:
    		print ('Landkarte: {0}'.format(landkarte))
    		print ('Keras Model: {0}'.format(kerasmodel))

    	env = Environment("Simulation2d/svg/"+landkarte, int(model_number))
    	env.use_observation_rotation_size(True)
    	env.set_observation_rotation_size(128)
    	env.set_mode(Mode.ALL_COMBINATION)

    	processor = DQNAgentProc()
    	states = env.observation_size()
    	actions = action_mapper.ACTION_SIZE

    	if DEBUG:
    		print('states: {0}'.format(states))
    		print('actions: {0}'.format(actions))

    	model = build_model(states, actions)
    	dqn = build_agent(model, actions, processor)
    	#name = build_name('dqn_dn_boltzmann_room')
    	name = build_name(kerasmodel)

    	dqn.compile(Adam(lr=1e-3), metrics=['mse'])
    	dqn.load_weights(name)

    	scores = dqn.test(env, nb_episodes=1000, visualize=False )
    	#print (np.mean(scores.history['episode_reward']))
    	print ('Wir haben gewonnen: {0} zeit'.format(processor.get_gewonnen()))

    else:
    	print ('Das ist nicht richtig. Bitte informiere Sie die Landkarte und die Keras model. ')
    	exit(0)


if __name__ == "__main__":
    main()
