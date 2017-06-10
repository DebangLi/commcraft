import argparse
import time
import numpy as np
import multi_agent_env as sc
import utils
from itertools import count
from collections import namedtuple
import visdom

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

parser = argparse.ArgumentParser(description='Independent agent to fight')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
					help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
					help='random seed (default: 543)')
parser.add_argument('--log_interval', type=int, default=10, metavar='N',
					help='interval between training status logs (default: 10)')
parser.add_argument('--ip', help='server ip')
parser.add_argument('--port', help='server port', default="11111")
parser.add_argument('--model', default='', help="path to model (to continue training)")
parser.add_argument('--sl', default= False, help='started with supervised learning')
parser.add_argument('--eposides', type=int, default=0, help='start from which eposides')
args = parser.parse_args()
print(args)

DISTANCE_FACTOR = 16
class IndependentAgents(object):
	def __init__(self, action_space):
		self.action_space = action_space

	def act(self, obs):
		nagent = len(obs) - np.sum(obs, axis = 0, dtype = np.int32)[5]
		nenemy = len(obs) - nagent
		
		action = np.zeros([nagent, self.action_space.shape[0]])
		if nenemy == 0:
			return None

		n = 0
		for i in range(len(obs)):
			if obs[i][5] == 0:
				# initilize the input to the model 5+7*9 = 68
				input = np.zeros(68)
				enemy_table = -np.ones(5)
				input[0] = obs[i][1]
				input[1] = obs[i][2]
				input[2] = obs[i][3]
				input[3] = obs[i][4]
				input[4] = obs[i][5]
				k = 5
				ind_enemy = 0
				for j in range(len(obs)):
					if j != i:
						dis = utils.get_distance(obs[i][6], -obs[i][7], obs[j][6], -obs[j][7]) / DISTANCE_FACTOR - 1
						degree = utils.get_degree(obs[i][6], -obs[i][7], obs[j][6], -obs[j][7]) / 180
						input[k] = degree
						k += 1
						input[k] = dis
						k += 1
						# hp[0,100]
						input[k] = obs[j][1]
						k += 1
						# sheild[0,100]
						input[k] = obs[j][2]
						k += 1
						# cooldown[0,1]
						input[k] = obs[j][3]
						k += 1
						# fround range[0,1]
						input[k] = obs[j][4]
						k += 1
						# is enemy, 0 for myself 1 for enemy
						input[k] = obs[j][5]
						k += 1
						if obs[j][5] == 1:
							enemy_table[ind_enemy] = obs[j][0]
							ind_enemy += 1
				act = select_action(input)
		
				action[n][0] = obs[i][0]
				#action[n][1] = 1
				#action[n][2] = 0
				action[n][3] = 1
				action[n][4] = -1
				if act[0,0] == 0:
					action[n][1] = 1
					action[n][4] = enemy_table[0]
				elif act[0,0] == 1:
					action[n][1] = 1
					action[n][4] = enemy_table[1]
				elif act[0,0] == 2:
					action[n][1] = 1
					action[n][4] = enemy_table[2]
				elif act[0,0] == 3:
					action[n][1] = 1
					action[n][4] = enemy_table[3]
				elif act[0,0] == 4:
					action[n][1] = 1
					action[n][4] = enemy_table[4]
				else:
					action[n][1] = -1
					if act[0,0] == 5:
						action[n][2] = 0
					elif act[0,0] == 6:
						action[n][2] = 0.25
					elif act[0,0] == 7:
						action[n][2] = 0.5
					elif act[0,0] == 8:
						action[n][2] = 0.75
					elif act[0,0] == 9:
						action[n][2] = 1
					elif act[0,0] == 10:
						action[n][2] = -0.75
					elif act[0,0] == 11:
						action[n][2] = -0.5
					elif act[0,0] == 12:
						action[n][2] = -0.25

				n = n + 1
		for i in range(n-1):
			model.rewards.append(0)

		return action

class Policy(nn.Module):
	def __init__(self):
		super(Policy, self).__init__()
		self.affine1 = nn.Linear(68, 256)
		self.affine2 = nn.Linear(256, 512)
		self.affine3 = nn.Linear(512, 128)
		self.affine4 = nn.Linear(128,13)
		self.saved_actions = []
		self.rewards = []
		self.saved_probs = []

	def forward(self, x):
		x = F.relu(self.affine1(x))
		x = F.relu(self.affine2(x))
		x = F.relu(self.affine3(x))
		action = self.affine4(x)
		return F.softmax(action)

model = Policy()
if args.model != '':
	model.load_state_dict(torch.load(args.model))
	print('loading model...')
	print(model)
model.cuda()

def select_action(state):
	state = torch.from_numpy(state).float().unsqueeze(0)
	probs = model(Variable(state.cuda()))
	action = probs.multinomial()
	return action.data

if __name__ == '__main__':

	env = sc.MultiAgentEnv(args.ip, args.port, speed=60, max_episode_steps=2000)
	env.seed(123)
	torch.manual_seed(123)

	agent = IndependentAgents(env.action_space)

	while True:
		obs = env.reset()
		'''
		done = False
		while not done:
			action = agent.act(obs)
			obs, reward, done, info = env.step(action)
		episodes += 1
		'''
		done = False
		while not done:
			action = agent.act(obs)
			obs, reward, done, _ = env.step(action)

	env.close()
