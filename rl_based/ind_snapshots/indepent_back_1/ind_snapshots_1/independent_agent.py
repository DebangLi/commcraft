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

DISTANCE_FACTOR = 16
vis = visdom.Visdom(env='ind', port=8099)
log_actions=[]
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
		self.affine1 = nn.Linear(68, 128)
		self.affine2 = nn.Linear(128,128)
		self.affine3 = nn.Linear(128,13)
		self.saved_actions = []
		self.rewards = []
		self.saved_probs = []

	def forward(self, x):
		x = F.relu(self.affine1(x))
		x = F.relu(self.affine2(x))
		action = self.affine3(x)
		return F.softmax(action)

model = Policy()
if args.model != '':
	model.load_state_dict(torch.load(args.model))
	print('loading model...')
	print(model)
model.cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-2)

def select_action(state):
	state = torch.from_numpy(state).float().unsqueeze(0)
	probs = model(Variable(state.cuda()))
	model.saved_probs.append(probs)
	action = probs.multinomial()
	#log_actions.append(action.data[0,0])
	#vis.line(np.array(log_actions), win=0)
	log = open('ind_snapshots/log_action.txt','a')
	log.write(str(action.data[0,0])+'\n')
	log.close()
	model.saved_actions.append(action)
	return action.data

def finish_episode(episodes):
	R = 0
	rewards = []
	# get the accumlated reward
	for r in model.rewards[::-1]:
		R = r + args.gamma * R
		rewards.insert(0,R)
	rewards = torch.Tensor(rewards)
	rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
	for action, r in zip(model.saved_actions, rewards):
		action.reinforce(r)
	#optimizer.zero_grad()
	autograd.backward(model.saved_actions, [None for _ in model.saved_actions])
	#optimizer.step()
	if episodes % 16 == 0:
		optimizer.step()
		optimizer.zero_grad()
	del model.rewards[:]
	del model.saved_actions[:]
	del model.saved_probs[:]

def supervised():

	loss = 0
	print('supervised learning.......')
	criterion = nn.MSELoss()
	criterion.cuda()
	label = torch.zeros([1,13])
	label[0,0] = 1
	label = Variable(label)
	label = label.cuda()
	optimizer.zero_grad()
	for prob in model.saved_probs:
	   loss += criterion(prob, label)
	loss.backward()
	optimizer.step()
	del model.rewards[:]
	del model.saved_actions[:]
	del model.saved_probs[:]

if __name__ == '__main__':

	env = sc.MultiAgentEnv(args.ip, args.port, speed=30)
	env.seed(123)
	torch.manual_seed(123)

	agent = IndependentAgents(env.action_space)
	#vis = visdom.Visdom(env='Independent')

	episodes = args.eposides
	start = time.time()
	log_reward_sum = []
	log_reward = []
	while True:
		obs = env.reset()
		'''
		done = False
		while not done:
			action = agent.act(obs)
			obs, reward, done, info = env.step(action)
		episodes += 1
		'''
		reward_sum = 0
		for t in range(500):
			action = agent.act(obs)
			obs, reward, done, _ = env.step(action)
			model.rewards.append(reward)
			reward_sum += reward
			if done:
				break
		episodes += 1
		if episodes < 50 and args.sl:
			supervised()
			optimizer.zero_grad()
		else:
			finish_episode(episodes)
		log = open('ind_snapshots/log.txt','a')
		log.write(str(episodes)+' '+str(reward_sum)+' '+str(reward)+'\n')
		log.close()
		#log_reward_sum.append(reward_sum)
		#log_reward.append(reward)
		#vis.line(np.array(log_reward_sum), win=1)
		#vis.line(np.array(log_reward), win=2)
		if episodes % args.log_interval == 0:
		#if episodes % 1 == 0:
			print('Episodes {}\tCost time: {:4f}\tThe sum of reward: {:5d}\t The last reward: {:5d}'.format(episodes, time.time()-start, reward_sum, reward))
			start = time.time()
		if episodes % 500 == 0:
			torch.save(model.state_dict(),'./ind_snapshots/episodes_%d.pth' % (episodes))
		if episodes % 100000 == 0:
			break

	env.close()
