import visdom
import numpy as np
import time

action_file = 'rewad.txt'
log_file = 'log.txt'

while True:
	file_action = open(action_file,'r')
	file = open(log_file,'r')

	file_action_lines = file_action.readlines()
	file_lines = file.readlines()

	length_action = len(file_action_lines)
	actions = np.zeros(length_action)
	rewards = np.zeros(length_action)
	i = 0
	for line in file_action_lines:
		if i > 0:
			lines_split = line.split(' ')
			actions[i] = np.float(lines_split[0])
			rewards[i] = np.float(lines_split[1])
		i += 1
	length_log = len(file_lines)
	reward1 = np.zeros(length_log)
	reward2 = np.zeros(length_log)
	win_rate = np.zeros(length_log)
	x = np.zeros(length_log)
	j = 0
	win_num = 0
	for line2 in file_lines:
		lines = line2.split(' ')
		reward1[j] = float(lines[1])
		reward2[j] = float(lines[2])
		#print('--------------------------------------')
		#print(reward2[j])
		#print(win_num)

		if reward2[j] == 1000:
			win_num += 1
		x[j] = j
		win_rate[j] = float(win_num) / (j+1)
		#print(win_rate[j])
		j += 1

	vis = visdom.Visdom(env='ind2', port=8097)
	vis.line(actions, win=0)
	#print(win_rate)
	vis.line(reward1, win=1)
	vis.line(reward2, win=2)
	vis.line(win_rate,X=x, win=3)
	vis.line(rewards, win=4)
	print('done...')
	time.sleep(5)

