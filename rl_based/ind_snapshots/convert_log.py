import visdom
import numpy as np
import time

#action_file = 'rewad.txt'
log_file = 'log.txt'

while True:
	file = open(log_file,'r')

	file_lines = file.readlines()

	length_log = len(file_lines)
	reward1 = np.zeros(length_log)
	reward2 = np.zeros(length_log)
	win_rate = np.zeros(length_log)
	j = 0
	win_num = 0
	for line2 in file_lines:
		lines = line2.split(' ')
		reward1[j] = float(lines[1])
		reward2[j] = float(lines[2])

		if reward2[j] == 100:
			win_num += 1
		win_rate[j] = float(win_num) / (j+1)
		with open(os.path.join('', 'winrate_ind'), 'a+') as f:
			f.write('{} {}\n'.format(j, win_rate[j]))
		#print(win_rate[j])
		j += 1

	break

