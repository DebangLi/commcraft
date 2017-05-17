import argparse
import numpy as np
import commcraft.multi_agent_env as sc


class WeakestAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, obs):
        nagent = len(obs) - np.sum(obs, axis = 0)[5]
        action = np.zeros([nagent, self.action_space.shape])
        tmp_hp = 1000000
        for i in len(obs):
            hp = obs[i][1] + obs[i][2]
            if obs[i][5] != 0 and hp < tmp_hp:
                target_id = obs[i][0]
                tmp_hp = hp
        n = 0 
        for i in len(obs):
            if obs[i][5] == 0:
                action[n][0] = obs[i][0]
                action[n][1] = 1
                action[n][4] = target_id

        return action


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', help='server ip')
    parser.add_argument('--port', help='server port', default="11111")
    args = parser.parse_args()

    env = sc.MultiAgentEnv(args.ip, args.port, speed=30)
    env.seed(123)
    agent = WeakestAgent(env.action_space)

    episodes = 0
    while True:
        obs = env.reset()
        done = False
        while not done:
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
        episodes += 1

    env.close()
