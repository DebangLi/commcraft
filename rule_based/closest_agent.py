import argparse
import numpy as np
import multi_agent_env as sc
import utils


class ClosetAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, obs):
        nagent = len(obs) - np.sum(obs, axis = 0, dtype = np.int32)[5]
        action = np.zeros([nagent, self.action_space.shape[0]])
        kill_all = True
        n = 0
        for i in range(len(obs)):
            if obs[i][5] == 0:
                distance = 1e6
                target_id = -1
                for j in range(len(obs)):
                    if obs[j][5] == 1:
                        dis = utils.get_distance(obs[i][6], obs[i][7], obs[j][6], obs[j][7])
                        if dis < distance:
                            target_id = obs[j][0]
                            distance = dis
                if target_id == -1:
                    return None

                action[n][0] = obs[i][0]
                action[n][1] = 1
                action[n][4] = target_id
                n = n + 1

        return action


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', help='server ip')
    parser.add_argument('--port', help='server port', default="11111")
    args = parser.parse_args()

    env = sc.MultiAgentEnv(args.ip, args.port, speed=30)
    env.seed(123)
    agent = ClosetAgent(env.action_space)

    episodes = 0
    while True:
        obs = env.reset()
        done = False
        while not done:
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
        episodes += 1

    env.close()
