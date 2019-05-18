from gym_torcs import TorcsEnv
from mp_utils import Counter
import torch.multiprocessing as mp
import torch
import numpy as np

class Worker(mp.Process):
    def __init__(self, rank, global_net, counter, max_time):
        super(Worker, self).__init__()
        self.rank = rank
        self.global_net = global_net
        self.counter = counter
        self.max_time = max_time

    def run(self):
        eps_n = 0
        while self.counter.value() < self.max_time:
            pass



env = TorcsEnv(path='/usr/local/share/games/torcs/config/raceman/quickrace.xml')

state = env.reset(relaunch=True, sampletrack=True, render=False)
print('State:', state)
print('Angle        :', state[0])
print('Track        :', state[1:20])
print('TrackPos     :', state[20])
print('Speed        :', state[21:24])
print('WheelSpeeds  :', state[24:28])
print('rpm          :', state[28])
eps_reward = 0
eps_n = 1
for i in range(500):
    state, reward, done, _ = env.step([0., 0., 0.])
    eps_reward += reward
    if done:
        print('Episode', eps_n, 'Reward:', eps_reward)
        state = env.reset(relaunch=eps_n%5==0, sampletrack=True, render=False)
        eps_reward = 0
        eps_n += 1
