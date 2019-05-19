from torch.multiprocessing import Lock, Value
import torch
import numpy as np

class Counter(object):
    def __init__(self, val=0):
        self.val = Value('i', val)
        self.lock = Lock()

    def increment(self):
        with self.lock:
            self.val.value += 1

    def value(self):
        with self.lock:
            return self.val.value

class A3C_args:
    def __init__(self):
        self.max_time = 1000000
        self.nstep = 20
        self.max_eps_time = 2000
        self.gamma = 0.97
        self.beta = 0.01
        self.lr = 0.0001
        self.num_workers = 1#8

class TrajectoryList:
    def __init__(self):
        self.values = []
        self.rewards = []
        self.log_probs = []
        self.entropies = []

    def append(self, value, reward, log_prob, entropy):
        self.values.append(value)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.entropies.append(entropy)

def normal(x, mu, sigma):
    pi = np.array([np.pi])
    pi = torch.from_numpy(pi).float()
    a = (-1 * (x - mu).pow(2) / (2 * sigma)).exp()
    b = 1 / (2 * sigma * pi.expand_as(sigma)).sqrt()
    return a * b
