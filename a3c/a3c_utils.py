from torch.multiprocessing import Lock, Value
import torch

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
        self.lr = 0.0001

class TrajectoryList:
    def __init__(self, values=[], rewards=[], log_probs=[], entropies=[]):
        self.values = values
        self.rewards = rewards
        self.log_probs = log_probs
        self.entropies = entropies

    def append(value, reward, log_prob, entropy):
        self.values.append(value)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.entropies.append(entropy)

    def __len__(self):
        return len(self.rewards)
