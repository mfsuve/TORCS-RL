from torch.multiprocessing import Lock, Value
import torch
from network import A3C_Network
import numpy as np
import logging
import logging.handlers
import os

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
        self.max_eps_time = 300
        self.gamma = 0.99
        self.beta = 0.01
        self.lr = 0.001
        self.num_workers = 8
        self.plot_rate = 30

class Recorder:
    def __init__(self, manager):
        self.rewards = manager.list()
        self.best_rewards = manager.list()
        self.best_net = A3C_Network(29, 3)
        self.best_net.share_memory()
        self.time_steps = manager.list()

class Logger:
    def __init__(self):
        self.logger = logging.getLogger('a3c_logger')
        self.logger.setLevel(logging.INFO)
        server = '127.0.0.1:3000'
        path = '/'
        method = 'POST'
        handler = logging.handlers.HTTPHandler(server, path, method)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        try:
            os.remove('logger/logs.txt')
        except FileNotFoundError:
            pass

    def log(self, msg):
        self.logger.info(msg)

def normal(x, mu, sigma):
    pi = np.array([np.pi])
    pi = torch.from_numpy(pi).float()
    a = (-1 * (x - mu).pow(2) / (2 * sigma)).exp()
    b = 1 / (2 * sigma * pi.expand_as(sigma)).sqrt()
    return a * b
