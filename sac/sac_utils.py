import os
import time
import torch


class SAC_args:
    def __init__(self):
        self.max_eps = 3000
        self.max_eps_time = 1000
        self.gamma = 0.985
        self.soft_tau = 0.01
        self.lr = 0.001
        self.alpha = 0.2
        self.buffer_size = 300000
        self.batch_size = 128
        self.device = 'cpu'
        self.test_rate = 5
        self.plot_per = 20
        self.clipgrad = True
        self.start_sigma = 0.9
        self.end_sigma = 0.1
        self.theta = 0.15
        self.change_track_per = 100


class OrnsteinUhlenbeckProcess(object):

    def __init__(self, theta, sigma, dim, mu=0):
        self.theta = theta
        self.sigma = sigma
        self.mu = mu
        self.state = mu
        self.dim = dim

    def noise(self):
        v = self.theta * (self.mu - self.state) + self.sigma * torch.randn(self.dim)
        self.state += v
        return self.state

    def reset(self):
        self.state = self.mu


class Checkpoint(object):
    def __init__(self, folder=None):
        self.state = {
            'best_value': None,
            'best_q1': None,
            'best_q2': None,
            'best_policy': None
        }
        if folder is None:
            self.folder = '.'
        else:
            self.folder = folder

    def save(self, filename):
        path = os.path.join(self.folder, filename)
        torch.save(self.state, path)

    def update(self, best_value, best_q1, best_q2, best_policy):
        self.state['best_value'] = best_value.state_dict()
        self.state['best_q1'] = best_q1.state_dict()
        self.state['best_q2'] = best_q2.state_dict()
        self.state['best_policy'] = best_policy.state_dict()


def log(*msg, end=None):
    with open('logger/logs.txt', 'a+') as log_file:
        time_str = time.strftime('%d-%b-%y %H:%M:%S', time.localtime())
        if end is None:
            print(f'{time_str}\t', *msg, file=log_file)
        else:
            print(f'{time_str}\t', *msg, file=log_file, end=end)


def remove_log_file():
    if os.path.exists('logger/logs.txt'):
        os.remove('logger/logs.txt')


def make_sure_dir_exists(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir)
