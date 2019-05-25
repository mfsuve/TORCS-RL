import os
import time


class SAC_args:
    def __init__(self):
        self.max_eps = 100000
        self.max_eps_time = 1000
        self.gamma = 0.985
        self.soft_tau = 0.01
        self.lr = 0.001
        self.alpha = 0.2
        self.buffer_size = 300000
        self.batch_size = 128
        self.device = 'cpu'
        self.test_rate = 5
        self.test_per = 50
        self.save_per = 500
        self.clipgrad = True
        self.start_sigma = 0.9
        self.end_sigma = 0.1
        self.theta = 0.15


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


def log(msg):
    with open('logger/logs.txt', 'a+') as log_file:
        time_str = time.strftime('%d-%b-%y %H:%M:%S', time.localtime())
        print(f'{time_str}\t{msg}', file=log_file)


def remove_log_file():
    if os.path.exists('logger/logs.txt'):
        os.remove('logger/logs.txt')


def make_sure_dir_exists(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir)
