import os
import time


class SAC_args:
    def __init__(self):
        self.max_eps = 100000
        self.max_eps_time = 500
        self.gamma = 0.99
        self.soft_tau = 1e-2
        self.lr = 3e-4
        self.alpha = 0.2
        self.buffer_size = 1000000
        self.batch_size = 128
        self.device = 'cpu'
        self.test_rate = 5
        self.test_per = 10


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
