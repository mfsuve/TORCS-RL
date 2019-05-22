from a3c_utils import Counter, A3C_args, Recorder
from torch.multiprocessing import Lock, Manager
import torch
import numpy as np
from network import A3C_Network
from agent import Worker, Tester
from shared_adam import Shared_Adam
from time import sleep

counter = Counter()
lock = Lock()
args = A3C_args()
global_net = A3C_Network(29, 3)
global_net.share_memory()
opt = Shared_Adam(global_net.parameters(), lr=args.lr)

# Creating training processes
workers = [Worker(i, global_net, counter, lock, opt, args) for i in range(args.num_workers)]
# Creating testing process
recorder = Recorder(Manager())
workers.append(Tester(args.num_workers, global_net, counter, lock, recorder, args))

for w in workers:
    w.start()
    # sleep(0.1)

for w in workers:
    w.join()
