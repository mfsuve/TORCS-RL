from a3c_utils import Counter, A3C_args
from torch.multiprocessing import Lock
import torch
import numpy as np
from network import A3C_Network
from agent import A3C_Agent
from shared_adam import Shared_Adam
from time import sleep

counter = Counter()
lock = Lock()
args = A3C_args()
global_net = A3C_Network(29, 3)
global_net.share_memory()
opt = Shared_Adam(global_net.parameters(), lr=args.lr)

workers = []
for i in range(args.num_workers):
    worker = A3C_Agent(i, global_net, counter, lock, opt, args)
    workers.append(worker)
    worker.start()
    sleep(0.1)

for w in workers: w.join()
