from a3c_utils import Counter, A3C_args, TrajectoryList
from torch.multiprocessing import Lock, Process
import torch
import numpy as np
from network import A3C_Network
from agent import A3C_Agent
from shared_adam import Shared_Adam

class Worker(Process):
    def __init__(self, rank, global_net, counter, lock, opt, args):
        super(Worker, self).__init__()
        self.agent = A3C_Agent(rank, global_net, lock, opt, args)
        self.rank = rank
        self.counter = counter
        self.args = args

    def run(self):
        eps_n = 0
        eps_time = 0
        self.agent.reset()
        while self.counter.value() < self.args.max_time:

            trajectories = TrajectoryList()
            for step in range(self.args.nstep):
                action, value, log_prob, entropy = self.agent.soft_policy()
                self.agent.state, reward, done, info = self.agent.env.step(action)
                # reward = max(min(float(reward), 1.0), -1.0)

                eps_time += 1
                self.agent.done = done or eps_time >= args.max_eps_time
                self.counter.increment()

                trajectories.append(value, reward, log_prob, entropy)

                if self.agent.done:
                    break

            if self.agent.done:
                eps_time = 0
                eps_n += 1

            self.agent.update(trajectories)
            self.agent.reset()

counter = Counter()
lock = Lock()
args = A3C_args()
global_net = A3C_Network(29, 3)
global_net.share_memory()
opt = Shared_Adam(global_net.parameters(), lr=args.lr)

workers = [Worker(i, global_net, counter, lock, opt, args) for i in range(args.num_workers)]

for w in workers: w.start()
for w in workers: w.join()
