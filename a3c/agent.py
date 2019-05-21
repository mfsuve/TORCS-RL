import torch
from torch.multiprocessing import Process
import torch.nn.functional as F
import numpy as np
from gym_torcs import TorcsEnv
from network import A3C_Network
from a3c_utils import normal

class A3C_Agent(Process):
    def __init__(self, rank, global_net, counter, lock, opt, args):
        super(A3C_Agent, self).__init__()
        self.network = A3C_Network(29, 3, lock)
        self.global_net = global_net
        self.name = f'Process_{rank}'
        self.counter = counter
        self.lock = lock
        self.opt = opt
        self.args = args
        self.done = True
        self.env = TorcsEnv(port=3101+rank, path='/usr/local/share/games/torcs/config/raceman/quickrace.xml')
        self.reset()

    def run(self):
        eps_n = 0
        eps_time = 0
        eps_r = 0
        while self.counter.value() < self.args.max_time:
            for step in range(self.args.nstep):
                action, value, log_prob, entropy = self.soft_policy()
                self.state, reward, done, info = self.env.step(action)
                # reward = max(min(float(reward), 1.0), -1.0)

                eps_time += 1
                eps_r += reward
                self.done = done or eps_time >= self.args.max_eps_time
                self.counter.increment()

                self.append(value, reward, log_prob, entropy)

                if self.done:
                    break

            if self.done:
                eps_time = 0
                eps_n += 1
                with open(f'../../logs/{self.name}.txt', 'a+') as file:
                    print(f'{self.name} | Episode {eps_n}\t:\tElapsed Time: {self.counter.value():<15}Reward: {eps_r}', \
                            file=file)
                eps_r = 0

            self.update()
            self.reset()

    def soft_policy(self):
        input = torch.from_numpy(self.state).unsqueeze(0).float()
        value, mu, sigma = self.network(input)
        mu = torch.clamp(mu, -1.0, 1.0)
        sigma = F.softplus(sigma) + 1e-5
        eps = torch.randn(mu.size())
        pi = np.array([np.pi])
        pi = torch.from_numpy(pi).float()

        action = (mu + sigma.sqrt() * eps).detach()
        prob = normal(action, mu, sigma)
        action = torch.clamp(action, -1.0, 1.0)
        entropy = 0.5 * ((sigma * 2 * pi.expand_as(sigma)).log() + 1)
        log_prob = (prob + 1e-6).log()
        return action.numpy(), value, log_prob, entropy

    def loss(self):
        actor_loss, critic_loss = 0, 0
        R = torch.zeros(1, 1)
        if not self.done:
            value = self.soft_policy()[1]
            R = value.detach()

        self.values.append(R)
        gae = torch.zeros(1, 1)
        for i in reversed(range(len(self.rewards))):
            R = self.args.gamma * R + self.rewards[i]
            advantage = R - self.values[i]
            critic_loss = critic_loss + 0.5 * advantage.pow(2)

            delta_t = self.rewards[i] + self.args.gamma * self.values[i + 1] - self.values[i]
            gae = gae * self.args.gamma + delta_t

            actor_loss = actor_loss - self.log_probs[i].sum() * gae.detach() - self.args.beta * self.entropies[i].sum()

        return actor_loss, critic_loss

    def global_update(self):
        if next(self.network.parameters()).is_shared():
            raise RuntimeError("Global network(shared) called global update!")
        for global_p, self_p in zip(self.global_net.parameters(), self.network.parameters()):
            if global_p.grad is not None:
                continue
            else:
                global_p._grad = self_p.grad
        self.opt.step()

    def update(self):
        actor_loss, critic_loss = self.loss()

        self.network.zero_grad()
        (actor_loss + 0.5 * critic_loss).backward()
        with self.lock:
            self.global_update()

    def reset(self):
        # Synchronizing
        self.network.reset(self.global_net, self.done)
        if self.done:
            self.state = self.env.reset(relaunch=self.done, sampletrack=True, render=False)
        self.values = []
        self.rewards = []
        self.log_probs = []
        self.entropies = []

    def append(self, value, reward, log_prob, entropy):
        self.values.append(value)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.entropies.append(entropy)
