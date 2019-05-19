import torch
import torch.nn.functional as F
import numpy as np
from gym_torcs import TorcsEnv
from network import A3C_Network
from a3c_utils import normal

class A3C_Agent(object):
    def __init__(self, rank, global_net, lock, opt, args):
        self.network = A3C_Network(29, 3, lock)
        self.global_net = global_net
        self.lock = lock
        self.opt = opt
        self.args = args
        self.done = True
        self.env = TorcsEnv(port=3101+rank, path='/usr/local/share/games/torcs/config/raceman/quickrace.xml')

    def soft_policy(self):
        value, mu, sigma = self.network(self.state)
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
        return action, value, log_prob, entropy

    def loss(self, t):
        actor_loss, critic_loss = 0, 0
        R = torch.zeros(1, 1)
        if not self.done:
            value = self.soft_policy()[1]
            R = value.detach()

        t.values.append(R)
        gae = torch.zeros(1, 1)
        for i in reversed(range(len(t))):
            R = gamma * R + t.rewards[i]
            advantage = R - t.values[i]
            critic_loss = critic_loss + 0.5 * advantage.pow(2)

            delta_t = t.rewards[i] + self.args.gamma * t.values[i + 1] - t.values[i]
            gae = gae * self.args.gamma + delta_t

            actor_loss = actor_loss - t.log_probs[i].sum() * gae.detach() - self.args.beta * t.entropies[i].sum()

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

    def update(self, t):
        actor_loss, critic_loss = self.loss(t)

        self.network.zero_grad()
        (actor_loss + 0.5 * critic_loss).backward()
        with self.lock:
            self.global_update()

    def reset(self):
        # Synchronizing
        self.network.reset(self.global_net, self.done)
        if self.done:
            self.state = self.env.reset(relaunch=self.done, sampletrack=True, render=False)
