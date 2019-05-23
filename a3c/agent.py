import torch
from torch.multiprocessing import Process
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from gym_torcs import TorcsEnv
from network import A3C_Network
from mail_config import Config
from mail_util import send_mail
from a3c_utils import normal, Logger
from time import sleep
from threading import Thread

class A3C_Agent(Process):
    def __init__(self, rank, global_net, counter, lock, args):
        super(A3C_Agent, self).__init__()
        self.network = A3C_Network(29, 3, lock)
        self.global_net = global_net
        self.counter = counter
        self.lock = lock
        self.args = args
        self.done = True
        self.name = ''
        self.env = TorcsEnv(port=3101+rank, path='/usr/local/share/games/torcs/config/raceman/quickrace.xml')
        self.reset()

    def reset(self, relaunch=None):
        if relaunch is None:
            relaunch = self.done
        # Synchronizing
        self.time_step = self.counter.value()
        self.network.reset(self.global_net, self.done)
        if self.done:
            self.state = self.env.reset(relaunch=relaunch, sampletrack=True, render=False)
        self.values = []
        self.rewards = []
        self.log_probs = []
        self.entropies = []

    def internal_log(self, *msg):
        with open(f'../../logs/{self.name}.txt', 'a+') as file:
            print(*msg, file=file)


class Worker(A3C_Agent):
    def __init__(self, rank, global_net, counter, lock, opt, args):
        super(Worker, self).__init__(rank, global_net, counter, lock, args)
        self.name = f'Worker_{rank}'
        self.opt = opt

    def run(self):
        eps_n = 0
        eps_time = 0
        eps_r = 0
        relaunch = False
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
                relaunch = eps_n % 10 == 0
                self.internal_log(f'{self.name} | Episode {eps_n:<4}: Elapsed Time: {self.counter.value():<10} Reward: {eps_r}')
                eps_r = 0

            self.update()
            self.reset(relaunch=relaunch)
            relaunch = False

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

    def append(self, value, reward, log_prob, entropy):
        self.values.append(value)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.entropies.append(entropy)


class Tester(A3C_Agent):
    def __init__(self, rank, global_net, counter, lock, recorder, args):
        super(Tester, self).__init__(rank, global_net, counter, lock, args)
        self.network.eval()
        self.name = f'Tester'
        self.recorder = recorder
        self.logger = Logger()

    def run(self):
        eps_n = 0
        eps_time = 0
        eps_r = 0
        best_r = -np.inf
        while self.counter.value() < self.args.max_time:

            action = self.greedy_policy()
            self.state, reward, done, info = self.env.step(action)

            eps_time += 1
            eps_r += reward
            self.done = done or eps_time >= self.args.max_eps_time

            if self.done:
                eps_time = 0
                eps_n += 1
                if eps_r >= best_r:
                    best_r = eps_r
                    self.recorder.best_net.load_state_dict(self.network.state_dict())
                self.recorder.rewards.append(eps_r)
                self.recorder.best_rewards.append(best_r)
                self.recorder.time_steps.append(self.time_step)
                self.logger.log(f'Reward: {eps_r:.5f}\tBest Reward: {best_r:.5f}')
                if (eps_n + 1) % self.args.plot_rate == 0:
                    self.plot()
                eps_r = 0

            with torch.no_grad():
                self.reset()

    def greedy_policy(self):
        with torch.no_grad():
            input = torch.from_numpy(self.state).unsqueeze(0).float()
            value, mu, sigma = self.network(input)
            mu = torch.clamp(mu, -1.0, 1.0).squeeze()
            action = mu.cpu().numpy()
        return action

    def plot(self):
        def save_and_send(title, x, r, b):
            path = '../../logs/plot.png'
            fig = plt.figure()
            plt.title(title)
            plt.xlabel('time steps')
            plt.plot(x, r, label='rewards')
            plt.plot(x, b, label='best rewards')
            plt.legend()
            fig.savefig(path)
            sleep(5)
            send_mail('Plot of rewards and best rewards', path, Config())
        mail_thread = Thread(target=save_and_send, args=(f'Time: {self.time_step}',
                                                         list(self.recorder.time_steps),
                                                         list(self.recorder.rewards),
                                                         list(self.recorder.best_rewards)))
        mail_thread.start()
