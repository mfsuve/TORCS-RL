from sac_utils import SAC_args, make_sure_dir_exists, log, remove_log_file, OrnsteinUhlenbeckProcess
from buffer import ReplayBuffer
from network import ValueNetwork, SoftQNetwork, PolicyNetwork
from gym_torcs import TorcsEnv
from torch import nn, optim, FloatTensor
import torch
import numpy as np
import matplotlib.pyplot as plt
import time


class SAC_Agent:

    def __init__(self):
        self.env = TorcsEnv(path='/usr/local/share/games/torcs/config/raceman/quickrace.xml')
        self.args = SAC_args()
        self.buffer = ReplayBuffer(self.args.buffer_size)

        action_dim = self.env.action_space.shape[0]
        state_dim = self.env.observation_space.shape[0]
        hidden_dim = 256

        self.value_net = ValueNetwork(state_dim, hidden_dim).to(self.args.device)
        self.target_value_net = ValueNetwork(state_dim, hidden_dim).to(self.args.device)

        self.soft_q_net1 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(self.args.device)
        self.soft_q_net2 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(self.args.device)

        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(self.args.device)

        self.target_value_net.load_state_dict(self.value_net.state_dict())

        self.value_criterion = nn.MSELoss()
        self.soft_q_loss1 = nn.MSELoss()
        self.soft_q_loss2 = nn.MSELoss()

        self.value_opt = optim.Adam(self.value_net.parameters(), lr=self.args.lr)
        self.soft_q_opt1 = optim.Adam(self.soft_q_net1.parameters(), lr=self.args.lr)
        self.soft_q_opt2 = optim.Adam(self.soft_q_net2.parameters(), lr=self.args.lr)
        self.policy_opt = optim.Adam(self.policy_net.parameters(), lr=self.args.lr)

        current_time = time.strftime('%d-%b-%y-%H.%M.%S', time.localtime())
        self.plot_folder = f'plots/{current_time}'
        make_sure_dir_exists(self.plot_folder)
        remove_log_file()

    def train(self):
        time = 0
        eps_n = 0
        rewards = []
        test_rewards = []
        for eps_n in range(1, self.args.max_eps + 1):  # Train loop
            state = self.env.reset(relaunch=(eps_n - 1) % 100 == 0, render=False, sampletrack=False)
            eps_r = 0
            sigma = (self.args.start_sigma - self.args.end_sigma) * (
                max(0, 1 - eps_n / self.args.max_eps)) + self.args.end_sigma
            randomprocess = OrnsteinUhlenbeckProcess(hyprm.theta, sigma, outsize)

            for step in range(self.args.max_eps_time):  # Episode
                if time > 1000:
                    action = self.policy_net.get_action(state, randomprocess).detach()
                    next_state, reward, done, _ = self.env.step(action.numpy())
                else:  # Random actions for the first few times
                    action = self.env.action_space.sample()
                    next_state, reward, done, _ = self.env.step(action)

                self.buffer.push(state, action, reward, next_state, done)

                state = next_state
                eps_r += reward
                time += 1

                if len(self.buffer) > self.args.batch_size:
                    self.update()

                if done:
                    break

            rewards.append(eps_r)

            log(f'Episode {eps_n:<4} Reward: {eps_r}')

            if eps_n % self.args.test_per == 0:
                test_reward = self.test(eps_n)
                test_rewards.append(test_reward)
                self.plot(rewards, test_rewards, eps_n)

    def update(self):
        self.policy_net.train()
        state, action, reward, next_state, done = self.buffer.sample(self.args.batch_size)

        state = FloatTensor(state, dtype=torch.float32).to(self.args.device)
        next_state = FloatTensor(next_state, dtype=torch.float32).to(self.args.device)
        action = FloatTensor(action, dtype=torch.float32).to(self.args.device)
        reward = FloatTensor(reward, dtype=torch.float32).unsqueeze(1).to(self.args.device)
        done = FloatTensor(np.float32(done), dtype=torch.float32).unsqueeze(1).to(self.args.device)

        predicted_q_value1 = self.soft_q_net1(state, action)
        predicted_q_value2 = self.soft_q_net2(state, action)
        predicted_value = self.value_net(state)
        new_action, log_prob, epsilon, mean, log_std = self.policy_net.evaluate(state)

        # Training Q function
        target_value = self.target_value_net(next_state)
        target_q_value = reward + (1 - done) * self.args.gamma * target_value
        q_value_loss1 = self.soft_q_loss1(predicted_q_value1, target_q_value.detach())
        q_value_loss2 = self.soft_q_loss2(predicted_q_value2, target_q_value.detach())

        self.soft_q_opt1.zero_grad()
        q_value_loss1.backward()
        if self.args.clipgrad:
            self.clip_grad(self.soft_q_net1.parameters())
        self.soft_q_opt1.step()
        self.soft_q_opt2.zero_grad()
        q_value_loss2.backward()
        if self.args.clipgrad:
            self.clip_grad(self.soft_q_net2.parameters())
        self.soft_q_opt2.step()

        # Training Value function
        predicted_new_q_value = torch.min(self.soft_q_net1(state, new_action), self.soft_q_net2(state, new_action))
        target_value_func = predicted_new_q_value - self.args.alpha * log_prob.sum()
        value_loss = self.value_criterion(predicted_value, target_value_func.detach())

        self.value_opt.zero_grad()
        value_loss.backward()
        if self.args.clipgrad:
            self.clip_grad(self.value_net.parameters())
        self.value_opt.step()

        # Training Policy function
        policy_loss = (log_prob - predicted_new_q_value).mean()

        self.policy_opt.zero_grad()
        policy_loss.backward()
        if self.args.clipgrad:
            self.clip_grad(self.policy_net.parameters())
        self.policy_opt.step()

        # Updating target value network
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.args.soft_tau) + param.data * self.args.soft_tau
            )

    def test(self, eps_n):
        rewards = 0
        for _ in range(self.args.test_rate):
            state = self.env.reset()
            for t in range(50000):
                action = self.policy_net.get_action(state)
                state, reward, done, _ = self.env.step(action.detach())
                reward += reward
                if done:
                    break
        avg_reward = rewards / self.args.test_rate
        log(f'Test Results | Episode {eps_n:<4} Average Reward: {avg_reward}')
        return avg_reward

    def plot(self, rewards, test_rewards, eps_n):
        figure = plt.figure()
        plt.plot(rewards, label='Train Rewards')
        plt.plot(range(self.args.test_per, eps_n + 1, self.args.test_per), test_rewards, label='Test Rewards')
        plt.xlabel('Episode')
        plt.legend()
        plt.savefig(f'{self.plot_folder}/{eps_n}')

    def clip_grad(self, parameters):
        for param in parameters:
            param.grad.data.clamp_(-1, 1)
