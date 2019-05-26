from sac_utils import SAC_args, make_sure_dir_exists, log, remove_log_file, OrnsteinUhlenbeckProcess, Checkpoint
from mail_util import send_mail
from buffer import ReplayBuffer
from network import ValueNetwork, SoftQNetwork, PolicyNetwork
from gym_torcs import TorcsEnv
from torch import nn, optim, FloatTensor
import torch
import numpy as np
import matplotlib.pyplot as plt
import time


class SAC_Agent:

    def __init__(self, load_from=None):
        self.env = TorcsEnv(path='/usr/local/share/games/torcs/config/raceman/quickrace.xml')
        self.args = SAC_args()
        self.buffer = ReplayBuffer(self.args.buffer_size)

        action_dim = self.env.action_space.shape[0]
        state_dim = self.env.observation_space.shape[0]
        hidden_dim = 256

        self.action_size = action_dim
        self.state_size = state_dim

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
        self.model_save_folder = f'model/{current_time}'
        make_sure_dir_exists(self.plot_folder)
        make_sure_dir_exists(self.model_save_folder)
        self.cp = Checkpoint(self.model_save_folder)

        if load_from is not None:
            try:
                self.load_checkpoint(load_from)
            except FileNotFoundError:
                print(f'{load_from} not found. Running default.')

    def train(self):
        remove_log_file()
        time = 0
        eps_n = 0
        rewards = []
        test_rewards = []
        best_reward = -np.inf
        # sample_track = False
        for eps_n in range(1, self.args.max_eps + 1):  # Train loop
            relaunch = (eps_n - 1) % (100 / self.args.test_rate) == 0
            # if not sample_track:
            #     sample_track = (eps_n - 1) % self.args.change_track_per == 0
            # if sample_track and relaunch:
            #     log('Sampling new track')
            state = self.env.reset(relaunch=relaunch, render=False, sampletrack=False)#sample_track)
            # if relaunch:
            #     sample_track = False
            eps_r = 0
            sigma = (self.args.start_sigma - self.args.end_sigma) * (
                max(0, 1 - (eps_n - 1) / self.args.max_eps)) + self.args.end_sigma
            randomprocess = OrnsteinUhlenbeckProcess(self.args.theta, sigma, self.action_size)

            for step in range(self.args.max_eps_time):  # Episode
                if time > 1000:
                    action = self.policy_net.get_train_action(state, randomprocess).detach()
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

            test_reward = self.test(eps_n)
            test_rewards.append(test_reward)

            if test_reward > best_reward:
                best_reward = test_reward
                self.save_checkpoint(eps_n, best_reward)

            log(f'Episode {eps_n:<4} Reward: {eps_r:<10.5f} Test Reward: {test_reward:<10.5f}')

            if eps_n % self.args.plot_per == 0:
                self.plot(rewards, test_rewards, eps_n)

    def update(self):
        self.policy_net.train()
        state, action, reward, next_state, done = self.buffer.sample(self.args.batch_size)

        state = FloatTensor(state).to(self.args.device)
        next_state = FloatTensor(next_state).to(self.args.device)
        action = FloatTensor(action).to(self.args.device)
        reward = FloatTensor(reward).unsqueeze(1).to(self.args.device)
        done = FloatTensor(np.float32(done)).unsqueeze(1).to(self.args.device)

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
        rewards = []
        for step in range(self.args.test_rate):
            state = self.env.reset(relaunch=False, render=False, sampletrack=False)
            running_reward = 0
            for t in range(self.args.max_eps_time):
                action = self.policy_net.get_test_action(state)
                state, reward, done, _ = self.env.step(action.detach())
                running_reward += reward
                if done:
                    break
            rewards.append(running_reward)
        avg_reward = sum(rewards) / self.args.test_rate
        return avg_reward

    def plot(self, rewards, test_rewards, eps_n):
        torch.save({
            'train_rewards': rewards,
            'test_rewards': test_rewards
        }, f'{self.plot_folder}/{eps_n}.pth')
        figure = plt.figure()
        plt.plot(rewards, label='Train Rewards')
        plt.plot(test_rewards, label='Test Rewards')
        plt.xlabel('Episode')
        plt.legend()
        plt.savefig(f'{self.plot_folder}/{eps_n}.png')
        try:
            send_mail(f'Torcs SAC | Episode {eps_n}', f'{self.plot_folder}/{eps_n}.png')
            log('Mail has been sent.')
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            emsg = e.args[-1]
            emsg = emsg[:1].lower() + emsg[1:]
            log('Couldn\'t send mail because', emsg)

    def clip_grad(self, parameters):
        for param in parameters:
            param.grad.data.clamp_(-1, 1)

    def save_checkpoint(self, eps_n, test_reward):
        self.cp.update(self.value_net, self.soft_q_net1, self.soft_q_net2, self.policy_net)
        self.cp.save(f'e{eps_n}-r{test_reward:.4f}.pth')
        log(f'Saved checkpoint at episode {eps_n}.')

    def load_checkpoint(self, load_from):
        state_dicts = torch.load(load_from)
        self.value_net.load_state_dict(state_dicts['best_value'])
        self.soft_q_net1.load_state_dict(state_dicts['best_q1'])
        self.soft_q_net2.load_state_dict(state_dicts['best_q2'])
        self.policy_net.load_state_dict(state_dicts['best_policy'])
        print(f'Loaded from {load_from}.')

    def race(self, sampletrack=True):
        with torch.no_grad():
            state = self.env.reset(relaunch=False, render=True, sampletrack=sampletrack)
            running_reward = 0
            done = False
            while not done:
                action = self.policy_net.get_test_action(state)
                state, reward, done, _ = self.env.step(action.detach())
                running_reward += reward

            print('Reward:', running_reward)
