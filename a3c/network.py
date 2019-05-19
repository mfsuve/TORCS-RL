import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

class A3C_Network(nn.Module):
    def __init__(self, state_size, action_size, lock=None):
        super(A3C_Network, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.lrelu1 = nn.LeakyReLU(0.1)
        self.fc2 = nn.Linear(256, 256)
        self.lrelu2 = nn.LeakyReLU(0.1)
        self.fc3 = nn.Linear(256, 128)
        self.lrelu3 = nn.LeakyReLU(0.1)
        self.fc4 = nn.Linear(128, 128)
        self.lrelu4 = nn.LeakyReLU(0.1)

        self.hidden_size = 128

        self.lstm = nn.LSTMCell(128, 128)
        self.critic_linear = nn.Linear(128, 1)
        self.actor_linear = nn.Linear(128, action_size)
        self.actor_linear2 = nn.Linear(128, action_size)

        # initializing weights
        self.apply(self.init_weights)
        lrelu = nn.init.calculate_gain('leaky_relu')
        self.fc1.weight.data.mul_(lrelu)
        self.fc2.weight.data.mul_(lrelu)
        self.fc3.weight.data.mul_(lrelu)
        self.fc4.weight.data.mul_(lrelu)

        self.actor_linear.weight.data = self.init_std(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.actor_linear2.weight.data = self.init_std(
            self.actor_linear2.weight.data, 0.01)
        self.actor_linear2.bias.data.fill_(0)
        self.critic_linear.weight.data = self.init_std(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self.train()
        if lock is not None:
            self.lock = lock
        self.hx = None
        self.cx = None

    def forward(self, x):
        x = self.lrelu1(self.fc1(x))
        x = self.lrelu2(self.fc2(x))
        x = self.lrelu3(self.fc3(x))
        x = self.lrelu4(self.fc4(x))

        x = x.view(1, -1)
        self.hx, self.cx = self.lstm(x, (self.hx, self.cx))
        x = self.hx

        return self.critic_linear(x).squeeze(), \
               F.softsign(self.actor_linear(x)).squeeze(), \
               self.actor_linear2(x).squeeze()

    def init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            m.weight.data.normal_(0, 1)
            m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
            if m.bias is not None:
                m.bias.data.fill_(0)

    def init_std(self, weights, std=1.0):
        x = torch.randn(weights.size())
        x *= std / torch.sqrt((x**2).sum(1, keepdim=True))
        return x

    def reset(self, global_net, done):
        with self.lock:
            self.load_state_dict(global_net.state_dict())
        if done:
            self.hx = torch.zeros(1, self.hidden_size)
            self.cx = torch.zeros(1, self.hidden_size)
        else:
            self.hx = self.hx.detach()
            self.cx = self.cx.detach()
