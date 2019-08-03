from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import norm_col_init, weights_init, normalized_columns_initializer





class A3Clstm(torch.nn.Module):
    def __init__(self, num_inputs, action_space):
        super(A3Clstm, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 5, stride=1, padding=2)
        self.maxp1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, 5, stride=1, padding=1)
        self.maxp2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 4, stride=1, padding=1)
        self.maxp3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.maxp4 = nn.MaxPool2d(2, 2)

        self.lstm = nn.LSTMCell(1024, 512)
        num_outputs = action_space.n
        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, num_outputs)

        self.apply(weights_init)
        relu_gain = nn.init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.conv3.weight.data.mul_(relu_gain)
        self.conv4.weight.data.mul_(relu_gain)
        self.actor_linear.weight.data = norm_col_init(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = norm_col_init(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self.train()

    def forward(self, inputs):
        inputs, (hx, cx) = inputs
        x = F.relu(self.maxp1(self.conv1(inputs)))
        x = F.relu(self.maxp2(self.conv2(x)))
        x = F.relu(self.maxp3(self.conv3(x)))
        x = F.relu(self.maxp4(self.conv4(x)))

        x = x.view(x.size(0), -1)

        hx, cx = self.lstm(x, (hx, cx))

        x = hx

        return self.critic_linear(x), self.actor_linear(x), (hx, cx)








if __name__ == '__main__':
    from deepmind_env import LabEnvironment
    from gym_env import GymEnvironment
    import numpy as np
    # env = LabEnvironment(env_name='nav_maze_static_01')
    # model = UNREALModule(3, action_size=env.get_action_size('nav_maze_static_01'))
    env = GymEnvironment(env_name='Pong-v4')
    model = UNREALModule(3, action_size=env.get_action_size('Pong-v4'))
    env.reset()
    #
    cx = Variable(torch.zeros(1, 256))
    hx = Variable(torch.zeros(1, 256))
    observation, reward, terminal, pixel_change = env.step(0)
    # observation = np.transpose(observation, [2, 0, 1])
    ls = torch.from_numpy(np.array([[0,0,0,0,0,0,0], [1,1,1,1,1,1,0]], dtype=np.float32))
    v, pi, (hx, cx) = model(task_type='pc', states=torch.from_numpy(np.array([observation, observation.copy()])), hx=hx, cx=cx, last_action_rewards=ls)
    print('v={0}'.format(v))
    print('pi={0}'.format(pi))

    env.close()