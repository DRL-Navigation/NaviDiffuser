## multi-obejcetive super mario bros
## modified by Runzhe Yang on Dec. 8, 2018

import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
import math
from torch.nn import init

from torch.distributions.categorical import Categorical

from typing import List, Tuple


def mlp(input_mlp: List[Tuple[int, int]]) -> nn.Sequential:
    if not input_mlp:
        return nn.Sequential()
    mlp_list = []
    for input_dim, out_put_dim, af in input_mlp:
        mlp_list.append(nn.Linear(input_dim, out_put_dim, bias=True))
        if af == "relu":
            mlp_list.append(nn.ReLU())
        if af == 'sigmoid':
            mlp_list.append(nn.Sigmoid())
    return nn.Sequential(*mlp_list)

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
        

class BaseActorCriticNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(BaseActorCriticNetwork, self).__init__()
        linear = nn.Linear
        self.feature = nn.Sequential(
            linear(input_size, 128),
            nn.ReLU(),
            linear(128, 128),
            nn.ReLU()
        )
        self.actor = linear(128, output_size)
        self.critic = linear(128, 1)

        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                init.kaiming_uniform_(p.weight)
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                init.kaiming_uniform_(p.weight, a=1.0)
                p.bias.data.zero_()

    def forward(self, state):
        x = self.feature(state)
        policy = self.actor(x)
        value = self.critic(x)
        return policy, value


class DeepCnnActorCriticNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(DeepCnnActorCriticNetwork, self).__init__()
        
        linear = nn.Linear

        self.feature = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=5,
                stride=2),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=4,
                stride=1),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=4,
                stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4),
            nn.ReLU(),
            Flatten(),
            linear(50176, 512),
            nn.ReLU()
        )
        self.actor = linear(512, output_size)
        self.critic = linear(512, 1)

        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                init.kaiming_uniform_(p.weight)
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                init.kaiming_uniform_(p.weight, a=1.0)
                p.bias.data.zero_()

    def forward(self, state):
        x = self.feature(state)
        policy = self.actor(x)
        value = self.critic(x)
        return policy, value


class CnnActorCriticNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(CnnActorCriticNetwork, self).__init__()
        
        linear = nn.Linear

        self.feature = nn.Sequential(
            nn.Conv2d(
                in_channels=4,
                out_channels=32,
                kernel_size=8,
                stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1),
            nn.LeakyReLU(),
            Flatten(),
            linear(
                7 * 7 * 64,
                512),
            nn.LeakyReLU(),
        )
        self.actor = linear(512, output_size)
        self.critic = linear(512, 1)

        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                init.kaiming_uniform_(p.weight)
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                init.kaiming_uniform_(p.weight, a=1.0)
                p.bias.data.zero_()

    def forward(self, state):
        x = self.feature(state)
        policy = self.actor(x)
        value = self.critic(x)
        return policy, value


class NaiveMoCnnActorCriticNetwork(nn.Module):
    def __init__(self, input_size, output_size, reward_size):
        super(NaiveMoCnnActorCriticNetwork, self).__init__()
        linear = nn.Linear
        self.feature = nn.Sequential(
            nn.Conv2d(
                in_channels=4,
                out_channels=32,
                kernel_size=8,
                stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1),
            nn.LeakyReLU(),
            Flatten(),
            linear(
                7 * 7 * 64,
                512),
            nn.LeakyReLU(),
        )
        self.actor = nn.Sequential(
            linear(512+reward_size, 128),
            nn.LeakyReLU(),
            linear(128, output_size),
        )
        self.critic = nn.Sequential(
            linear(512+reward_size, 128),
            nn.LeakyReLU(),
            linear(128, 1),
        )

        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                init.kaiming_uniform_(p.weight)
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                init.kaiming_uniform_(p.weight, a=1.0)
                p.bias.data.zero_()

    def forward(self, state, preference):
        x = self.feature(state)
        x = torch.cat((x, preference), dim=1)
        policy = self.actor(x)
        value = self.critic(x)
        return policy, value


class EnveMoCnnActorCriticNetwork(nn.Module):
    def __init__(self, input_size, output_size, reward_size):
        super(EnveMoCnnActorCriticNetwork, self).__init__()
        
        linear = nn.Linear

        self.conv1d1_actor =  torch.nn.Conv1d(1, 32, 5, 2, "valid")
        self.conv1d2_actor =  torch.nn.Conv1d(32, 32, 3, 2, "valid")
        self.fc_1d_actor = mlp([ (7616, 256, "relu")])
        self.fc1_actor = mlp([ (5, 256, "relu")])
        self.fc2_actor = nn.Linear(256+256+256, 512)
        self.fc3_actor = mlp([ (reward_size, 256, "relu")])

        self.conv1d1_critic =  torch.nn.Conv1d(1, 32, 5, 2, "valid")
        self.conv1d2_critic =  torch.nn.Conv1d(32, 32, 3, 2, "valid")
        self.fc_1d_critic = mlp([ (7616, 256, "relu")])
        self.fc1_critic = mlp([ (5, 256, "relu")])
        self.fc2_critic = nn.Linear(256+256+256, 512)
        self.fc3_critic = mlp([ (reward_size, 256, "relu")])

        self.actor = nn.Sequential(
            linear(512, 128),
            nn.LeakyReLU(),
            linear(128, output_size),
        )
        self.critic = nn.Sequential(
            linear(512, 128),
            nn.LeakyReLU(),
            linear(128, reward_size),
        )

        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                init.kaiming_uniform_(p.weight)
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                init.kaiming_uniform_(p.weight, a=1.0)
                p.bias.data.zero_()
    
    def _encode_laser_actor(self, x):

        x = self.conv1d1_actor(x)
        x = self.conv1d2_actor(x)
        x = self.fc_1d_actor(x.view(x.shape[0], -1))

        return x
    
    def _encode_laser_critic(self, x):

        x = self.conv1d1_critic(x)
        x = self.conv1d2_critic(x)
        x = self.fc_1d_critic(x.view(x.shape[0], -1))

        return x

    def forward(self, state, preference):
        laser = state[:, :, :960]
        vector = state[:, :, 960:].squeeze(1)

        encoded_laser_actor = self._encode_laser_actor(laser)
        encoded_vector_actor = self.fc1_actor(vector)
        encoded_preference_actor = self.fc3_actor(preference)
        x_actor = torch.cat((encoded_laser_actor, encoded_vector_actor, encoded_preference_actor), dim=1)
        x_actor = self.fc2_actor(x_actor)

        policy = self.actor(x_actor)

        encoded_laser_critic = self._encode_laser_critic(laser)
        encoded_vector_critic = self.fc1_critic(vector)
        encoded_preference_critic = self.fc3_critic(preference)
        x_critic = torch.cat((encoded_laser_critic, encoded_vector_critic, encoded_preference_critic), dim=1)
        x_critic = self.fc2_critic(x_critic)

        value = self.critic(x_critic)
        return policy, value
