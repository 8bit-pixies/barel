import pandas as pd
import random
import json
import numpy as np
from barel.policy.dqn import DQNTrainer

from examples.prepare_simple_navigation import SimpleNavigation

import torch
from torch import nn as nn
from torch.nn import functional as F


class Mlp(nn.Module):
    def __init__(
        self,
        hidden_sizes,
        output_size,
        input_size,
        init_w=3e-3,
        hidden_activation=F.relu,
        output_activation=lambda x: x,
        b_init_value=0.0,
        layer_norm=False,
        layer_norm_kwargs=None,
    ):
        super().__init__()

        if layer_norm_kwargs is None:
            layer_norm_kwargs = dict()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.layer_norm = layer_norm
        self.fcs = []
        self.layer_norms = []
        in_size = input_size

        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            fc.bias.data.fill_(b_init_value)
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)

            if self.layer_norm:
                ln = LayerNorm(next_size)
                self.__setattr__("layer_norm{}".format(i), ln)
                self.layer_norms.append(ln)

        self.last_fc = nn.Linear(in_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.fill_(0)

    def forward(self, input, return_preactivations=False):
        h = input
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            if self.layer_norm and i < len(self.fcs) - 1:
                h = self.layer_norms[i](h)
            h = self.hidden_activation(h)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)
        if return_preactivations:
            return output, preactivation
        else:
            return output


qf = Mlp(
    hidden_sizes=[32, 32],
    input_size=4,
    output_size=4,
)
target_qf = Mlp(
    hidden_sizes=[32, 32],
    input_size=4,
    output_size=4,
)

qf_criterion = nn.MSELoss()
policy = DQNTrainer(
    qf=qf,
    target_qf=target_qf,
    qf_criterion=qf_criterion,
)

with open("examples/simple_nav.json", "r") as f:
    data = json.load(f)

for traj in data:
    policy.learn_one(traj)

env = SimpleNavigation()
obs = env.reset()
act = policy.get_action(obs)
obs, _, _, _ = env.step(act)
