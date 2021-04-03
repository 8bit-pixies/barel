"""
Based on rlkit implementation:

https://github.com/vitchyr/rlkit/blob/master/rlkit/torch/dqn/dqn.py
"""


from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn

# import rlkit.torch.pytorch_util as ptu
# from rlkit.core.eval_util import create_stats_ordered_dict
# from rlkit.torch.torch_rl_algorithm import TorchTrainer


def soft_update_from_to(source, target, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


class DQNTrainer(object):
    def __init__(
        self,
        qf,
        target_qf,
        learning_rate=1e-3,
        soft_target_tau=1e-3,
        target_update_period=1,
        qf_criterion=None,
        discount=0.99,
        reward_scale=1.0,
    ):
        super().__init__()
        self.qf = qf
        self.target_qf = target_qf
        self.learning_rate = learning_rate
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period
        self.qf_optimizer = optim.Adam(
            self.qf.parameters(),
            lr=self.learning_rate,
        )
        self.discount = discount
        self.reward_scale = reward_scale
        self.qf_criterion = qf_criterion or nn.MSELoss()
        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True

    def learn_one(self, traj):
        """
        learn over a single trajectory
        """
        obs = torch.Tensor(traj["obs"])
        rewards = torch.Tensor(traj["reward"])
        actions = torch.Tensor(traj["action"])
        next_obs = torch.Tensor(traj["next_obs"])
        terminals = torch.Tensor(traj["done"])

        """
        Compute loss
        """

        best_action_idxs = self.qf(next_obs).max(1, keepdim=True)[1]
        target_q_values = self.target_qf(next_obs).gather(1, best_action_idxs).detach()
        y_target = rewards + (1.0 - terminals) * self.discount * target_q_values.squeeze(1)
        y_target = y_target.detach()
        y_pred = torch.sum(self.qf(obs) * actions.reshape(-1, 1), dim=1, keepdim=True)
        qf_loss = self.qf_criterion(y_pred.squeeze(1), y_target)

        """
        Update networks
        """
        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        self.qf_optimizer.step()

        """
        Soft target network updates
        """
        if self._n_train_steps_total % self.target_update_period == 0:
            soft_update_from_to(self.qf, self.target_qf, self.soft_target_tau)

        """
        Save some statistics for eval using just one batch.
        """
        self._n_train_steps_total += 1

    def get_action(self, obs):
        obs = np.expand_dims(obs, axis=0)
        obs = torch.from_numpy(obs).float()
        q_values = self.qf(obs).squeeze(0)
        q_values_np = q_values.to("cpu").detach().numpy()
        return q_values_np.argmax()
