"""
Using UCI Mushroom dataset like here: 
https://arxiv.org/pdf/1807.09809.pdf

"""

import pandas as pd
import random
import json
import numpy as np
from barel.policy.vw import ContextualBandit
from barel.metric import Metric
from barel.rollout import rollout

with open("examples/mushroom.json", "r") as f:
    data = json.load(f)


class Mushroom(object):
    """
    Notice that although we've put this in a gym format
    there isn't a concept of a transition probability between
    states
    """

    actions = data["actions"]
    action_mapping = data["action_mapping"]
    reverse_mapping = data["reverse_mapping"]
    labels = data["y_test"]
    data = pd.DataFrame(data["X_test"])
    max_cycle = 100

    def get_mushroom(self):
        indx = int(random.choice(range(len(self.data.index))))
        X = self.data.iloc[indx]
        self.label = self.labels[indx]
        return X

    def get_reward(self, act):
        if act == "s":
            return random.choice([0, 1])
        if act == "e" and self.label == "p":
            self.cycle = 100
            return 0
        else:
            return 1

    def reset(self):
        self.cycle = 0
        return self.get_mushroom()

    def step(self, action):
        act = self.reverse_mapping[str(action)]
        reward = self.get_reward(act)
        obs = self.get_mushroom()
        self.cycle += 1
        done = self.cycle >= self.max_cycle
        return obs, reward, done, {}


# create thing.
policy = ContextualBandit(3, return_proba=True)
metric = Metric()
X_train = pd.DataFrame(data["X"])
y_train = pd.DataFrame(data["y"])

for i in X_train.index:
    X = X_train.loc[i][data["feature_set"]]
    y = y_train.loc[i][["action", "reward", "probability"]]
    policy.learn_one(X.to_dict(), *y.to_list())
    for _ in range(10):
        metric.add(rollout(Mushroom(), policy))
    metric.update()
