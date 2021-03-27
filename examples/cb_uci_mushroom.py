"""
Using UCI Mushroom dataset like here: 
https://arxiv.org/pdf/1807.09809.pdf

"""

import pandas as pd
import random
import numpy as np
from barel.policy.vw import ContextualBandit
from barel.metric import Metric


column_names = [
    "classes",
    "cap_shape",
    "cap_surface",
    "cap_color",
    "bruises",
    "odor",
    "gill_attachment",
    "gill_spacing",
    "gill_size",
    "gill_color",
    "stalk_shape",
    "stalk_root",
    "stalk_surface_above_ring",
    "stalk_surface_below_ring",
    "stalk_color_above_ring",
    "stalk_color_below_ring",
    "veil_type",
    "veil_color",
    "ring_number",
    "ring_type",
    "spore_print_color",
    "population",
    "habitat",
]

feature_set = [x for x in column_names if x != "classes"]
actions = list(set(data["classes"].to_list())) + ["s"]

data = pd.read_csv("examples/agaricus-lepiota.data", header=None, names=column_names)

# build an offline training set from random sample
train_sample = data.iloc[random.sample(list(data.index), 100)]
chosen_actions = [random.choice(actions) for _ in range(100)]

train_sample["actions"] = chosen_actions
train_sample["probability"] = 0.33
train_sample[["reward"]] = list((train_sample["classes"] == train_sample["actions"]).astype(float))
train_sample[train_sample["actions"] == "s"]["reward"] = [
    random.choice([0, 1]) for _ in range(np.sum(train_sample["actions"] == "s"))
]
train_sample["action"] = [{x: y + 1 for y, x in enumerate(actions)}[x] for x in train_sample["actions"].to_list()]

# create thing.
policy = ContextualBandit(3, feature_set, return_proba=True)

for i in train_sample.index:
    X = train_sample.loc[i, feature_set]
    y = train_sample.loc[i, ["action", "reward", "probability"]]
    policy.learn_one(X.to_dict(), *y.to_list())
