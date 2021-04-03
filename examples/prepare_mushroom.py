"""
This is to re-format the data for offline learning
to simplify the scripts

We will provide a training dataset.
"""

import json
import pandas as pd
import numpy as np
import random

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


data = pd.read_csv("examples/agaricus-lepiota.data", header=None, names=column_names)
feature_set = [x for x in column_names if x != "classes"]
actions = list(set(data["classes"].to_list())) + ["s"]
action_mapping = {x: y for y, x in enumerate(actions)}
reverse_mapping = {int(y): x for x, y in action_mapping.items()}

# build an offline training set from random sample
train_index = random.sample(list(data.index), 100)
train_sample = data.iloc[train_index]
chosen_actions = [random.choice(actions) for _ in range(100)]

train_sample["actions"] = chosen_actions
train_sample["probability"] = 0.33
train_sample[["reward"]] = list((train_sample["classes"] == train_sample["actions"]).astype(float))
train_sample[train_sample["actions"] == "s"]["reward"] = [
    random.choice([0, 1]) for _ in range(np.sum(train_sample["actions"] == "s"))
]
train_sample["action"] = [action_mapping[x] for x in train_sample["actions"].to_list()]

X = train_sample[feature_set]
y = train_sample[["action", "reward", "probability"]]

test_sample = data.iloc[[x for x in data.index if x not in train_index]]
X_test = test_sample[feature_set]
y_test = [action_mapping[x] for x in test_sample["classes"].to_list()]  # this is typically dependent on the environment

# export.

data = {
    "X": X.to_dict(),
    "y": y.to_dict(),
    "X_test": X_test.to_dict(),
    "y_test": y_test,
    "actions": actions,
    "action_mapping": action_mapping,
    "reverse_mapping": reverse_mapping,
    "column_names": column_names,
    "feature_set": feature_set,
}

with open("examples/mushroom.json", "w") as outfile:
    outfile.write(json.dumps(data))
