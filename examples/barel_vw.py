"""
Example using: https://vowpalwabbit.org/tutorials/contextual_bandits.html
"""

from barel.policy.vw import ContextualBandit
from barel.metric import Metric

train_data = [
    {
        "action": 1,
        "cost": 2,
        "probability": 0.4,
        "feature1": "a",
        "feature2": "c",
        "feature3": "",
    },
    {
        "action": 3,
        "cost": 0,
        "probability": 0.2,
        "feature1": "b",
        "feature2": "d",
        "feature3": "",
    },
    {
        "action": 4,
        "cost": 1,
        "probability": 0.5,
        "feature1": "a",
        "feature2": "b",
        "feature3": "",
    },
    {
        "action": 2,
        "cost": 1,
        "probability": 0.3,
        "feature1": "a",
        "feature2": "b",
        "feature3": "c",
    },
    {
        "action": 3,
        "cost": 1,
        "probability": 0.7,
        "feature1": "a",
        "feature2": "d",
        "feature3": "",
    },
]

train_df = pd.DataFrame(train_data)

train_df["index"] = range(1, len(train_df) + 1)
train_df = train_df.set_index("index")
train_df["reward"] = -train_df["cost"]
policy = ContextualBandit(4, ["feature1", "feature2", "feature3"], return_proba=True)
metric = Metric()

for i in train_df.index:
    X = train_df.loc[i, ["feature1", "feature2", "feature3"]]
    y = train_df.loc[i, ["action", "reward", "probability"]]
    policy.learn_one(X.to_dict(), *y.to_list())

    # this is a dummy just to ensure metric works
    for reward in policy.get_action_proba(X.to_dict()):
        metric.add(reward)
    metric.update()
