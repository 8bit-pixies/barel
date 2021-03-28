"""
This is a vw wrapper - only supports `cb` mode as we're only
doing batch rl.
"""

from vowpalwabbit import pyvw
import pandas as pd
import numpy as np


def to_vw_instance_format(context, action=None, reward=None, proba=None, feature_names=None):
    """
    convert dataset to vw format
    """
    if type(context) is dict:
        vw_context = " ".join([f"{i}:{x}" for i, x in context.items() if x != 0])
    elif feature_names is None:
        vw_context = " ".join([f"c{i}:{x}" for i, x in enumerate(context) if x != 0])
    else:
        vw_context = " ".join([f"{i}:{x}" for i, x in zip(feature_names, context) if x != 0])

    if action is not None:
        cost = -reward
        vw_label = f"{action}:{cost}:{proba}"
    else:
        vw_label = ""

    instance = f"{vw_label} | {vw_context}"
    return instance


class ContextualBandit(object):
    """
    Contextual bandit object
    TODO: Mixin for interface standardisation

    Usage:

        train_data = [{'action': 1, 'cost': 2, 'probability': 0.4, 'feature1': 'a', 'feature2': 'c', 'feature3': ''},
                    {'action': 3, 'cost': 0, 'probability': 0.2, 'feature1': 'b', 'feature2': 'd', 'feature3': ''},
                    {'action': 4, 'cost': 1, 'probability': 0.5, 'feature1': 'a', 'feature2': 'b', 'feature3': ''},
                    {'action': 2, 'cost': 1, 'probability': 0.3, 'feature1': 'a', 'feature2': 'b', 'feature3': 'c'},
                    {'action': 3, 'cost': 1, 'probability': 0.7, 'feature1': 'a', 'feature2': 'd', 'feature3': ''}]

        train_df = pd.DataFrame(train_data)

        # Add index to data frame
        train_df['index'] = range(1, len(train_df) + 1)
        train_df = train_df.set_index("index")
        policy = ContextualBandit(4, ["feature1", "feature2", "feature3"])

        for i in train_df.index:
            X = train_df.loc[i, ["feature1", "feature2", "feature3"]]
            y = train_df.loc[i, ['action', 'cost', 'probability']]
            policy.learn_one(X.to_dict(), *y.to_list())

        policy.get_action(X.to_dict())

        policy_proba = ContextualBandit(4, ["feature1", "feature2", "feature3"], return_proba=True)

        for i in train_df.index:
            X = train_df.loc[i, ["feature1", "feature2", "feature3"]]
            y = train_df.loc[i, ['action', 'cost', 'probability']]
            policy_proba.learn_one(X.to_dict(), *y.to_list())

        policy_proba.get_action_proba(X.to_dict())
    """

    def __init__(self, num_actions, feature_names=None, return_proba=False, quiet=True):
        if quiet:
            quiet_option = "--quiet"
        else:
            quiet_option = ""
        if return_proba:
            vw_option = "cb_explore"
        else:
            vw_option = "cb"
        self.model = pyvw.vw(f"--{vw_option} {num_actions} {quiet_option}")
        self.feature_names = feature_names
        self.return_proba = return_proba

    def learn_one(self, context, action, reward, proba):
        if type(context) is pd.DataFrame:
            self.feature_names = list(context.columns)
        instance = to_vw_instance_format(context, action, reward, proba, self.feature_names)
        self.model.learn(instance)
        return self

    def get_action(self, context):
        instance = to_vw_instance_format(context, feature_names=self.feature_names)
        action = self.model.predict(instance)

        if self.return_proba:
            action = np.argmax(action)  # vw is 1 index
        else:
            action = action - 1  # vw is 1 index
        return action

    def get_action_proba(self, context):
        if not self.return_proba:
            raise Exception("Policy initialised without return proba!")
        instance = to_vw_instance_format(context, feature_names=self.feature_names)
        return self.model.predict(instance)
