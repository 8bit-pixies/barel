import json
import pandas as pd
import numpy as np
import random


class SimpleNavigation(object):
    """
    This environment has 4 rooms, where the aim to arrive at the goal room.
    As it is fairly simple, we can hardcode all the transitions etc...

    ```
    +-+-+
    |0|1|
    +-+-+
    |2|3|
    +-+-+
    ```
    """

    state = 0

    def reset(self):
        self.state = 0
        return self.gen_obs()

    def gen_obs(self):
        state_mapping = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        return state_mapping[self.state]

    def gen_reward(self):
        state_mapping = [-0.2, -0.1, -0.1, 1]
        return state_mapping[self.state]

    def step(self, action):
        """
        action is: UP DOWN LEFT RIGHT
        """
        state_action_mapping = {0: [0, 2, 0, 1], 1: [1, 3, 0, 1], 2: [0, 2, 2, 3], 3: [1, 3, 2, 3]}
        self.state = state_action_mapping[self.state][action]
        reward = self.gen_reward()
        obs = self.gen_obs()
        done = self.state == 3
        return obs, reward, done, {}


#
num_trajectories = 100
data = []
max_cycle = 100

for _ in range(num_trajectories):
    obs_list = []
    action_list = []
    done_list = []
    reward_list = []
    next_obs_list = []
    env = SimpleNavigation()
    obs = env.reset()
    done = False

    for _ in range(max_cycle):
        action = random.choice(range(4))
        next_obs, reward, done, _ = env.step(action)
        obs_list.append(obs)
        action_list.append(action)
        reward_list.append(reward)
        next_obs_list.append(next_obs)
        done_list.append(done)
        obs = next_obs
        if done:
            break
    data.append(
        {"obs": obs_list, "action": action_list, "done": done_list, "reward": reward_list, "next_obs": next_obs_list}
    )


with open("examples/simple_nav.json", "w") as outfile:
    outfile.write(json.dumps(data))
