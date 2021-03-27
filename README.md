# Barel

Batch Reinforcment Learning for Python.

Its hard putting reinforcement learning in production. Barel's ambition is to make it a bit easier to do so; under the guise of batch reinforcement learning (aka offline reinforcement learning).

To start off with, this library will be a simple (opinionated) wrapper around vowpal wabbit, which I intend on extending to deep learning via pytorch. 

# Design Philosophy

Many of the challenges in productionising reinforcement learning for model-free approaches is the coupling between the setup of the training scheme and the underlying environments. 

This is more than reasonable in academic papers, and their respective setup, but does make it tremendously hard to compare different algorithms with each other on a level playing field.

The goal of this library is not necessarily to create a new evaluation benchmark, but instead to focus on an accessible API for training and productionising reinforcement learning (while probably breaking a few "best practises")

## Training Loop

The training loop relies on offline approach without interaction. 

# Quick Start

_the library isn't written yet, and this is merely the proposed API_

**Contextual Bandits**

_this is fairly redundant as vowpal wabbit would probably outperform whatever I can come up with, but for normalising the API interfaces and completeness_

Even in the contextual bandit format, vowpal wabbit actually demands a multi-line format for a single "instance" (though only a single function call per instance, particularly in `adf` form). My interpretation of what this could look like in a simplified API is shown below.

```py
from barel import policy
from barel import rollout
from barel import metric


# input is: context, (action, reward, expl_proba), avail_actions [optional]
# n.b. vowpal wabbit uses cost (minimise) instead of reward (maximise)
for context, action, reward, proba in dataset:
    # this may change based on how contextual bandits work, as its not trajectory based.
    total_reward = rollout(env, policy)  # not used in true batch rl; do we overload it in vw style?
    metric = metric.update(total_reward)  # not used in true batch rl
    policy.learn_one(context, action, reward, proba)  #  in vw - this is interpretted as a multiline instance
    # policy.get_action(context)
```

**Markov Decision Process**

If we extend the ideas from vowpal wabbit to MDPs, then perhaps the policy should be trained on a trajectory basis rather than each individual state. This becomes more apparent if the policy leverages GRUs (n.b. I know that with the correct representation you don't need whole trajectories to train GRU-style models).

```py
from barel import policy
from barel import rollout
from barel import metric


# trajectory is the full s, a, r, s tuple across time as well...
for trajectory in dataset:
    # we learn based on trajectories in the replay buffer
    # this differs from various approaches which randomly sample from replay buffer
    total_reward = rollout(env, policy)  # not used in true batch rl
    metric = metric.update(total_reward)  # not used in true batch rl
    policy.learn_one(trajectory)
    # policy.get_action(state)

```




