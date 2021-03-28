def rollout(env, policy):
    obs = env.reset()

    done = False
    total_reward = 0
    while not done:
        act = policy.get_action(obs)
        obs, reward, done, _ = env.step(act)
        total_reward += reward
    return total_reward
