import gym
from gym.wrappers import Monitor
import torch
import numpy as np


def run_episode(agent, T_max=400, visualize_behavior=True, seed=None):

    episode_record = {
        'world_observation': [],
        'action': [],
        'reward': [],
        'world_image': [],
        'brain_state': [],
        'mental_state': [],
        'reported_mental_state': []
    }

    env = gym.make("LunarLander-v2")
    # if visualize_behavior:
    #     env = Monitor(env, './video/' + agent.name, force=True)
    if seed is not None:
        env.seed(seed)
    state = env.reset()
    for t in range(T_max):
        action = agent.act(state)
        if visualize_behavior:
            world_image = env.render('rgb_array')
        else:
            world_image = None
        state, reward, done, _ = env.step(action)

        episode_record['world_observation'].append(state)
        episode_record['action'].append(action)
        episode_record['reward'].append(reward)
        episode_record['world_image'].append(world_image)

        if done:
            break

        try:
            episode_record['brain_state'].append(agent.brain_state(state))
        except AttributeError:
            pass
        try:
            episode_record['mental_state'].append(agent.mental_state(state))
        except AttributeError:
            pass
        try:
            episode_record['reported_mental_state'].append(agent.report_mental_state(state))
        except AttributeError:
            pass

    env.close()

    return episode_record
