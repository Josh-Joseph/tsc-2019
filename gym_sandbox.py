
import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from PIL import Image


env = gym.make('LunarLander-v2')
env.seed(0)
print('State shape: ', env.observation_space.shape)
print('Number of actions: ', env.action_space.n)

# from dqn_agent import Agent

# agent = Agent(state_size=8, action_size=4, seed=0)

# watch an untrained agent
state = env.reset()
for j in range(200):
    # action = agent.act(state)
    #     tmp = env.render('rgb_array')
    action = env.action_space.sample()
    image = Image.fromarray(env.render(mode='rgb_array'))
    env.close()
    # imshow(image)
    print(image)
    #     print(tmp)
    state, reward, done, _ = env.step(action)
    print('{}, {}, {}'.format(state, action, reward), end='')
    if done:
        break

# env.close()
