# Adapted from https://github.com/udacity/deep-reinforcement-learning/blob/master/dqn/solution/Deep_Q_Network_Solution.ipynb


import torch
import numpy as np
import gym
from collections import deque
from dqn_agent_with_internal_states import DQNInternalStateAgent
import matplotlib.pyplot as plt


def dqn(agent, env, n_episodes=3000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon
    for i_episode in range(1, n_episodes + 1):
        state = env.reset()
        recurrent_activation = np.array([[0., 0., 0., 0., 0.]], dtype=np.float32)
        score = 0
        for t in range(max_t):
            prev_recurrent_activation = recurrent_activation[0]
            state_and_prev_recurrent = np.concatenate((state, prev_recurrent_activation))
            action = agent.act(state_and_prev_recurrent, eps)
            recurrent_activation = agent.recurrent_activation(state_and_prev_recurrent)
            next_state, reward, done, _ = env.step(action)
            agent.step(state_and_prev_recurrent, action, recurrent_activation, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")

        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))

        if np.mean(scores_window) >= 225.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), '../models/trained_model.pth')
            break

    return scores


def train_agent():
    agent = DQNInternalStateAgent(state_size=8, action_size=4, recurrent_size=5, seed=0)

    env = gym.make('LunarLander-v2')
    env.seed(0)

    scores = dqn(agent, env)

    # plot the scores
    fig = plt.figure()
    _ = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()


def load_pretrained_agent():
    agent = DQNInternalStateAgent()
    agent.qnetwork_local.load_state_dict(torch.load('../models/trained_model.pth', map_location='cpu'))
    return agent


if __name__ == "__main__":
    train_agent()
