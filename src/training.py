import torch
import numpy as np
import gym
from collections import deque
from dqn_agent import DQNMentalAgent
import matplotlib.pyplot as plt


def train_agent():
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    agent = DQNMentalAgent(state_size=8, action_size=4, mental_size=5, seed=0)
    # file_name = '../models/solved.pth'
    file_name = '../models/checkpoint.pth'
    if False:
        try:
            agent.qnetwork_local.load_state_dict(torch.load('../' + file_name, map_location='cpu'))
            agent.qnetwork_target.load_state_dict(torch.load('../' + file_name, map_location='cpu'))
        except FileNotFoundError:
            agent.qnetwork_local.load_state_dict(torch.load(file_name, map_location='cpu'))
            agent.qnetwork_target.load_state_dict(torch.load( file_name, map_location='cpu'))

    def dqn(n_episodes=3000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
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
            mental_activation = np.array([[0., 0., 0., 0., 0.]], dtype=np.float32)
            score = 0
            for t in range(max_t):
                prev_mental_activation = mental_activation[0]
                state_and_prev_mental = np.concatenate((state, prev_mental_activation))
                action = agent.act(state_and_prev_mental, eps)
                mental_activation = agent.mental_activation(state_and_prev_mental)
                next_state, reward, done, _ = env.step(action)
                agent.step(state_and_prev_mental, action, mental_activation, reward, next_state, done)
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
                torch.save(agent.qnetwork_local.state_dict(), '../models/checkpoint.pth')
            if np.mean(scores_window) >= 225.0:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100, np.mean(scores_window)))
                torch.save(agent.qnetwork_local.state_dict(), '../models/solved.pth')
                break
        return scores

    env = gym.make('LunarLander-v2')
    env.seed(0)

    scores = dqn()

    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()


def load_pretrained_agent():

    agent = DQNMentalAgent()
    file_name = '../models/checkpoint.pth'
    # file_name = '../models/solved.pth'
    try:
        agent.qnetwork_local.load_state_dict(torch.load('../' + file_name, map_location='cpu'))
    except FileNotFoundError:
        agent.qnetwork_local.load_state_dict(torch.load(file_name, map_location='cpu'))

    return agent



if __name__ == "__main__":
    train_agent()