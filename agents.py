import random
import torch
from dqn_agent import DQNAgent


device = torch.device("cpu")


class DoNothingAgent(object):

    def __init__(self, no_opt_action_index=0):
        self.no_opt_action_index = no_opt_action_index

    def act(self, state):
        return self.no_opt_action_index


class RandomAgent(object):

    def __init__(self, num_actions=4):
        self.num_actions = num_actions

    def act(self, state):
        return random.randint(0, self.num_actions-1)


class TrainedBehaviorAgent(object):

    def __init__(self, state_size=8, action_size=4, seed=0):
        self.dqn_agent = DQNAgent(state_size=state_size, action_size=action_size, seed=seed)
        self.dqn_agent.qnetwork_local.load_state_dict(torch.load('../checkpoint0.pth'))

    def act(self, state):
        return self.dqn_agent.act(state)
