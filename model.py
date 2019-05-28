import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    # https://github.com/udacity/deep-reinforcement-learning/blob/master/dqn/solution/model.py

    def __init__(self, state_size, action_size, seed, fc1_units=32, fc2_units=32):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class MLP(nn.Module):
    def __init__(self, num_neurons, num_mental_states, fc_units=32):
        super(MLP, self).__init__()
        self.num_neurons = num_neurons
        self.num_mental_states = num_mental_states
        self.fc_units = fc_units
        self.fc1 = nn.Linear(self.num_neurons, self.fc_units)
        self.fc2 = nn.Linear(self.fc_units, self.fc_units)
        self.fc3 = nn.Linear(self.fc_units, self.num_mental_states)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
