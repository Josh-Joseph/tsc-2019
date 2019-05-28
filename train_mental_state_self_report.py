import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
from dqn_agent import DQNAgent
import matplotlib
from PIL import Image
from dqn_agent import DQNAgent
from sandbox import Agent
from collections import OrderedDict

matplotlib.use("TkAgg")
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


class MLP(nn.Module):
    def __init__(self, num_neurons, num_mental_states, fc_units=32):
        super(MLP, self).__init__()
        self.num_neurons = num_neurons
        self.num_mental_states = num_mental_states
        self.fc_units = fc_units
        self.layers = nn.Sequential(
            nn.Linear(self.num_neurons, self.fc_units),
            nn.ReLU(),
            nn.Linear(self.fc_units, self.fc_units),
            nn.ReLU(),
            nn.Linear(self.fc_units, self.num_mental_states)
        )

    def forward(self, x):
        # convert tensor (128, 1, 28, 28) --> (128, 1*28*28)
        # x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x



def collect_labeled_data(n_episodes=20):
    agent = Agent(state_size=8,
                  action_size=4,
                  seed=0,
                  actions_NN_path='./checkpoint0.pth',
                  mental_state_clf='',
                  mental_state_mapping='')

    episode_history = {
        'state': [],
        'action': [],
        'reward': [],
        'world_image': [],
        'brain_state': [],
        'mental_state': [],
        'reported_mental_state': [],
        'labeled_mental_state': []
    }

    env = gym.make("LunarLander-v2")
    env.seed(12)
    for i in range(n_episodes):
        state = env.reset()
        for t in range(400):
            action = agent.act(state)
            state, reward, done, _ = env.step(action)

            episode_history['state'].append(state)
            episode_history['action'].append(action)
            episode_history['reward'].append(reward)
            # episode_history['world_image'].append(world_image)
            episode_history['brain_state'].append(agent.get_brain_state(state))
            episode_history['mental_state'].append(agent.get_mental_state(state))
            episode_history['reported_mental_state'].append(agent.report_mental_state(state))
            episode_history['labeled_mental_state'].append([
                e(state) for _, e in mental_states
            ])
    return episode_history


# np.sum([np.prod(array.shape) for array in agent.get_brain_state(state)])


mental_states = [
    ["I believe I'm falling to fast", lambda state: state[3] < -0.2],
    ["I desire to land", lambda state: state[-2] != 1 and state[-1] != 1],
    ["I'm afraid to tip over", lambda state: np.abs(state[5]) > 0.2],
    ["I desire to go left", lambda state: state[0] > 0.2],
    ["I desire to go right", lambda state: state[0] < -0.2]
]


episode_history = collect_labeled_data(n_episodes=3)

X = np.vstack([np.concatenate([array.flatten() for array in brain_state]) for brain_state in episode_history['brain_state']])
y = np.asarray(episode_history['labeled_mental_state'], dtype=int)


model = MLP(num_neurons=1408, num_mental_states=len(mental_states), fc_units=32)


optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MultiLabelSoftMarginLoss()

mean_train_losses = []
mean_valid_losses = []
valid_acc_list = []
epochs = 100

for epoch in range(epochs):
    model.train()

    train_losses = []
    valid_losses = []

    optimizer.zero_grad()
    outputs = model(torch.tensor(X))
    loss = loss_fn(outputs.cpu(), torch.tensor(y, dtype=torch.float32))
    loss.backward()
    optimizer.step()

    train_losses.append(loss.item())

    # for i, (images, labels) in enumerate(train_loader):
    #
    #     optimizer.zero_grad()
    #
    #     outputs = model(images)
    #     loss = loss_fn(outputs, labels)
    #     loss.backward()
    #     optimizer.step()
    #
    #     train_losses.append(loss.item())
    #
    #     if (i * 128) % (128 * 100) == 0:
    #         print(f'{i * 128} / 50000')

    model.eval()
    correct = 0
    total = 0
    # with torch.no_grad():
    #     for i, (images, labels) in enumerate(valid_loader):
    #         outputs = model(images)
    #         loss = loss_fn(outputs, labels)
    #
    #         valid_losses.append(loss.item())
    #
    #         _, predicted = torch.max(outputs.data, 1)
    #         correct += (predicted == labels).sum().item()
    #         total += labels.size(0)

    mean_train_losses.append(np.mean(train_losses))
    # mean_valid_losses.append(np.mean(valid_losses))

    # accuracy = 100 * correct / total
    accuracy = 0
    valid_acc_list.append(accuracy)
    print('epoch : {}, train loss : {:.4f}, valid loss : {:.4f}, valid acc : {:.2f}%' \
          .format(epoch + 1, np.mean(train_losses), np.mean(valid_losses), accuracy))

agent = Agent(state_size=8,
              action_size=4,
              seed=0,
              actions_NN_path='./checkpoint0.pth',
              mental_state_clf='',
              mental_state_mapping='')

env = gym.make("LunarLander-v2")
env.seed(12)
state = env.reset()

brain_state = agent.get_brain_state(state)
brain_state = np.concatenate([array.flatten() for array in brain_state])
mental_state = F.sigmoid(model(torch.tensor(brain_state)))
for i, mental_s in enumerate(mental_state):
    if mental_s > 0.5:
        print(mental_states[i])

for t in range(200):
    action = agent.act(state)
    state, reward, done, _ = env.step(action)

torch.save(model.state_dict(), 'mental_classifer.pth')

print(1)