import torch
import torch.nn.functional as F
from dqn_agent import DQNAgent


def train_agent():
    pass


def load_pretrained_agent():
    class PhsyicalistBrain(object):

        def __init__(self):
            self.behavior_network = DQNAgent()
            try:
                self.behavior_network.qnetwork_local.load_state_dict(torch.load('../checkpoint0.pth',
                                                                                map_location='cpu'))
            except FileNotFoundError:
                self.behavior_network.qnetwork_local.load_state_dict(torch.load('./checkpoint0.pth',
                                                                                map_location='cpu'))
            self.behavior_network.qnetwork_local.fc1.cpu()
            self.behavior_network.qnetwork_local.fc2.cpu()
            self.behavior_network.qnetwork_local.fc3.cpu()

        def brain_state(self, world_observation):
            brain_state = []
            network, activations = self.behavior_network.qnetwork_local, torch.tensor(world_observation)
            for i, subnet in enumerate(network.children()):
                subbrain_state = subnet.weight.detach().clone().cpu().numpy() * activations.detach().clone().cpu().numpy()
                activations = F.relu(subnet.cpu().float()(torch.tensor(activations).float()))
                brain_state.append(subbrain_state)
            return brain_state

        def suggest_action(self, state):
            return self.behavior_network.act(state)

    class TrainedBehaviorAgent(object):

        def __init__(self):
            self.brain = PhsyicalistBrain()

        def act(self, world_observation):
            return self.brain.suggest_action(world_observation)

        def image_brain_state(self, world_observation):
            return self.brain.brain_state(world_observation)

    agent = TrainedBehaviorAgent()

    return agent