# Adapted from https://github.com/udacity/deep-reinforcement-learning/blob/master/dqn/solution/dqn_agent.py


import numpy as np
import random
import copy
from collections import namedtuple, deque
import torch
import torch.nn.functional as F
import torch.optim as optim
from model import InternalQNetwork


BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate
UPDATE_EVERY = 4        # how often to update the network


device = torch.device("cpu")


def classify_states(observation):
    def high(observation):
        return observation[:, 1] > 0.5

    def low(observation):
        return observation[:, 1] <= 0.5

    def left(observation):
        return observation[:, 0] > 0.

    def right(observation):
        return observation[:, 0] <= 0.

    def falling_too_fast(observation):
        return observation[:, 3] < -0.2

    target_recurrents = torch.stack([
        high(observation),
        low(observation),
        left(observation),
        right(observation),
        falling_too_fast(observation)
    ], dim=1).float()

    return target_recurrents


class DQNInternalStateAgent(object):
    """Interacts with and learns from the environment."""

    def __init__(self, state_size=8, action_size=4, recurrent_size=5, seed=0):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = InternalQNetwork(state_size, action_size, recurrent_size, seed).to(device)
        self.qnetwork_target = InternalQNetwork(state_size, action_size, recurrent_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state_and_prev_recurrent, action, recurrent, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state_and_prev_recurrent, action, recurrent, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state_and_prev_recurrent, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state_and_prev_recurrent = torch.from_numpy(state_and_prev_recurrent).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state_and_prev_recurrent)[:, :4]
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def recurrent_activation(self, state_and_prev_recurrent):
        state_and_prev_recurrent = torch.from_numpy(state_and_prev_recurrent).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            recurrent_activation = self.qnetwork_local(state_and_prev_recurrent)[:, -5:]
        self.qnetwork_local.train()
        return recurrent_activation.cpu().numpy()

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states_and_prev_recurrents, actions, recurrents, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        next_states_and_recurrents = torch.cat([next_states, recurrents], dim=1)
        Q_targets_next = self.qnetwork_target(next_states_and_recurrents).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states_and_prev_recurrents).gather(1, actions)

        # Compute loss
        loss_rl = F.mse_loss(Q_expected, Q_targets)

        states = states_and_prev_recurrents[:, :8]
        target_recurrents = classify_states(states)
        recurrent_pred = self.qnetwork_local(states_and_prev_recurrents)[:, -5:]

        loss_internal_states = F.multilabel_soft_margin_loss(recurrent_pred, target_recurrents)

        loss = loss_rl + loss_internal_states

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def get_brain_state(self, state_and_recurrent):

        observation = state_and_recurrent[:8]
        prev_recurrent = state_and_recurrent[-5:]

        weights, activations = list(), list()

        activations.append(observation)
        networks = list(self.qnetwork_local.children())
        for subnet in networks:
            weights.append(subnet.weight.detach().cpu().numpy())

        activations.append(F.relu(networks[0](torch.tensor(activations[0]))).detach().numpy())
        activations.append(F.relu(networks[1](torch.cat([torch.tensor(activations[1]),
                                                         torch.tensor(prev_recurrent)]))).detach().numpy())
        activations.append(torch.sigmoid(networks[2](torch.cat([torch.tensor(activations[1]),
                                                                torch.tensor(activations[2])]))).detach().numpy())
        activations.append(networks[3](torch.tensor(activations[2])).detach().numpy())

        brain_state = dict()
        brain_state['layer_weights'] = weights
        brain_state['activations'] = activations

        return brain_state


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state_and_prev_recurrent", "action", "recurrent", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state_and_prev_recurrent, action, recurrent, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state_and_prev_recurrent, action, recurrent, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states_and_recurrents = torch.from_numpy(np.vstack([e.state_and_prev_recurrent for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        recurrents = torch.from_numpy(np.vstack([e.recurrent for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states_and_recurrents, actions, recurrents, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
