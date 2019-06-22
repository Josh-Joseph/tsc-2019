import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork, MentalQNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
# BATCH_SIZE = 128         # minibatch size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate
# LR = 5e-5               # learning rate
UPDATE_EVERY = 4        # how often to update the network

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

class BehaviorNetwork(object):

    def __init__(self, path_to_pretrained_network=None):
        assert path_to_pretrained_network is not None
        self.dqn_agent = DQNAgent()
        self.dqn_agent.qnetwork_local.load_state_dict(torch.load(path_to_pretrained_network, map_location='cpu'))
        self.dqn_agent.qnetwork_local.fc1.cpu()
        self.dqn_agent.qnetwork_local.fc2.cpu()
        self.dqn_agent.qnetwork_local.fc3.cpu()

    def get_action(self, state):
        return self.dqn_agent.act(state)


# def classify_states(observation):
#     def high(observation):
#         return observation[:, 1] > 0.2
#
#     def low(observation):
#         return observation[:, 1] <= 0.2
#
#     def left(observation):
#         return observation[:, 0] < -0.2
#
#     def right(observation):
#         return observation[:, 0] > 0.2
#
#     def middle(observation):
#         return observation[:, 0].abs() < 0.2
#
#     # important_areas = [high, low, left, right, middle]
#
#     target_mentals = torch.stack([
#         high(observation),
#         low(observation),
#         left(observation),
#         right(observation),
#         middle(observation)
#     ], dim=1).float()
#
#     return target_mentals


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

    # important_areas = [high, low, left, right, middle]

    target_mentals = torch.stack([
        high(observation),
        low(observation),
        left(observation),
        right(observation),
        falling_too_fast(observation)
    ], dim=1).float()

    return target_mentals


class DQNMentalAgent(object):
    """Interacts with and learns from the environment."""

    def __init__(self, state_size=8, action_size=4, mental_size=5, seed=0):
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
        self.qnetwork_local = MentalQNetwork(state_size, action_size, mental_size, seed).to(device)
        self.qnetwork_target = MentalQNetwork(state_size, action_size, mental_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = MentalReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state_and_prev_mental, action, mental, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state_and_prev_mental, action, mental, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state_and_prev_mental, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state_and_prev_mental = torch.from_numpy(state_and_prev_mental).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state_and_prev_mental)[:, :4]
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def mental_activation(self, state_and_prev_mental):
        state_and_prev_mental = torch.from_numpy(state_and_prev_mental).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            mental_activation = self.qnetwork_local(state_and_prev_mental)[:, -5:]
        self.qnetwork_local.train()
        return mental_activation.cpu().numpy()

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states_and_prev_mentals, actions, mentals, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        next_states_and_mentals = torch.cat([next_states, mentals], dim=1)
        Q_targets_next = self.qnetwork_target(next_states_and_mentals).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states_and_prev_mentals).gather(1, actions)

        # Compute loss
        loss_rl = F.mse_loss(Q_expected, Q_targets)

        states = states_and_prev_mentals[:, :8]
        target_mentals = classify_states(states)
        mental_pred = self.qnetwork_local(states_and_prev_mentals)[:, -5:]

        loss_internal_states = F.multilabel_soft_margin_loss(mental_pred, target_mentals)

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


class DQNAgent(object):
    """Interacts with and learns from the environment."""

    def __init__(self, state_size=8, action_size=4, seed=0):
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
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
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
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


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
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class MentalReplayBuffer:
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
        self.experience = namedtuple("Experience", field_names=["state_and_prev_mental", "action", "mental", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state_and_prev_mental, action, mental, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state_and_prev_mental, action, mental, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states_and_mentals = torch.from_numpy(np.vstack([e.state_and_prev_mental for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        mentals = torch.from_numpy(np.vstack([e.mental for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states_and_mentals, actions, mentals, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)