import copy
import numpy as np
import torch
import matplotlib.pylab as plt
import seaborn as sns
import torch.nn.functional as F


device = torch.device("cpu")
DEFAULT_FIG_SIZE = (7, 2)


third_person_brain_state_ontology = {
    'observation': np.array,
    'action': np.int64,
    'layer_weights': list,
    'activations': list
}


def get_brain_state(agent, state_and_recurrent):

    observation = state_and_recurrent[:8]
    prev_recurrent = state_and_recurrent[-5:]

    weights = []
    activations = []

    activations.append(observation)
    networks = list(agent.qnetwork_local.children())
    for subnet in networks:
        weights.append(subnet.weight.detach().cpu().numpy())

    activations.append(F.relu(networks[0](torch.tensor(activations[0]))).detach().numpy())
    activations.append(F.relu(networks[1](torch.cat([torch.tensor(activations[1]),
                                                     torch.tensor(prev_recurrent)]))).detach().numpy())
    activations.append(torch.sigmoid(networks[2](torch.cat([torch.tensor(activations[1]),
                                                            torch.tensor(activations[2])]))).detach().numpy())
    activations.append(networks[3](torch.tensor(activations[2])).detach().numpy())

    brain_state = copy.deepcopy(third_person_brain_state_ontology)
    brain_state['observation'] = observation
    brain_state['action'] = agent.act(observation)
    brain_state['layer_weights'] = weights
    brain_state['activations'] = activations

    return brain_state


def visualize_weights(brain_state, figsize=DEFAULT_FIG_SIZE):

    f, axs = plt.subplots(1, len(brain_state['layer_weights']), figsize=figsize, constrained_layout=True)

    f.suptitle("network layer weights", weight='bold', size=16)

    for i, layer_weights in enumerate(brain_state['layer_weights']):
        sns.heatmap(layer_weights, cbar=False, xticklabels=False, yticklabels=False, ax=axs[i])
        axs[i].set_title('Layer {} weights '.format(i+1))


def visualize_activations(brain_state, figsize=DEFAULT_FIG_SIZE):

    f, axs = plt.subplots(1, len(brain_state['activations']), figsize=figsize, constrained_layout=True)

    for i, activations in enumerate(brain_state['activations']):

        sns.heatmap(np.array([activations]).T, cbar=False, xticklabels=False, yticklabels=False, ax=axs[i])
        axs[i].set_title('Activations {}'.format(i))

    f.suptitle("network activations at time t", weight='bold', size=16)




if __name__ == "__main__":
    from src.training import load_pretrained_agent

    agent = load_pretrained_agent()

    from src import world

    episode_history = world.run_episode(agent, visualize_behavior=False)

    episode_index = 50
    brain_state = get_brain_state(agent, episode_history['observation'][episode_index])
    visualize_weights(brain_state)
    visualize_activations(brain_state)
    print(1)