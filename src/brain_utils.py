import numpy as np
import matplotlib.pylab as plt
import seaborn as sns


DEFAULT_FIG_SIZE = (7, 2)



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