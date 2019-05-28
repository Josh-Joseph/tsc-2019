import numpy as np
import matplotlib
import matplotlib.pylab as plt
import seaborn as sns
import torch
import torch.nn.functional as F
import glob
import io
import base64
from IPython.display import clear_output


device = torch.device("cpu")
DEFAULT_FIG_SIZE = (7, 2)


def visualize_weights(network, figsize=DEFAULT_FIG_SIZE):
    f, axs = plt.subplots(1, len(list(network.children())), figsize=figsize)

    for i, subnet in enumerate(network.children()):
        sns.heatmap(subnet.weight.detach().cpu().numpy(),
                    cbar=False, xticklabels=False, yticklabels=False, ax=axs[i])
        axs[i].set_title('Weights of Region {}'.format(i))
    f.tight_layout()


def visualize_activations(network, x, figsize=DEFAULT_FIG_SIZE):
    f, axs = plt.subplots(1, len(list(network.children())), figsize=figsize)
    activations = x
    for i, subnet in enumerate(network.children()):
        try:
            brain_state = subnet.weight.detach().clone().cpu().numpy() * activations.detach().clone().cpu().numpy()
        except AttributeError:
            brain_state = subnet.weight.detach().clone().cpu().numpy() * activations
        activations = F.relu(subnet.cpu().float()(torch.tensor(activations).float()))

        sns.heatmap(brain_state,
                    cbar=False, xticklabels=False, yticklabels=False, ax=axs[i])
        axs[i].set_title('Activations of Region {}'.format(i))
    f.tight_layout()


def visualize_state(x, figsize=(7, 4)):
    f, ax = plt.subplots(figsize=figsize)
    plt.imshow(np.asarray(x), interpolation='nearest', aspect='auto')
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.tick_params(axis='both', which='both', length=0)
    f.tight_layout()


def visualize_subjective_mental_state(agent, brain_state, figsize=DEFAULT_FIG_SIZE):
    f, ax = plt.subplots(1, 1, figsize=figsize)
    subjective_mental_state = agent.mind.subjective_mental_state(np.concatenate([array.flatten() for array in brain_state]))
    sns.heatmap([subjective_mental_state], cbar=False, xticklabels=False, yticklabels=False, ax=ax)
    ax.set_title('Subjective Mental State')
    f.tight_layout()


def animate_episode_history(episode_history, agent, steps_size=25, pause=3, exclude_mental=False):
    for i in range(0, len(episode_history['world_observation']), steps_size):
        episode_index = i
        if episode_history['brain_state'] is not None and not exclude_mental:
            try:
                brain_state = np.concatenate([array.flatten() for array in episode_history['brain_state'][episode_index]])
                visualize_subjective_mental_state(agent, brain_state)
                visualize_activations(agent.mind.mental_state_classifier, brain_state)
            except AttributeError:
                pass
        try:
            visualize_activations(agent.brain.behavior_network.qnetwork_local, episode_history['world_observation'][episode_index])
        except AttributeError:
            pass
        visualize_state(episode_history['world_image'][episode_index])
        clear_output()
        if episode_history['reported_mental_state'] is not None:
            print('\n'.join(episode_history['reported_mental_state'][episode_index]))
        plt.pause(pause)


# def show_video():
#   mp4list = glob.glob('video/*.mp4')
#   if len(mp4list) > 0:
#     mp4 = mp4list[0]
#     video = io.open(mp4, 'r+b').read()
#     encoded = base64.b64encode(video)
#
#     ipythondisplay.display(HTML(data='''<video alt="test" autoplay
#                 loop controls style="height: 400px;">
#                 <source src="data:video/mp4;base64,{0}" type="video/mp4" />
#              </video>'''.format(encoded.decode('ascii'))))
#   else:
#     print("Could not find video")
#
