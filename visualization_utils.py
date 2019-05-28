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
DEFAULT_FIG_SIZE = (7, 4)


def visualize_weights(network, figsize=DEFAULT_FIG_SIZE):
    f, axs = plt.subplots(len(list(network.children())), 1, figsize=figsize)

    for i, subnet in enumerate(network.children()):
        sns.heatmap(subnet.weight.detach().cpu().numpy(),
                    cbar=False, xticklabels=False, yticklabels=False, ax=axs[i])
        axs[i].set_title('Weights of Region {}'.format(i))
    f.tight_layout()


def visualize_activations(network, x, figsize=DEFAULT_FIG_SIZE):
    f, axs = plt.subplots(len(list(network.children())), 1, figsize=figsize)
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


def visualize_state(x, figsize=DEFAULT_FIG_SIZE):
    f, ax = plt.subplots(figsize=figsize)
    plt.imshow(np.asarray(x), interpolation='nearest', aspect='auto')
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.tick_params(axis='both', which='both', length=0)
    f.tight_layout()


def animate_episode_history(episode_history, agent, steps_size=25, pause=3):
    for i in range(0, len(episode_history['world_observation']), steps_size):
        episode_index = i
        visualize_state(episode_history['world_image'][episode_index])
        try:
            visualize_activations(agent.dqn_agent.qnetwork_local, episode_history['world_observation'][episode_index])
        except AttributeError:
            pass
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
