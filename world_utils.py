import numpy as np
import matplotlib.pylab as plt


def visualize_world(x, figsize=(7, 4)):
    f, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    plt.imshow(np.asarray(x), interpolation='nearest', aspect='auto')
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.tick_params(axis='both', which='both', length=0)
    plt.title('3rd-person view of the world', weight='bold', size=16)
    # f.tight_layout()
