import gym
import numpy as np
import matplotlib.pylab as plt


def run_episode(agent, T_max=500, visualize_behavior=True, seed=0):

    episode_record = {
        'observation': [],
        'action': [],
        'reward': [],
        'world_image': [],
        'brain_state': [],
        'recurrent_state': []
    }

    env = gym.make("LunarLander-v2")
    if seed is not None:
        env.seed(seed)
    state = env.reset()
    recurrent_activation = np.array([[0., 0., 0., 0., 0.]], dtype=np.float32)
    for t in range(T_max):
        if hasattr(agent, 'recurrent_activation'):
            prev_recurrent_activation = recurrent_activation[0]
            state_and_prev_recurrent = np.concatenate((state, prev_recurrent_activation))
            action = agent.act(state_and_prev_recurrent)
            recurrent_activation = agent.recurrent_activation(state_and_prev_recurrent)
        else:
            action = agent.act(state)
        if visualize_behavior:
            world_image = env.render('rgb_array')
        else:
            world_image = None
        state, reward, done, _ = env.step(action)

        if hasattr(agent, 'recurrent_activation'):
            episode_record['observation'].append(np.concatenate([state, recurrent_activation[0]]))
        else:
            episode_record['observation'].append(state)
        episode_record['action'].append(action)
        episode_record['reward'].append(reward)
        episode_record['world_image'].append(world_image)

        if done:
            break

    env.close()

    return episode_record


def visualize(x, figsize=(7, 4)):
    f, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    plt.imshow(np.asarray(x), interpolation='nearest', aspect='auto')
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.tick_params(axis='both', which='both', length=0)
