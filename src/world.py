import gym
import numpy as np
import matplotlib.pylab as plt


def run_episode(agent, T_max=400, visualize_behavior=True, seed=2):

    episode_record = {
        'observation': [],
        'action': [],
        'reward': [],
        'world_image': [],
        'brain_state': [],
        'mental_state': [],
        'reported_mental_state': []
    }

    env = gym.make("LunarLander-v2")
    # if visualize_behavior:
    #     env = Monitor(env, './video/' + agent.name, force=True)
    if seed is not None:
        env.seed(seed)
    state = env.reset()
    mental_activation = np.array([[0., 0., 0., 0., 0.]], dtype=np.float32)
    for t in range(T_max):
        if hasattr(agent, 'mental_activation'):
            prev_mental_activation = mental_activation[0]
            state_and_prev_mental = np.concatenate((state, prev_mental_activation))
            action = agent.act(state_and_prev_mental)
            mental_activation = agent.mental_activation(state_and_prev_mental)
        else:
            action = agent.act(state)
        if visualize_behavior:
            world_image = env.render('rgb_array')
        else:
            world_image = None
        state, reward, done, _ = env.step(action)

        if hasattr(agent, 'mental_activation'):
            episode_record['observation'].append(np.concatenate([state, mental_activation[0]]))
        else:
            episode_record['observation'].append(state)
        episode_record['action'].append(action)
        episode_record['reward'].append(reward)
        episode_record['world_image'].append(world_image)

        if done:
            break

        # try:
        #     episode_record['brain_state'].append(agent.image_brain_state(state))
        # except AttributeError:
        #     pass
        # try:
        #     episode_record['mental_state'].append(agent.mind.subjective_mental_state(agent.image_brain_state(state)))
        # except AttributeError:
        #     pass
        # try:
        #     episode_record['reported_mental_state'].append(agent.report_mental_state(state))
        # except AttributeError:
        #     pass

    env.close()

    return episode_record


def visualize(x, figsize=(7, 4)):
    f, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    plt.imshow(np.asarray(x), interpolation='nearest', aspect='auto')
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.tick_params(axis='both', which='both', length=0)
    plt.title('3rd-person view of the world', weight='bold', size=16)
    # f.tight_layout()
