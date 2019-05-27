import matplotlib.pylab as plt
import seaborn as sns
import time
import torch
import gym
from dqn_agent import DQNAgent







class Agent(DQNAgent):

    def __init__(self, state_size, action_size, seed,
                 actions_NN_path, mental_state_clf, mental_state_mapping):
        super(Agent, self).__init__(state_size, action_size, seed)

        self.actions_NN_path = actions_NN_path
        agent.qnetwork_local.load_state_dict(torch.load(self.actions_NN_path))

        self.mental_state_clf = mental_state_clf
        self.mental_state_mapping = mental_state_mapping

    def get_action(self, world_state):
        return self.act(world_state)

    def get_brain_state(self, world_state):
        pass

    def report_mental_state(self, world_state):
        return ["I believe I'm falling to fast",
                "I desire to land",
                "I desire to move more to the left"]
        # return self.mental_state_clf(world_state)



def animate_agent(agent, environment_name):

    episode_history = {
        'state': [],
        'action': [],
        'reward': [],
        'world_image': [],
        'brain_state': [],
        'mental_state': []
    }

    env = gym.make(environment_name)
    env.seed(0)
    state = env.reset()
    for j in range(200):
        action = agent.act(state)
        world_image = env.render('rgb_array')
        state, reward, done, _ = env.step(action)

        episode_history['state'].append(state)
        episode_history['action'].append(action)
        episode_history['reward'].append(reward)
        episode_history['world_image'].append(world_image)
        episode_history['brain_state'].append(agent.get_brain_state(state))
        episode_history['mental_state'].append(agent.get_mental_state(state))

        if done:
            break

    env.close()

    for i in range(len(episode_history['state'])):
        f, (ax1, ax2, ax3) = plt.subplots(3, 1)
        ax1.imshow(episode_history['world_image'][i])
        ax1.set_title("Objective Environment")
        sns.heatmap(episode_history['world_image'][i], cbar=False, xticklabels=False, yticklabels=False, ax=ax2)
        ax2.set_title("Agent's Brain State")
        sns.heatmap(episode_history['brain_state'][i], cbar=False, xticklabels=False, yticklabels=False, ax=ax3)
        ax3.set_title("Agent's Mental State")
        plt.tight_layout()
        # display.clear_output(wait=True)
        # display.display(f)
        print()
        print("Agent's Behavioral Description:")
        print('State: {}\nAction:{}'.format(episode_history['state'][i],
                                            episode_history['action'][i]), end='')
        print()
        print()
        print()
        print("Agent's Self Reported Mental State:")
        print('\n'.join(episode_history['mental_state'][i]), end='')
        time.sleep(2.0)
        plt.close()


if __name__ == "__main__":
    agent = Agent(state_size=8,
                  action_size=4,
                  seed=0,
                  actions_NN_path='./checkpoint.pth',
                  mental_state_clf='',
                  mental_state_mapping='')








