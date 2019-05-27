import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
import time
import torch
import torch.nn.functional as F
import gym
from dqn_agent import DQNAgent
import matplotlib
from PIL import Image

matplotlib.use("TkAgg")
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


class Agent(DQNAgent):

    def __init__(self, state_size, action_size, seed,
                 actions_NN_path, mental_state_clf, mental_state_mapping):
        super(Agent, self).__init__(state_size, action_size, seed)

        self.actions_NN_path = actions_NN_path
        self.qnetwork_local.load_state_dict(torch.load(self.actions_NN_path))

        self.mental_state_clf = mental_state_clf
        self.mental_state_mapping = mental_state_mapping

    def get_action(self, world_state):
        return self.act(world_state)

    def get_brain_state(self, world_state):
        activations_1 = F.relu(self.qnetwork_local.fc1(torch.tensor(world_state).to(device)))
        activations_2 = F.relu(self.qnetwork_local.fc2(activations_1))
        activations_3 = F.relu(self.qnetwork_local.fc3(activations_2))

        brain_state = []

        brain_state.append(self.qnetwork_local.fc1.weight.detach().cpu().numpy() * world_state)
        brain_state.append(self.qnetwork_local.fc2.weight.detach().cpu().numpy() * activations_1.detach().cpu().numpy())
        brain_state.append(self.qnetwork_local.fc3.weight.detach().cpu().numpy() * activations_2.detach().cpu().numpy())

        return brain_state

    def get_mental_state(self, world_state):
        return np.random.uniform(size=(4, 1))

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
        'mental_state': [],
        'reported_mental_state': []
    }

    f, axs = plt.subplots(5, 1)
    f.set_size_inches(w=10, h=7)

    env = gym.make(environment_name)
    env.seed(12)
    state = env.reset()
    for t in range(400):
        action = agent.act(state)
        world_image = env.render('rgb_array')
        state, reward, done, _ = env.step(action)

        for ax in axs:
            ax.clear()
        for j in [1, 2, 3]:
            sns.heatmap(agent.get_brain_state(state)[j-1], cbar=False, xticklabels=False, yticklabels=False, ax=axs[j])
            axs[j].set_title("Agent's Brain State")
        # sns.heatmap(episode_history['mental_state'][i], cbar=False, xticklabels=False, yticklabels=False, ax=axs[4])
        axs[4].set_title("Agent's Mental State")
        plt.tight_layout()
        # display.clear_output(wait=True)
        # display.display(f)
        # print()
        # print("Agent's Behavioral Description:")
        # print('State: {}\nAction:{}'.format(episode_history['state'][i],
        #                                     episode_history['action'][i]), end='')
        # print()
        # print()
        # print()
        # print("Agent's Self Reported Mental State:")
        # print('\n'.join(episode_history['reported_mental_state'][i]), end='')
        # time.sleep(2.0)
        # plt.close()
        plt.pause(0.05)



        episode_history['state'].append(state)
        episode_history['action'].append(action)
        episode_history['reward'].append(reward)
        episode_history['world_image'].append(world_image)
        episode_history['brain_state'].append(agent.get_brain_state(state))
        episode_history['mental_state'].append(agent.get_mental_state(state))
        episode_history['reported_mental_state'].append(agent.report_mental_state(state))

        if done:
            break

    env.close()

    f_world, ax_world = plt.subplots(1, 1)
    f, axs = plt.subplots(5, 1)
    f.set_size_inches(w=15, h=10)

    for i in range(len(episode_history['state'])):
        # f, axs = plt.subplots(5, 1)
        for ax in axs:
            ax.clear()
        ax_world.imshow(episode_history['world_image'][i])
        ax_world.set_title("Objective Environment")
        for j in [1, 2, 3]:
            sns.heatmap(episode_history['brain_state'][i][j-1], cbar=False, xticklabels=False, yticklabels=False, ax=axs[j])
            axs[j].set_title("Agent's Brain State")
        # sns.heatmap(episode_history['mental_state'][i], cbar=False, xticklabels=False, yticklabels=False, ax=axs[4])
        axs[4].set_title("Agent's Mental State")
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
        print('\n'.join(episode_history['reported_mental_state'][i]), end='')
        # time.sleep(2.0)
        # plt.close()
        plt.pause(0.05)

    plt.show()

if __name__ == "__main__":

    # import matplotlib.pyplot as plt
    # import numpy as np
    # import matplotlib
    #
    # matplotlib.use("TkAgg")
    #
    # x = np.linspace(0, 6 * np.pi, 100)
    # y = np.sin(x)
    #
    # # plt.ion()
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # line1, = ax.plot(x, y, 'r-')
    # plt.draw()
    #
    # for phase in np.linspace(0, 10 * np.pi, 500):
    #     ax.clear()
    #     # line1.set_ydata(np.sin(x + phase))
    #     ax.plot(x, np.sin(x + phase), 'r-')
    #     # plt.draw()
    #     plt.pause(0.02)
    #
    # # plt.ioff()
    # plt.show()



    agent = Agent(state_size=8,
                  action_size=4,
                  seed=0,
                  actions_NN_path='./checkpoint.pth',
                  mental_state_clf='',
                  mental_state_mapping='')

    animate_agent(agent, "LunarLander-v2")








