def high(observation):
    return observation[1] > 0.2

def low(observation):
    return observation[1] <= 0.2

def left(observation):
    return observation[0] < -0.2

def right(observation):
    return observation[0] > 0.2

def middle(observation):
    return abs(observation[0]) < 0.2

important_areas = [high, low, left, right, middle]




def get_mental_state(agent, observation):
    return [area(observation) for area in important_areas]


def report_mental_state(mental_state):

    beliefs = []

    for i, area in enumerate(important_areas):
        if mental_state[i]:
            beliefs.append(area.__name__)

    print('3rd-person report of the 1st-person view: "I have the beliefs: {}"'.format(', '.join(beliefs)))


if __name__ == "__main__":
    from src.training import load_pretrained_agent

    agent = load_pretrained_agent()

    from src import world

    episode_history = world.run_episode(agent, visualize_behavior=False)

    episode_index = 50
    mental_state = get_mental_state(agent, episode_history['observation'][episode_index])

    report_mental_state(mental_state)
    print(1)