import random


class DoNothingAgent(object):

    def act(self, observation):
        return 0


class RandomAgent(object):

    def act(self, observation):
        return random.randint(0, 3)

