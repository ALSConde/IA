import tf_agents
import random


class RandomTTTAgent:
    """
     Arguments:
        symbol: [1|2] - player symbol
    Usage:
        Agent =  RandomTTTAgent(1) # Plays first

    """

    def __init__(self, symbol):
        self.symbol = symbol
        self.trainable = False

    def action(self, timestep):
        board = timestep.observation
        empty_slots = []
        for i in range(0, 3):
            for j in range(0, 3):
                if board[(i, j)] == 0:
                    empty_slots.append([i, j])

        choice = random.choice(empty_slots)
        return tf_agents.trajectories.PolicyStep(
            action=choice + [self.symbol], state=board, info=self.symbol
        )
