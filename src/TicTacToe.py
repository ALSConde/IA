import numpy as np
from tf_agents.specs import array_spec
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts


class TicTacToe(py_environment.PyEnvironment):
    def __init__(self):
        self._action_spec = array_spec.BoundedArraySpec(
            (3,), np.int32, 0, 3, "action"
        )  # [x,y,symbol]
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(3, 3), dtype=np.int32, minimum=0, name="observation"
        )
        self.board = np.zeros(shape=(3, 3), dtype=np.int32)

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self.board = np.zeros(shape=(3, 3), dtype=np.int32)
        return ts.restart(np.array(self.board))

    def _step(self, action):
        """
        Rewards:
        -1 for invalid moves
        1 if you make the winning move
        0 otherwise
        """
        if len(action) != 2:
            index = tuple(action[:-1])
        else:
            index = tuple(action)

        if not self.inrange(index[0], index[1]) or self.board[tuple(index)] != 0:
            print("INVALID ", action)
            return ts.termination(np.array(self.board), -1)

        else:
            self.board[tuple(index)] = action[-1]
            if self.check_winner(index, action[-1]):
                return ts.termination(np.array(self.board), 1)

            elif self.check_draw():
                return ts.termination(np.array(self.board), 0)

            else:
                return ts.transition(np.array(self.board), 0)

    def inrange(self, i, j):
        if 0 <= i <= 2 and 0 <= j <= 2:
            return True
        return False

    def check_winner(self, index, player):
        """
        Utility function to check if the current move is a winning move
        index: (x,y) coordinates of the current move
        player: [1|2] symbol of the player who made the current move
        """
        x, y = index[0], index[1]

        for i in [-1, 0, 1]:  # for all neighbours of current move
            for j in [-1, 0, 1]:
                if i == j == 0:  # except itself
                    continue

                if self.inrange(x + i, y + j):
                    if self.board[(x + i, y + j)] == player:  # has the same symbol
                        if self.inrange(
                            x + i * 2, y + j * 2
                        ):  # check along the same direction(when you are starting from corners)
                            if self.board[(x + i * 2, y + j * 2)] == player:
                                return True

                        if self.inrange(
                            x - i, y - j
                        ):  # check the other direction(when you are starting from middle)
                            if self.board[(x - i, y - j)] == player:
                                return True

        return False

    def check_draw(self):
        """
        Utility function to check if the game ended in a draw
        Note: check for draw only after checking for a winner
        """
        for i in [0, 1, 2]:
            for j in [0, 1, 2]:
                if self.board[(i, j)] == 0:
                    return False

        return True

    def display(self):
        print(self.board)
