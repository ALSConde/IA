import numpy as np
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
        self._current_time_step = None  # Initialize _current_time_step

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self.board = np.zeros(shape=(3, 3), dtype=np.int32)
        self._current_time_step = ts.restart(np.array(self.board))
        return self._current_time_step

    def _step(self, action):
        if len(action) == 3:
            index = tuple(action[:-1])
        else:
            index = tuple(action)
        
        if not self.inrange(index[0], index[1]) or self.board[tuple(index)] != 0:
            print("INVALID ", action)
            return ts.TimeStep(
                step_type=ts.StepType.LAST,
                reward=-1,
                discount=0.0,
                observation=np.array(self.board),
            )
        else:
            self.board[tuple(index)] = action[-1]
            if self.check_winner(index, action[-1]):
                self._current_time_step = ts.TimeStep(
                    step_type=ts.StepType.LAST,
                    reward=1,
                    discount=0.0,
                    observation=np.array(self.board),
                )
            elif self.check_draw():
                self._current_time_step = ts.TimeStep(
                    step_type=ts.StepType.LAST,
                    reward=0,
                    discount=0.0,
                    observation=np.array(self.board),
                )
            else:
                self._current_time_step = ts.TimeStep(
                    step_type=ts.StepType.MID,
                    reward=0,
                    discount=1.0,
                    observation=np.array(self.board),
                )
            return self._current_time_step

    def inrange(self, i, j):
        if 0 <= i <= 2 and 0 <= j <= 2:
            return True
        return False

    def check_winner(self, index, player):
        x, y = index[0], index[1]

        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if i == j == 0:
                    continue
                if self.inrange(x + i, y + j):
                    if self.board[(x + i, y + j)] == player:
                        if self.inrange(x + i * 2, y + j * 2):
                            if self.board[(x + i * 2, y + j * 2)] == player:
                                return True
                        if self.inrange(x - i, y - j):
                            if self.board[(x - i, y - j)] == player:
                                return True
        return False

    def check_draw(self):
        for i in [0, 1, 2]:
            for j in [0, 1, 2]:
                if self.board[(i, j)] == 0:
                    return False
        return True

    def display(self):
        print(self.board)

    def is_valid(self, action):
        index = tuple(action[:-1])
        if not self.inrange(index[0], index[1]) or self.board[tuple(index)] != 0:
            return False
        return True
