import numpy as np
import tf_agents


class MinMaxAgent:  # builds a minmax tree

    """
    Arguments:
           symbol: [1|2] - player symbol
           verbose: (optional) [True|False] - want messages printed to console output?

       Usage:
           Agent =  MinMaxAgent(2, True) -- plays first and prints debug messages

    """

    def __init__(self, symbol, verbose=False):
        self.symbol = symbol
        self.verbose = verbose
        self.trainable = False

    def action(self, timeStep):
        board = np.array(timeStep.observation)

        act = self.getBestAction(np.array(board))

        return tf_agents.trajectories.PolicyStep(
            action=list(act) + [self.symbol], state=board, info=self.symbol
        )

    def getBestAction(self, board):
        empty_slots = []
        for i in range(0, 3):
            for j in range(0, 3):
                if board[(i, j)] == 0:
                    empty_slots.append([i, j])

        best_score = -100
        act = ()
        temp_board = np.array(board)
        if self.verbose:
            print("MinMax Says:")
        for [i, j] in empty_slots:
            temp_board[(i, j)] = self.symbol
            score = self.minmax(np.array(temp_board), 0, False, (i, j, self.symbol))
            if self.verbose:
                print(i, j, score)
            if score > best_score:
                best_score = score
                act = (i, j)
            temp_board[(i, j)] = 0

        return act

    def minmax(self, board, depth, maximise, last_move):
        """
        MinMax tree: Alternate between your and opponent's move.
        In your move you'll pick the maximising choice; In the opponents move pick the minimising choice(which is his maximum)
        Here:
            A winning move gets 10 points
            A losing move gets -10 points
            A drawn board gets 0
            All intermediate moves gets (finalScore - numberOfStepsLeft)
        Imagine a tree with leaf nodes having 0/10/-10 and with each level above gets max or min of all its branches depending on the level
        """

        if self.check_winner(board, last_move):
            if last_move[-1] == self.symbol:
                return 10
            else:
                return -10

        if self.check_draw(board):  # check draw only after checking for winners
            return 0

        empty_slots = []
        for i in range(0, 3):
            for j in range(0, 3):
                if board[(i, j)] == 0:
                    empty_slots.append([i, j])

        temp_board = np.array(board)

        if maximise:
            best_val = -100
            for i, j in empty_slots:
                player = 3 - last_move[-1]  # switch b/w 1 and 2

                temp_board[(i, j)] = player  # pick one of the empty spots

                best_val = max(
                    self.minmax(
                        np.array(temp_board), depth + 1, not maximise, (i, j, player)
                    ),
                    best_val,
                )

                temp_board[(i, j)] = 0  # clear the picked empty spot

        else:
            best_val = 100
            for i, j in empty_slots:
                player = 3 - last_move[-1]
                temp_board[(i, j)] = player

                best_val = min(
                    self.minmax(
                        np.array(temp_board), depth + 1, not maximise, (i, j, player)
                    ),
                    best_val,
                )

                temp_board[(i, j)] = 0

        return best_val

    def check_draw(self, board):
        board = np.array(board)
        for i in [0, 1, 2]:
            for j in [0, 1, 2]:
                if board[(i, j)] == 0:
                    return False

        return True

    def check_winner(self, board, last_move):
        board = np.array(board)
        x, y, player = last_move[0], last_move[1], last_move[2]

        for i in [-1, 0, 1]:  # for all neighbours of current move
            for j in [-1, 0, 1]:
                if i == j == 0:  # except itself
                    continue

                if self.inrange(x + i, y + j):
                    if board[(x + i, y + j)] == player:  # has the same symbol
                        if self.inrange(
                            x + i * 2, y + j * 2
                        ):  # check along the same direction(when you are starting from corners)
                            if board[(x + i * 2, y + j * 2)] == player:
                                return True

                        if self.inrange(
                            x - i, y - j
                        ):  # check the other direction(when you are starting from middle)
                            if board[(x - i, y - j)] == player:
                                return True

        return False

    def inrange(self, i, j):
        if 0 <= i <= 2 and 0 <= j <= 2:
            return True
        return False
