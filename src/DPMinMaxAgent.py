import numpy as np
import tf_agents
import pickle


class DPMinMaxAgent:  # minmax takes for ever, so memoize it
    """
    Here we use a dictionary to store the rewards, where board is the key and its minmax score is its value.
    Arguments:
        symbol: [1|2] - player symbol
        verbose: [True|False] - want messages printed to console output?
        saveTree: [True|False] - save the constructed tree dict into a file?
        loadTree: [True|False] - load a previously constructed tree dict from memory?
        saveTreeFreq: if saveTree is True, save the tree into memory once every 'x' updates. This is coded to decrease over time.

    Usage:
        Agent = DPMinMaxAgent(2) -- plays second, doesn't save or load from memory
        Agent = DPMinMaxAgent(1, verbose=True, saveTree=False, loadTree=True) -- to load from memory but not save it.

    """

    def __init__(
        self, symbol, verbose=False, saveTree=False, loadTree=False, saveTreeFreq=100
    ):
        self.symbol = symbol
        self.verbose = verbose
        self.trainable = False

        self.saveTreeFreq = saveTreeFreq
        self.saveTreeFreqStart = saveTreeFreq  # high val would mean a lot of them won't be saved; low value would update too many times, so decrease freq periodically
        self.saveTree = saveTree
        self.pickle_loaded = False
        if loadTree:
            try:
                with open("/kaggle/working/minmaxtree.pickle", "rb") as f:
                    self.tree = pickle.load(f)
                    self.pickle_loaded = True
            except Exception as e:
                self.pickle_loaded = False
                print(e)
                loadTree = False

        if not self.pickle_loaded:
            self.tree = {}

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
            score = self.minmax(temp_board, 0, False, (i, j, self.symbol))
            if self.verbose:
                print(i, j, score)
            if int(score) > int(best_score): # type: ignore
                best_score = score
                act = (i, j)
            temp_board[(i, j)] = 0

        return act

    def minmax(self, board, depth, maximise, last_move):
        last_move = list(last_move)

        if self.tree.get(
            board.tobytes()
        ):  # check dict and return value if it already exists
            return self.tree.get(board.tobytes())

        if self.check_winner(board, last_move):
            if last_move[-1] == self.symbol:
                return 20
            else:
                return -20

        if self.check_draw(board):
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
                player = 3 - last_move[-1]

                temp_board[(i, j)] = player
                temp_board_value = self.oneLess(
                    self.minmax(
                        np.array(temp_board), depth + 1, not maximise, (i, j, player)
                    )
                )
                if not self.tree.get(temp_board.tobytes()):  # store value if not exists
                    self.tree[temp_board.tobytes()] = temp_board_value
                    self.saveTreeFreq -= 1

                best_val = max(temp_board_value, best_val)
                temp_board[(i, j)] = 0

        else:
            best_val = 100
            for i, j in empty_slots:
                player = 3 - last_move[-1]

                temp_board[(i, j)] = player
                temp_board_value = self.oneLess(
                    self.minmax(
                        np.array(temp_board), depth + 1, not maximise, (i, j, player)
                    )
                )
                if not self.tree.get(temp_board.tobytes()):
                    self.tree[temp_board.tobytes()] = temp_board_value
                    self.saveTreeFreq -= 1

                best_val = min(temp_board_value, best_val)
                temp_board[(i, j)] = 0

        if self.saveTree and self.saveTreeFreq == 0:
            with open("/kaggle/working/minmaxtree.pickle", "wb") as f:
                pickle.dump(self.tree, f)
                self.saveTreeFreq = self.saveTreeFreqStart
                self.saveTreeFreqStart -= 1  # n*(n-1)/2 updates

        return best_val

    def oneLess(self, x):
        if x > 0:
            return x - 1
        elif x < 0:
            return x + 1
        else:
            return 0

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
