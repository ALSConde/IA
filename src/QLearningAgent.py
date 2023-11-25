import pickle
import collections
import random
import numpy as np
import tf_agents


class QLearningAgent:
    def __init__(
        self,
        symbol,
        trainable=True,
        alpha=0.1,
        gamma=1,
        load_av=False,
        save_av=False,
        saveAVFreq=100,
    ):
        """
        Action values: Dictionary with board as index and a dictionary of {state:{action:reward}} as values

        Arguments:
            symbol: [1|2] - player symbol
            trainable: [True|False] - should we use the current game for training?
            alpha: learning rate
            gamma: discount rate
            load_av: [True|False] - load a previously trained dict from memory?
            save_av: [True|False] - save action values into memory?
            saveAVFreq: save action values after these many updates

        Usage:
            Agent = QLearningAgent(2) -- plays second, doesn't save or load from memory
            Agent = QLearningAgent(1, save_av=False, load_av=True) -- load trained data from memory but don't save any updates.
            Agent = QLearningAgent(1, alpha=0.3, gamma=0.9)

        """
        self.symbol = symbol
        self.trainable = trainable
        self.alpha = alpha
        self.gamma = gamma

        self.saveAVFreq = saveAVFreq
        self.save_av = save_av
        self.pickle_loaded = False
        if load_av:
            try:
                with open("/kaggle/working/av.pickle", "rb") as f:
                    self.av = pickle.load(f)
                    self.pickle_loaded = True
            except Exception as e:
                self.pickle_loaded = False
                print(e)
                load_av = False

        if not self.pickle_loaded:
            self.av = collections.defaultdict(self.module_default_dict)

    def module_default_dict(self):  # pickle cant save lambdas
        return collections.defaultdict(int)

    def updateActionValue(self, qtuple):
        """
        qtuple holds:
            current state: board - 3x3 array
            action taken: indices - [i,j]
            reward obtained
            next state: board - 3x3 array
        """

        current = qtuple["cur"].tobytes()  # serialize it
        action = qtuple["act"] if len(qtuple["act"]) == 2 else qtuple["act"][:2]
        reward = qtuple["rew"]
        nextState = qtuple["nex"].tobytes()

        qmax = (
            max(self.av[(nextState, self.symbol)].values())
            if self.av[(nextState, self.symbol)].values()
            else 0
        )  # if next_state is not present in av, then return 0
        self.av[(current, self.symbol)][tuple(action)] += self.alpha * (
            reward + self.gamma * qmax - self.av[(current, self.symbol)][tuple(action)]
        )

        self.saveAVFreq -= 1
        if self.save_av and self.saveAVFreq == 0:
            with open("/kaggle/working/av.pickle", "wb") as f:
                pickle.dump(self.av, f)
            self.saveAVFreq = 100

    def action(self, timeStep):
        board = timeStep.observation
        qsa = dict(self.av[(board.tobytes(), self.symbol)])

        was_random_choice = False
        empty_slots = []  # find the list of actions possible
        for i in range(0, 3):
            for j in range(0, 3):
                if board[(i, j)] == 0:
                    empty_slots.append([i, j])

        while (
            qsa and max(qsa.values()) >= 0
        ):  # choose from the existing options only if it has a non -ve value
            act = max(qsa, key=qsa.get)  # type: ignore
            if board[tuple(act)] == 0:
                break
            qsa.pop(act, None)

        else:
            was_random_choice = True
            act = random.choice(empty_slots)  # else chose randomly

        return tf_agents.trajectories.PolicyStep(
            action=list(act) + [self.symbol], state=board, info=was_random_choice
        )

    def displayAV(self):
        print("***AV***")
        for k, v in dict(self.av).items():
            board = np.ndarray((3, 3), np.int32, k[0])
            print(board, k[1])
            print(v)

    def save_agent(self, file_path):
        """
        Save the Q-learning agent to a file using pickle.

        Arguments:
            file_path: File path to save the agent
        """
        try:
            with open(file_path, "wb") as f:
                pickle.dump(self, f)
            print(f"Agent saved successfully to {file_path}")
        except Exception as e:
            print(f"Error saving agent: {e}")

    def load(self, file_path):
        """
        Load a Q-learning agent from a file using pickle.

        Arguments:
            file_path: File path to load the agent
        """
        try:
            with open(file_path, "rb") as f:
                agent = pickle.load(f)
            print(f"Agent loaded successfully from {file_path}")
            return agent
        except Exception as e:
            print(f"Error loading agent: {e}")
            return None
