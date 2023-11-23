import tf_agents


class HumanTTTAgent:
    def __init__(self, symbol, verbose=False):
        self.symbol = symbol
        self.verbose = verbose
        self.trainable = False

    def action(self, timeStep):
        board = timeStep.observation

        empty_slots = []
        for i in range(0, 3):
            for j in range(0, 3):
                if board[(i, j)] == 0:
                    empty_slots.append([i, j])

        print(board)
        print("input space separated indices; choose from the empty slots")
        print("EmptySlots: ", empty_slots)
        i, j = self.get_inputs()
        tries = 2
        while [i, j] not in empty_slots and tries:
            print("invalid choice, input space seperated indices, tries left: ", tries)
            i, j = self.get_inputs()
            tries -= 1

        if not tries:
            print("Illiterate")

        act = [i, j, self.symbol]
        return tf_agents.trajectories.PolicyStep(
            action=act, state=board, info=self.symbol
        )

    def get_inputs(self):
        try:
            i, j = [int(x) for x in input().split()]
            return (i, j)
        except:
            return (9, 9)

    def updateActionValue(self, qtuple):
        pass
