from TicTacToe import TicTacToe


class GymTTT:
    """
    It is a wrapper around the tictactoe env.
    It takes an agent as argument which will be used to choose the action

    Arguments:
        agent: instance of an agent
        verbose: [True|False] want messages printed to console output?

    Usage:
        gym = GymTTT(Agent(),True)
    """

    def __init__(self, agent, verbose=False):
        self.agent = agent
        self.env = TicTacToe()
        self.verbose = verbose

    def reset(self):
        if (
            self.agent.symbol == 1
        ):  # if gym's agent has to start, then make the first move
            timeStep = self.env.reset()
            policyStep = self.agent.action(timeStep)
            timeStep = self.env.step(policyStep.action)
            timeStep = timeStep._replace(reward=timeStep.reward * -1)  # type: ignore # invert reward

            self.print_message("\n********GYM STARTS********")
            self.print_message("GYM Agent Move: " + str(policyStep.action[:2]))
            self.fancy_display(timeStep.observation)
            return timeStep

        else:
            self.print_message("\n********PLAYER STARTS********")
            return self.env.reset()

    def step(self, action):
        # player's move
        timeStep = self.env.step(action)
        self.print_message("Player Move: " + str(action[:2]))
        self.fancy_display(timeStep.observation)

        # if player's move was last
        if bool(timeStep.is_last()):
            if timeStep.reward == 0:
                self.print_message("***Game Over: Draw***")
            elif timeStep.reward == -1:
                self.print_message("***Invalid move: Gym Agent Wins")
            else:
                self.print_message("*** Yay!: Test Agent Wins ***")

            return timeStep

        else:
            # agent's move
            policyStep = self.agent.action(timeStep)
            timeStep = self.env.step(policyStep.action)
            timeStep = timeStep._replace(
                reward=timeStep.reward * -1 # type: ignore
            )  # invert agent's reward for player

            self.print_message("GYM Agent Move: " + str(policyStep.action[:2]))
            self.fancy_display(timeStep.observation)

            if bool(timeStep.is_last()):
                if timeStep.reward == 0:
                    self.print_message("***Game Over: Draw***")
                else:
                    self.print_message("***Yay!: Gym Agent Wins ***")

            return timeStep

    def print_message(self, message):
        if self.verbose:
            print(message)

    def fancy_display(self, board, action=[-1, -1]):
        # array representation was good enough
        if self.verbose:
            print(board)
