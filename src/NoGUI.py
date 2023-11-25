from GymTTT import GymTTT
from QLearningAgent import QLearningAgent
from HumanTTTAgent import HumanTTTAgent

# Create an instance of QLearningAgent
agent = QLearningAgent(symbol=1, trainable=False).load("./agents/dql/difficult/hard_agent.pickle")
player = HumanTTTAgent(symbol=2)

# Create an instance of GymTTT with the QLearningAgent
gym = GymTTT(agent, verbose=True)

# Reset the game to initialize it
timeStep = gym.reset()

# Game loop
while not timeStep.is_last():
    # Player's move
    # print("Enter your move:")
    action = player.action(timeStep).action
    timeStep = gym.step(action)

    if timeStep.is_last():
        break

# Display the final result
if timeStep.reward == 0:
    print("Game Over: Draw")
elif timeStep.reward == -1:
    print("Invalid move: Gym Agent Wins")
else:
    print("Yay!: Test Agent Wins")
