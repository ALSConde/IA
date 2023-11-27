from GymTTT import GymTTT
from QLearningAgent import QLearningAgent
from HumanTTTAgent import HumanTTTAgent
from MinMaxAgent import MinMaxAgent

# Create an instance of QLearningAgent
agent = QLearningAgent(1, False).load("./agents/dql/difficult/hard_agent.pickle")
player = MinMaxAgent(symbol=2)

# Create an instance of GymTTT with the QLearningAgent
gym = GymTTT(agent, verbose=True)

draw = 0.0
win = 0.0
lose = 0.0
games = 100
# Game loop
for i in range(0, games):
    # Reset the game to initialize it
    timeStep = gym.reset()
    while not bool(timeStep.is_last()):
        # Player's move
        action = player.action(timeStep).action
        timeStep = gym.step(action)
        
        if bool(timeStep.is_last()):
            # Display the final result
            if timeStep.reward == 0:
                draw += 1
            elif timeStep.reward == -1:
                win += 1
            else:
                lose += 1
            break

print(f"Draw ratio: {draw/games}")
print(f"Win ratio: {win/games}")
print(f"Lose ratio: {lose/games}")