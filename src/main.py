import collections
import os
import time
import matplotlib.pyplot as plt
from numpy import save
import tensorflow as tf

from DPMinMaxAgent import DPMinMaxAgent
from MinMaxAgent import MinMaxAgent
from RandomTTTAgent import RandomTTTAgent
from GymTTT import GymTTT
from QLearningAgent import QLearningAgent

def main():
    t = time.time()

    TestAgent = DPMinMaxAgent(1)  # Agent 1
    gymAgent = RandomTTTAgent(2)  # Agent 2

    gym = GymTTT(gymAgent, True)

    timeStep = gym.reset()

    while not bool(timeStep.is_last()):
        timeStep = gym.step(TestAgent.action(timeStep).action)

    print(time.time() - t)

    # Sanity test against a random agent
    t = time.time()

    TestAgent = QLearningAgent(1)  # Agent 1
    gymAgent = RandomTTTAgent(2)  # Agent 2

    gym = GymTTT(gymAgent, True)

    timeStep = gym.reset()

    while not bool(timeStep.is_last()):
        timeStep = gym.step(TestAgent.action(timeStep).action)

    print(time.time() - t)

    # stats
    results = []
    random_actions = []

    # initialize the agent to be train
    TestAgent = QLearningAgent(1)

    # initialize gym
    gymAgent = RandomTTTAgent(2)
    gym = GymTTT(gymAgent, False)

    start = time.time()
    games = 5000  # episodes
    for i in range(games):
        timeStep = gym.reset()
        random_action_count = 0
        while not bool(timeStep.is_last()):
            policyStep = TestAgent.action(timeStep)
            nexTimeStep = gym.step(policyStep.action)

            if bool(policyStep.info):
                random_action_count += 1

            qtuple = {}
            qtuple["cur"] = timeStep.observation
            qtuple["act"] = policyStep.action[:2] # type: ignore
            qtuple["rew"] = nexTimeStep.reward
            qtuple["nex"] = nexTimeStep.observation

            if TestAgent.trainable:
                TestAgent.updateActionValue(qtuple)

            timeStep = nexTimeStep

        results.append(timeStep.reward)  # collect stats
        random_actions.append(random_action_count)

    print(len(results))
    print(time.time() - start)
    print(
        "Win Percentage: ", results[-games:].count(1) / games
    )  # stats for last x games
    print("Draw Percentage: ", results[-games:].count(0) / games)
    print("Loss Percentage: ", results[-games:].count(-1) / games)

    freq = 50  # bucket-size for plotting higher sizes give smoother curves

    a, b = 0, len(
        results
    )  # range to plot; (change a to len(results)-x to plot only the last x)

    metrics = collections.defaultdict(list)

    for i in range(a, b, freq):
        if i == 0:
            continue

        metrics["games"].append(i)
        metrics["wins"].append(results[i - freq : i].count(1))
        metrics["draws"].append(results[i - freq : i].count(0))
        metrics["loses"].append(results[i - freq : i].count(-1))
        metrics["win_pct"].append(results[i - freq : i].count(1) / float(freq))

    metrics["randomness"] = [
        sum(random_actions[i - freq : i]) for i in range(a, b, freq) if i != 0
    ]  # how many of the actions were just random(by a q agent)

    plt.plot(metrics["games"], metrics["wins"], label="wins")
    # plt.plot(metrics['games'], metrics['win_pct'], label='wins_pct')
    plt.plot(metrics["games"], metrics["draws"], label="draws")
    plt.plot(metrics["games"], metrics["loses"], label="loses")
    # plt.plot(metrics['games'], metrics['randomness'], label='randomness')
    plt.legend(loc=0)
    plt.show()

    # save the agent
    save_path = "./agents/dql/difficult/hard_agent.pickle"
    TestAgent.save_agent(file_path=save_path)

if __name__ == "__main__":
    main()