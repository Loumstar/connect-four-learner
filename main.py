import arcade
import matplotlib.pyplot as plt

from . import agents
from .agents.learner import ConstantEpsilon
from .environment import ConnectFourEnvironment
from .simulation import ConnectFourSimulation

rows, columns = 6, 7
winning_size = 4

environment = ConnectFourEnvironment(
    rows, columns, winning_size, screen_size=600
)

epsilon = ConstantEpsilon(0.1)
agent = agents.QLearningConnectFourAgent(
    environment,
    epsilon,
    batch_size=64,
    adversary_update=2000,
    target_update=500,
)

random = agents.RandomConnectFourAgent(rows, columns)

# agent.load("./q_learner.pt")
rewards = agent.train(10000)
agent.save("q_learner.pt")

environment.reset()

ConnectFourSimulation(environment, agent, agent)
arcade.run()


plt.plot(rewards)
plt.show()
