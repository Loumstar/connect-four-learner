from typing import Union

from .learner import QLearningConnectFourAgent
from .random import RandomConnectFourAgent

ConnectFourAgents = Union[QLearningConnectFourAgent, RandomConnectFourAgent]
