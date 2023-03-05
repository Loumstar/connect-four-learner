import random
from collections import deque
from typing import List, NamedTuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

from ..environment import ConnectFourEnvironment
from .base import BaseConnectFourAgent


class ConvolutionalModel(nn.Module):
    def __init__(self, rows: int = 6, columns: int = 7) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=1),  # 1 x 6 x 7 -> 32 x 4 x 5
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3),  # 32 x 4 x 5 -> 64 x 2 x 3
            nn.ReLU(inplace=True),
        )

        self.fc = nn.Linear(64 * 2 * 3, columns)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        batch_size = state.size(0)
        out = self.layers(state.unsqueeze(1))

        return self.fc(out.reshape(batch_size, -1))


class FullyConnectedModel(nn.Module):
    def __init__(self, rows: int = 6, columns: int = 7) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(rows * columns, 50),
            nn.ReLU(inplace=True),
            nn.Linear(50, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 50),
            nn.ReLU(inplace=True),
            nn.Linear(50, columns),
            nn.ReLU(inplace=True),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        batch_size = state.size(0)
        state = state.reshape(batch_size, -1)

        return F.softmax(self.layers(state))


class Transition(NamedTuple):
    state: torch.Tensor
    action: torch.Tensor
    next_state: torch.Tensor
    reward: torch.Tensor


class ReplayBuffer(object):
    def __init__(self, capacity: int) -> None:
        self.memory = deque([], maxlen=capacity)

    def push(self, *args: torch.Tensor):
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)


class BaseEpsilon:
    def __init__(self, epsilon: float):
        self.epsilon = epsilon

    def __call__(self) -> float:
        return self.epsilon

    def __str__(self) -> str:
        name = self.__class__.__name__
        return f"{name}({self.epsilon:.3f})"

    def step(self, episode: int) -> None:
        raise NotImplementedError


class ConstantEpsilon(BaseEpsilon):
    def step(self, episode: int) -> None:
        pass


class LinearEpsilon(BaseEpsilon):
    def __init__(
        self, epsilon_start: float, epsilon_end: float, epsilon_rate: int
    ):
        super().__init__(epsilon_start)

        self.start = epsilon_start
        self.end = epsilon_end
        self.rate = epsilon_rate

    def step(self, episode: int) -> None:
        if episode < 0:
            raise ValueError("Episode cannot be negative.")

        proportion = max(0, (1 - episode / self.rate))
        decrement = (self.start - self.end) * proportion

        self.epsilon = self.end + decrement


class QLearningConnectFourAgent(BaseConnectFourAgent):
    def __init__(
        self,
        environment: ConnectFourEnvironment,
        epsilon: BaseEpsilon,
        memory_capacity: int = 1000,
        transition_reward: float = -0.2,
        terminal_reward: float = 100,
        illegal_move_reward: float = -2,
        batch_size: int = 16,
        discount: float = 0.9,
        lr: float = 1e-3,
        weight_decay: float = 1e-7,
        target_update: int = 100,
        adversary_update: int = 500,
        device: Union[torch.device, str] = "cpu",
    ) -> None:
        rows, columns = environment.rows, environment.columns
        super().__init__(rows, columns, device)

        self.env = environment

        self.policy = ConvolutionalModel(rows, columns).to(device)
        self.target = ConvolutionalModel(rows, columns).to(device)
        self.adversary = ConvolutionalModel(rows, columns).to(device)

        self.target.eval()
        self.adversary.eval()

        self.memory = ReplayBuffer(memory_capacity)
        self.batch_size = batch_size

        self.transition_reward = transition_reward
        self.terminal_reward = terminal_reward
        self.illegal_move_reward = illegal_move_reward

        self.epsilon = epsilon

        self.lr = lr
        self.weight_decay = weight_decay
        self.discount = discount

        self.target_update = target_update
        self.adversary_update = adversary_update

        self.optimiser = torch.optim.RMSprop(
            self.policy.parameters(), lr, weight_decay=weight_decay
        )

        self.loss = nn.MSELoss(reduction="sum")

    def action(self, state: torch.Tensor) -> torch.Tensor:
        return torch.argsort(
            self.policy(state.unsqueeze(0)), descending=True
        ).squeeze(0)

    def adversary_action(self, state: torch.Tensor) -> torch.Tensor:
        return torch.rand((self.columns), device=self.device).argsort()

        with torch.no_grad():
            return torch.argsort(
                self.adversary(state.unsqueeze(0)), descending=True
            ).squeeze(0)

    def __next_state_values(
        self, next_state: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        next_values = torch.zeros(self.batch_size, device=self.device)

        if not mask.any():
            return next_values

        actions = self.policy(next_state).argmax(1).detach()
        target_q = self.target(next_state).gather(1, actions.unsqueeze(1))
        next_values[mask] = target_q.squeeze(1)

        return next_values

    def __get_transition_batch(self) -> Transition:
        if len(self.memory) < self.batch_size:
            raise ValueError("Not enough samples to batch Replay Buffer.")

        transitions = self.memory.sample(self.batch_size)

        states = torch.stack([t.state for t in transitions])
        actions = torch.tensor([t.action for t in transitions])
        next_states = torch.stack([t.next_state for t in transitions])
        rewards = torch.tensor([t.reward for t in transitions])

        return Transition(states, actions, next_states, rewards)

    def episode(self) -> float:
        self.env.reset()
        rewards = 0

        while not self.env.finished:
            reward = torch.tensor(self.transition_reward)
            state = self.env.state.detach()

            actions = self.action(state)

            if random.random() < self.epsilon():
                actions = actions.gather(0, torch.randperm(actions.size(0)))

            illegal_moves, action = self.env.update(actions, 1)

            reward += self.illegal_move_reward * illegal_moves

            if not self.env.finished:
                adversary_actions = self.adversary_action(self.env.state)
                self.env.update(adversary_actions, -1)

            if self.env.winner != 0:
                reward += self.env.winner * self.terminal_reward

            self.memory.push(state, action, self.env.state, reward)

            rewards += float(reward)
            self.optimise()

        return float(rewards)

    def optimise(self) -> None:
        if len(self.memory) < self.batch_size:
            return

        self.optimiser.zero_grad()
        batch = self.__get_transition_batch()

        nf_mask = torch.tensor([s is not None for s in batch.next_state])

        nf_next_states = batch.next_state[nf_mask]

        with torch.no_grad():
            next_state_values = self.__next_state_values(
                nf_next_states, nf_mask
            )

        expected_q = (next_state_values * self.discount) + batch.reward
        predicted_q = (
            self.policy(batch.state)
            .gather(1, batch.action.unsqueeze(1))
            .squeeze(1)
        )

        loss = self.loss(expected_q, predicted_q)
        loss.backward()

        self.optimiser.step()

    def train(self, num_episodes: int) -> List[float]:
        self.policy.train()
        rewards = []

        pbar = tqdm.tqdm(range(num_episodes), desc="Training DQN Learner")

        for episode in pbar:
            self.epsilon.step(episode)

            reward = self.episode()
            rewards.append(reward)

            if episode % self.target_update == 0:
                self.target.load_state_dict(self.policy.state_dict())
            if episode % self.adversary_update == 0:
                self.adversary.load_state_dict(self.policy.state_dict())

            pbar.set_postfix(
                {"reward": sum(rewards[-20:]) / len(rewards[-20:])}
            )

        return rewards

    def save(self, path: str) -> None:
        torch.save(self.policy.state_dict(), path)

    def load(self, path: str) -> None:
        state_dict = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(state_dict)
