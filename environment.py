import functools
from typing import Literal, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


class ConnectFourEnvironment:
    def __init__(
        self,
        rows: int,
        columns: int,
        winning_size: int,
        screen_size: int,
        device: Union[torch.device, str] = "cpu",
    ) -> None:
        self.winning_size = winning_size
        self.columns = columns
        self.rows = rows

        if rows > columns:
            self.height = screen_size
            self.width = (screen_size * columns) / rows
            self.cell_size = screen_size / rows
        else:
            self.width = screen_size
            self.height = (screen_size * rows) / columns
            self.cell_size = screen_size / columns

        self.device = device

        self.state = torch.zeros((rows, columns)).to(device)
        self.available_columns = np.zeros(columns, dtype=int)

        self.history = []

        self.winner = 0
        self.finished = False

        diagonal = self.make_diagonal_filter(winning_size)

        self.diagonal_filter = torch.stack(
            (diagonal, diagonal.flip(0))
        ).unsqueeze(1)

        self.orthogonal_filter = (
            self.make_orthogonal_filter(winning_size)
            .unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(0)
        )

    @functools.cache
    def make_orthogonal_filter(self, size: int) -> torch.Tensor:
        return torch.ones(size, device=self.device)

    @functools.cache
    def make_diagonal_filter(self, size: int) -> torch.Tensor:
        return torch.diagflat(self.make_orthogonal_filter(size))

    def update(
        self, actions: torch.Tensor, agent: Literal[1, -1]
    ) -> Tuple[int, torch.Tensor]:
        for i, action in enumerate(actions):
            column = int(action)
            row = self.available_columns[column]

            if row >= self.rows:
                continue

            self.available_columns[column] += 1
            self.state[row, column] = agent
            self.history.append(((row, column), agent))

            self.finished, self.winner = self.check_winner(self.state)

            return i, action

        raise ValueError("Environment couldn't update with action.")

    def check_winner(self, state: torch.Tensor) -> Tuple[bool, int]:
        # check vertical wins
        state = state.unsqueeze(0)

        vertical = F.conv2d(state, self.orthogonal_filter)
        horizontal = F.conv2d(state, self.orthogonal_filter.mT)
        diagonals = F.conv2d(state, self.diagonal_filter)

        for image in (vertical, horizontal, diagonals):
            if image.max() / self.winning_size == 1:
                return True, 1
            elif image.min() / self.winning_size == -1:
                return True, -1

        if all(i >= self.rows for i in self.available_columns):
            return True, 0

        return False, 0

    def reset(self) -> None:
        self.state = torch.zeros((self.rows, self.columns)).to(self.device)
        self.available_columns = np.zeros(self.columns, dtype=int)
        self.history = []
        self.winner = 0
        self.finished = False
