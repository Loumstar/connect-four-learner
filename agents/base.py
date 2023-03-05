from typing import List, Union

import torch


class BaseConnectFourAgent:
    def __init__(
        self, rows: int, columns: int, device: Union[torch.device, str] = "cpu"
    ) -> None:
        self.rows = rows
        self.columns = columns
        self.device = device

    def action(self, state: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def update(self, state_dict: torch.Tensor) -> None:
        raise NotImplementedError

    def train(self, num_episodes: int) -> List[float]:
        raise NotImplementedError

    def save(self, path: str) -> None:
        raise NotImplementedError

    def load(self, path: str) -> None:
        raise NotImplementedError
