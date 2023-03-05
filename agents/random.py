import torch

from .base import BaseConnectFourAgent


class RandomConnectFourAgent(BaseConnectFourAgent):
    def action(self, state: torch.Tensor) -> torch.Tensor:
        return torch.rand((self.columns), device=self.device).argsort()
