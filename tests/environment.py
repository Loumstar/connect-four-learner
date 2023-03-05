import unittest

import torch

from ..environment import ConnectFourEnvironment


class EnvironmentTests(unittest.TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.env = ConnectFourEnvironment(6, 7, 4, 100)

    def test_horizontal_win(self):
        state = torch.zeros((6, 7))
        state[1, 2:6] = 1

        finished, winner = self.env.check_winner(state)

        self.assertTrue(finished)
        self.assertEqual(winner, 1)

    def test_vertical_win(self):
        state = torch.zeros((6, 7))
        state[2:6, 2] = 1

        finished, winner = self.env.check_winner(state)

        self.assertTrue(finished)
        self.assertEqual(winner, 1)

    def test_forward_diagonal_win(self):
        state = torch.zeros((6, 7))
        diagonal = torch.diagflat(torch.ones(4))
        state[2:6, 2:6] = diagonal

        finished, winner = self.env.check_winner(state)

        self.assertTrue(finished)
        self.assertEqual(winner, 1)

    def test_backward_diagonal_win(self):
        state = torch.zeros((6, 7))
        diagonal = torch.diagflat(torch.ones(4)).fliplr()
        state[2:6, 2:6] = diagonal

        finished, winner = self.env.check_winner(state)

        self.assertTrue(finished)
        self.assertEqual(winner, 1)


if __name__ == "__main__":
    unittest.main()
