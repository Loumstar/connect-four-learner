from typing import Tuple

import arcade

from . import agents
from .environment import ConnectFourEnvironment


class ConnectFourSimulation(arcade.Window):
    SCREEN_TITLE = "Connect Four RL Agent"
    UPDATE_RATE = 1

    BORDER_THICKNESS = 2

    BACKGROUND_COLOR = (0, 0, 0)
    BORDER_COLOUR = (128, 128, 128)
    AGENT_COLOUR = (0, 230, 118)
    ENEMY_COLOUR = (255, 23, 68)

    def __init__(
        self,
        environment: ConnectFourEnvironment,
        agent: agents.ConnectFourAgents,
        opponent: agents.ConnectFourAgents,
    ) -> None:
        super().__init__(
            width=int(environment.width),
            height=int(environment.height),
            update_rate=self.UPDATE_RATE,
        )  # type: ignore

        arcade.set_background_color(self.BACKGROUND_COLOR)

        self.environment = environment
        self.agent = agent
        self.opponent = opponent

        self.time_step = 0

        self.cell_size = self.environment.cell_size
        self.coin_size = 0.4 * self.cell_size

    def cell_coord(self, row: int, column: int) -> Tuple[float, float]:
        x = self.cell_size * (column + 0.5)
        y = self.cell_size * (row + 0.5)

        return x, y

    def draw_board(self) -> None:
        for i in range(self.environment.rows + 1):
            y = (self.cell_size * i) - (self.BORDER_THICKNESS // 2)
            arcade.draw_line(
                0,
                y,
                self.environment.width,
                y,
                color=self.BORDER_COLOUR,
                line_width=self.BORDER_THICKNESS,
            )

        for i in range(self.environment.columns + 1):
            x = (self.cell_size * i) - (self.BORDER_THICKNESS // 2)
            arcade.draw_line(
                x,
                0,
                x,
                self.environment.height,
                color=self.BORDER_COLOUR,
                line_width=self.BORDER_THICKNESS,
            )

    def on_update(self, delta_time: int) -> None:
        if self.environment.finished:
            return

        state = self.environment.state

        if self.time_step % 2 == 0:
            actions = self.agent.action(state)
            self.environment.update(actions, 1)
        else:
            actions = self.opponent.action(state)
            self.environment.update(actions, -1)

        self.time_step += 1

    def on_draw(self) -> None:
        if self.environment.finished:
            arcade.exit()

        arcade.start_render()
        self.draw_board()

        for cell, player in self.environment.history:
            x, y = self.cell_coord(*cell)
            color = self.AGENT_COLOUR if player > 0 else self.ENEMY_COLOUR
            arcade.draw_circle_filled(x, y, radius=self.coin_size, color=color)
