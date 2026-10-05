"""Common grid-world environment for human-policy experiments."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Heading(Enum):
    NORTH = 0
    EAST = 90
    SOUTH = 180
    WEST = 270


class Action(Enum):
    LEFT = "left"
    RIGHT = "right"
    FORWARD = "forward"
    STOP = "stop"


@dataclass(frozen=True)
class State:
    row: int
    col: int
    heading: Heading


@dataclass(frozen=True)
class StepResult:
    state: State
    reward: float
    terminated: bool
    blocked: bool


class GridEnvironment:
    """Deterministic wheelchair grid environment."""

    OFFSETS = {
        Heading.NORTH: (-1, 0),
        Heading.EAST: (0, 1),
        Heading.SOUTH: (1, 0),
        Heading.WEST: (0, -1),
    }

    def __init__(
        self,
        rows: int = 20,
        cols: int = 20,
        goal: tuple[int, int] = (1, 18),
    ):
        if rows <= 0 or cols <= 0:
            raise ValueError("rows and cols must be positive")

        if not (0 <= goal[0] < rows and 0 <= goal[1] < cols):
            raise ValueError("goal must lie inside the grid")

        self.rows = rows
        self.cols = cols
        self.goal = goal

    def is_valid(self, row: int, col: int) -> bool:
        return 0 <= row < self.rows and 0 <= col < self.cols

    def is_goal(self, state: State) -> bool:
        return (state.row, state.col) == self.goal

    def step(self, state: State, action: Action) -> StepResult:

        if self.is_goal(state):
            return StepResult(
                state=state,
                reward=0.0,
                terminated=True,
                blocked=False,
            )

        blocked = False

        if action == Action.LEFT:
            heading = Heading((state.heading.value - 90) % 360)
            next_state = State(state.row, state.col, heading)

        elif action == Action.RIGHT:
            heading = Heading((state.heading.value + 90) % 360)
            next_state = State(state.row, state.col, heading)

        elif action == Action.STOP:
            next_state = state

        elif action == Action.FORWARD:
            dr, dc = self.OFFSETS[state.heading]

            new_row = state.row + dr
            new_col = state.col + dc

            if self.is_valid(new_row, new_col):
                next_state = State(
                    new_row,
                    new_col,
                    state.heading,
                )
            else:
                next_state = state
                blocked = True

        else:
            raise ValueError(f"Unknown action: {action}")

        terminated = self.is_goal(next_state)

        if terminated:
            reward = 100.0
        elif blocked:
            reward = -5.0
        else:
            reward = -1.0

        return StepResult(
            state=next_state,
            reward=reward,
            terminated=terminated,
            blocked=blocked,
        )
