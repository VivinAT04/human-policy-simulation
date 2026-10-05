"""Tabular Q-Learning policy for the wheelchair grid."""

from __future__ import annotations

import random

import numpy as np

from human_policy_sim.environment import (
    Action,
    GridEnvironment,
    Heading,
    State,
)


class QLearningPolicy:
    """
    Tabular Q-Learning agent.

    State:
        (row, col, heading)

    Actions:
        LEFT, RIGHT, FORWARD, STOP

    Update rule:

        Q(s,a) <- Q(s,a) + alpha *
        [r + gamma * max Q(s',a') - Q(s,a)]

    Exploration is used ONLY during training.

    Evaluation uses the frozen greedy policy.
    """

    def __init__(
        self,
        env: GridEnvironment,
        alpha: float = 0.2,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.02,
        epsilon_decay: float = 0.9995,
        seed: int = 42,
    ):
        self.env = env

        self.alpha = alpha
        self.gamma = gamma

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.seed = seed

        self.actions = (
            Action.LEFT,
            Action.RIGHT,
            Action.FORWARD,
            Action.STOP,
        )

        self.action_to_index = {
            action: index
            for index, action in enumerate(self.actions)
        }

        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        # rows × cols × headings × actions
        self.q_table = np.zeros(
            (
                env.rows,
                env.cols,
                len(Heading),
                len(self.actions),
            ),
            dtype=np.float64,
        )

        self.training_episodes = 0
        self.training_successes = 0

    @staticmethod
    def _heading_index(heading: Heading) -> int:
        return {
            Heading.NORTH: 0,
            Heading.EAST: 1,
            Heading.SOUTH: 2,
            Heading.WEST: 3,
        }[heading]

    def _q_values(self, state: State) -> np.ndarray:
        return self.q_table[
            state.row,
            state.col,
            self._heading_index(state.heading),
        ]

    def _random_state(self) -> State:
        """
        Generate a random non-goal training state.

        Training from many starts allows the Q-table to learn a general
        policy rather than memorising only the dissertation start point.
        """

        while True:

            row = self.rng.randrange(self.env.rows)
            col = self.rng.randrange(self.env.cols)
            heading = self.rng.choice(tuple(Heading))

            state = State(
                row=row,
                col=col,
                heading=heading,
            )

            if not self.env.is_goal(state):
                return state

    def _greedy_action_index(
        self,
        state: State,
        random_ties: bool = False,
    ) -> int:
        """
        Return the index of the highest-valued action.

        During training, equal Q-values can be broken randomly to avoid
        systematic bias.

        During evaluation, ties are deterministic for reproducibility.
        """

        q_values = self._q_values(state)

        max_value = np.max(q_values)

        candidates = np.flatnonzero(
            np.isclose(q_values, max_value)
        )

        if random_ties:
            return int(self.np_rng.choice(candidates))

        return int(candidates[0])

    def _training_action(
        self,
        state: State,
        epsilon: float,
    ) -> int:

        if self.rng.random() < epsilon:
            return self.rng.randrange(len(self.actions))

        return self._greedy_action_index(
            state,
            random_ties=True,
        )

    def train(
        self,
        episodes: int = 30_000,
        max_steps: int = 200,
    ) -> list[float]:
        """
        Train the Q-Learning policy.

        Returns:
            episode reward history
        """

        epsilon = self.epsilon_start

        reward_history: list[float] = []

        successes = 0

        for episode in range(episodes):

            state = self._random_state()

            episode_reward = 0.0

            for _ in range(max_steps):

                action_index = self._training_action(
                    state,
                    epsilon,
                )

                action = self.actions[action_index]

                result = self.env.step(
                    state,
                    action,
                )

                current_q = self.q_table[
                    state.row,
                    state.col,
                    self._heading_index(state.heading),
                    action_index,
                ]

                if result.terminated:
                    target = result.reward
                else:
                    target = (
                        result.reward
                        + self.gamma
                        * np.max(
                            self._q_values(result.state)
                        )
                    )

                updated_q = (
                    current_q
                    + self.alpha
                    * (target - current_q)
                )

                self.q_table[
                    state.row,
                    state.col,
                    self._heading_index(state.heading),
                    action_index,
                ] = updated_q

                episode_reward += result.reward

                state = result.state

                if result.terminated:
                    successes += 1
                    break

            reward_history.append(episode_reward)

            epsilon = max(
                self.epsilon_end,
                epsilon * self.epsilon_decay,
            )

        self.training_episodes += episodes
        self.training_successes = successes

        return reward_history

    def action(self, state: State) -> Action:
        """
        Return the frozen greedy action.

        No epsilon exploration is used here.
        """

        if self.env.is_goal(state):
            return Action.STOP

        action_index = self._greedy_action_index(
            state,
            random_ties=False,
        )

        return self.actions[action_index]

    def save(self, path: str) -> None:
        np.save(path, self.q_table)

    def load(self, path: str) -> None:
        table = np.load(path)

        if table.shape != self.q_table.shape:
            raise ValueError(
                "Loaded Q-table has incompatible shape."
            )

        self.q_table = table.astype(
            np.float64,
            copy=True,
        )
