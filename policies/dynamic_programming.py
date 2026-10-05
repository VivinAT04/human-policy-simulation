"""Dynamic Programming policy using Value Iteration."""

from __future__ import annotations

from human_policy_sim.environment import (
    Action,
    GridEnvironment,
    Heading,
    State,
)


class ValueIterationPolicy:
    """
    Compute an optimal policy for the deterministic wheelchair grid.

    State:
        (row, col, heading)

    Actions:
        LEFT, RIGHT, FORWARD, STOP

    Bellman optimality equation:

        V(s) = max_a [ R(s,a,s') + gamma * V(s') ]

    The resulting policy maps:

        state -> optimal action
    """

    def __init__(
        self,
        env: GridEnvironment,
        gamma: float = 0.99,
        theta: float = 1e-10,
        max_iterations: int = 10_000,
    ):
        if not 0.0 <= gamma < 1.0:
            raise ValueError("gamma must satisfy 0 <= gamma < 1")

        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.max_iterations = max_iterations

        self.actions = (
            Action.LEFT,
            Action.RIGHT,
            Action.FORWARD,
            Action.STOP,
        )

        self.states = self._build_state_space()

        self.values = {
            state: 0.0
            for state in self.states
        }

        self.policy: dict[State, Action] = {}

        self.iterations = 0

    def _build_state_space(self) -> list[State]:
        """Enumerate every possible grid state."""

        return [
            State(row, col, heading)
            for row in range(self.env.rows)
            for col in range(self.env.cols)
            for heading in Heading
        ]

    def _action_value(
        self,
        state: State,
        action: Action,
    ) -> float:
        """Calculate Q(s,a) using the environment transition."""

        result = self.env.step(state, action)

        if result.terminated:
            return result.reward

        return (
            result.reward
            + self.gamma * self.values[result.state]
        )

    def solve(self) -> None:
        """Run Value Iteration and extract the optimal policy."""

        for iteration in range(1, self.max_iterations + 1):

            delta = 0.0

            new_values = self.values.copy()

            for state in self.states:

                # Goal states are terminal.
                if self.env.is_goal(state):
                    new_values[state] = 0.0
                    continue

                action_values = [
                    self._action_value(state, action)
                    for action in self.actions
                ]

                best_value = max(action_values)

                delta = max(
                    delta,
                    abs(best_value - self.values[state]),
                )

                new_values[state] = best_value

            self.values = new_values
            self.iterations = iteration

            if delta < self.theta:
                break

        # ----------------------------------------------------
        # Extract deterministic optimal policy
        # ----------------------------------------------------

        for state in self.states:

            if self.env.is_goal(state):
                self.policy[state] = Action.STOP
                continue

            best_action = max(
                self.actions,
                key=lambda action: self._action_value(
                    state,
                    action,
                ),
            )

            self.policy[state] = best_action

    def action(self, state: State) -> Action:
        """Return the optimal action for a state."""

        if not self.policy:
            raise RuntimeError(
                "Policy has not been solved. Call solve() first."
            )

        return self.policy[state]

    def value(self, state: State) -> float:
        """Return V(s)."""

        return self.values[state]
