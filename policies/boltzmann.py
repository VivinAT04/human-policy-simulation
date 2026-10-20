"""Boltzmann (softmax) action selection over fixed action values."""

import numpy as np


class BoltzmannPolicy:
    def __init__(self, base_policy, temperature=1.0, seed=42):
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("Temperature must be finite and positive")

        self.base_policy = base_policy
        self.temperature = float(temperature)
        self.rng = np.random.default_rng(seed)

    def action_values(self, state):
        """
        Obtain Q(s,a) from the fixed value-iteration solution.
        The underlying value function is not changed by temperature.
        """
        env = self.base_policy.env
        gamma = self.base_policy.gamma

        values = []
        for action in self.base_policy.actions:
            result = env.step(state, action)
            future = (
                0.0
                if result.terminated
                else self.base_policy.values[result.state]
            )
            values.append(result.reward + gamma * future)

        return np.asarray(values, dtype=float)

    def probabilities(self, state):
        q = self.action_values(state)

        # Numerically stable softmax
        scaled = (q - np.max(q)) / self.temperature
        weights = np.exp(scaled)
        return weights / weights.sum()

    def action(self, state):
        actions = self.base_policy.actions
        probabilities = self.probabilities(state)

        index = self.rng.choice(
            len(actions),
            p=probabilities,
        )
        return actions[index]
