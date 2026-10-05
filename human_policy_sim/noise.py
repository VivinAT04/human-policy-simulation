"""Action-noise model for physiological-control experiments."""

from __future__ import annotations

import random

from human_policy_sim.environment import Action


class ActionNoise:
    """
    Corrupt a policy action with probability ``noise_probability``.

    When corruption occurs, the intended action is replaced uniformly
    by one of the other available actions.
    """

    def __init__(
        self,
        noise_probability: float = 0.0,
        seed: int | None = None,
    ) -> None:

        if not 0.0 <= noise_probability <= 1.0:
            raise ValueError(
                "noise_probability must be between 0.0 and 1.0"
            )

        self.noise_probability = noise_probability
        self.rng = random.Random(seed)

        self.actions = tuple(Action)

    def apply(self, intended_action: Action) -> Action:
        """Return the action actually executed."""

        if self.noise_probability == 0.0:
            return intended_action

        if self.rng.random() >= self.noise_probability:
            return intended_action

        alternatives = [
            action
            for action in self.actions
            if action != intended_action
        ]

        return self.rng.choice(alternatives)
