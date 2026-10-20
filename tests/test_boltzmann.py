import numpy as np
import pytest

from human_policy_sim.environment import GridEnvironment, State, Heading
from policies.dynamic_programming import ValueIterationPolicy
from policies.boltzmann import BoltzmannPolicy


@pytest.fixture(scope="module")
def base_policy():
    env = GridEnvironment(
        rows=20,
        cols=20,
        goal=(1, 18),
    )
    policy = ValueIterationPolicy(env=env)
    policy.solve()
    return policy


def test_invalid_temperature(base_policy):
    for value in [0, -1, float("nan"), float("inf")]:
        with pytest.raises(ValueError):
            BoltzmannPolicy(base_policy, temperature=value)


def test_probabilities_sum_to_one(base_policy):
    policy = BoltzmannPolicy(base_policy, temperature=1.0)
    state = State(18, 1, Heading.NORTH)

    probabilities = policy.probabilities(state)

    assert len(probabilities) == 4
    assert np.all(probabilities >= 0)
    assert np.isclose(probabilities.sum(), 1.0)


def test_low_temperature_prefers_best_action(base_policy):
    policy = BoltzmannPolicy(base_policy, temperature=0.1)
    state = State(18, 1, Heading.NORTH)

    q = policy.action_values(state)
    probabilities = policy.probabilities(state)

    assert np.argmax(probabilities) == np.argmax(q)


def test_high_temperature_more_uniform(base_policy):
    state = State(18, 1, Heading.NORTH)

    low = BoltzmannPolicy(
        base_policy,
        temperature=0.1,
    ).probabilities(state)

    high = BoltzmannPolicy(
        base_policy,
        temperature=5.0,
    ).probabilities(state)

    assert high.max() - high.min() < low.max() - low.min()


def test_reproducibility(base_policy):
    state = State(18, 1, Heading.NORTH)

    a = BoltzmannPolicy(base_policy, temperature=2.0, seed=123)
    b = BoltzmannPolicy(base_policy, temperature=2.0, seed=123)

    assert [a.action(state) for _ in range(100)] == [
        b.action(state) for _ in range(100)
    ]
