import pytest

from human_policy_sim.environment import Action
from human_policy_sim.noise import ActionNoise


def test_zero_noise_preserves_action():

    noise = ActionNoise(
        noise_probability=0.0,
        seed=42,
    )

    for action in Action:
        for _ in range(100):
            assert noise.apply(action) == action


def test_full_noise_always_changes_action():

    noise = ActionNoise(
        noise_probability=1.0,
        seed=42,
    )

    for action in Action:
        for _ in range(100):
            assert noise.apply(action) != action


def test_noise_is_reproducible_with_same_seed():

    first = ActionNoise(
        noise_probability=0.5,
        seed=42,
    )

    second = ActionNoise(
        noise_probability=0.5,
        seed=42,
    )

    sequence_a = [
        first.apply(Action.FORWARD)
        for _ in range(100)
    ]

    sequence_b = [
        second.apply(Action.FORWARD)
        for _ in range(100)
    ]

    assert sequence_a == sequence_b


def test_different_seed_changes_sequence():

    first = ActionNoise(
        noise_probability=0.5,
        seed=42,
    )

    second = ActionNoise(
        noise_probability=0.5,
        seed=43,
    )

    sequence_a = [
        first.apply(Action.FORWARD)
        for _ in range(100)
    ]

    sequence_b = [
        second.apply(Action.FORWARD)
        for _ in range(100)
    ]

    assert sequence_a != sequence_b


def test_invalid_noise_probability_rejected():

    with pytest.raises(ValueError):
        ActionNoise(
            noise_probability=-0.1
        )

    with pytest.raises(ValueError):
        ActionNoise(
            noise_probability=1.1
        )


def test_empirical_noise_rate_is_close_to_requested_rate():

    noise = ActionNoise(
        noise_probability=0.3,
        seed=42,
    )

    trials = 10000

    changed = sum(
        noise.apply(Action.FORWARD)
        != Action.FORWARD
        for _ in range(trials)
    )

    observed_rate = changed / trials

    assert 0.28 <= observed_rate <= 0.32
