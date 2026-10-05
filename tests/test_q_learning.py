import numpy as np

from human_policy_sim.environment import (
    Action,
    GridEnvironment,
    Heading,
    State,
)
from policies.q_learning import QLearningPolicy


def test_q_table_shape():
    env = GridEnvironment()

    agent = QLearningPolicy(env)

    assert agent.q_table.shape == (
        20,
        20,
        4,
        4,
    )


def test_goal_returns_stop():
    env = GridEnvironment(
        goal=(1, 18),
    )

    agent = QLearningPolicy(env)

    state = State(
        1,
        18,
        Heading.NORTH,
    )

    assert agent.action(state) == Action.STOP


def test_q_update_occurs():
    env = GridEnvironment(
        rows=3,
        cols=3,
        goal=(0, 2),
    )

    agent = QLearningPolicy(
        env=env,
        seed=42,
    )

    before = agent.q_table.copy()

    agent.train(
        episodes=100,
        max_steps=50,
    )

    assert not np.array_equal(
        before,
        agent.q_table,
    )


def test_save_and_load(tmp_path):
    env = GridEnvironment()

    agent = QLearningPolicy(env)

    agent.q_table[0, 0, 0, 0] = 123.0

    path = tmp_path / "q_table.npy"

    agent.save(str(path))

    loaded = QLearningPolicy(env)
    loaded.load(str(path))

    assert loaded.q_table[
        0, 0, 0, 0
    ] == 123.0
