from human_policy_sim.environment import (
    Action,
    GridEnvironment,
    Heading,
    State,
)
from policies.mcts import MCTSPolicy


def test_goal_returns_stop():

    env = GridEnvironment(
        rows=5,
        cols=5,
        goal=(1, 3),
    )

    policy = MCTSPolicy(
        env,
        simulations=20,
        seed=42,
    )

    state = State(
        1,
        3,
        Heading.NORTH,
    )

    assert policy.action(state) == Action.STOP


def test_returns_valid_action():

    env = GridEnvironment(
        rows=5,
        cols=5,
        goal=(1, 3),
    )

    policy = MCTSPolicy(
        env,
        simulations=50,
        seed=42,
    )

    state = State(
        4,
        0,
        Heading.NORTH,
    )

    action = policy.action(state)

    assert action in tuple(Action)


def test_records_planning_information():

    env = GridEnvironment(
        rows=5,
        cols=5,
        goal=(1, 3),
    )

    policy = MCTSPolicy(
        env,
        simulations=30,
        seed=42,
    )

    state = State(
        4,
        0,
        Heading.NORTH,
    )

    policy.action(state)

    assert policy.last_simulations == 30
    assert policy.last_planning_time >= 0.0
