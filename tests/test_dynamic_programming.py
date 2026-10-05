from human_policy_sim.environment import (
    Action,
    GridEnvironment,
    Heading,
    State,
)
from policies.dynamic_programming import ValueIterationPolicy


def build_policy():
    env = GridEnvironment(
        rows=20,
        cols=20,
        goal=(1, 18),
    )

    policy = ValueIterationPolicy(env)
    policy.solve()

    return env, policy


def test_policy_created():
    env, policy = build_policy()

    expected_states = (
        env.rows
        * env.cols
        * len(Heading)
    )

    assert len(policy.policy) == expected_states


def test_goal_returns_stop():
    _, policy = build_policy()

    goal_state = State(
        1,
        18,
        Heading.NORTH,
    )

    assert policy.action(goal_state) == Action.STOP


def test_policy_reaches_goal():
    env, policy = build_policy()

    state = State(
        18,
        1,
        Heading.NORTH,
    )

    for _ in range(200):

        if env.is_goal(state):
            break

        action = policy.action(state)
        result = env.step(state, action)

        state = result.state

    assert env.is_goal(state)


def test_policy_does_not_get_stuck():
    env, policy = build_policy()

    state = State(
        18,
        1,
        Heading.NORTH,
    )

    visited = []

    for _ in range(200):

        if env.is_goal(state):
            break

        visited.append(state)

        action = policy.action(state)
        state = env.step(state, action).state

    assert env.is_goal(state)

    # The deterministic optimal policy should not loop.
    assert len(visited) == len(set(visited))
