from human_policy_sim.environment import (
    Action,
    GridEnvironment,
    Heading,
    State,
)


def test_forward():
    env = GridEnvironment()
    state = State(10, 10, Heading.NORTH)

    result = env.step(state, Action.FORWARD)

    assert result.state == State(9, 10, Heading.NORTH)
    assert result.reward == -1.0
    assert not result.blocked


def test_left_turn():
    env = GridEnvironment()
    state = State(10, 10, Heading.NORTH)

    result = env.step(state, Action.LEFT)

    assert result.state == State(10, 10, Heading.WEST)


def test_right_turn():
    env = GridEnvironment()
    state = State(10, 10, Heading.NORTH)

    result = env.step(state, Action.RIGHT)

    assert result.state == State(10, 10, Heading.EAST)


def test_boundary_block():
    env = GridEnvironment()
    state = State(0, 5, Heading.NORTH)

    result = env.step(state, Action.FORWARD)

    assert result.state == state
    assert result.blocked
    assert result.reward == -5.0


def test_goal_reward():
    env = GridEnvironment(goal=(1, 18))
    state = State(2, 18, Heading.NORTH)

    result = env.step(state, Action.FORWARD)

    assert result.state == State(1, 18, Heading.NORTH)
    assert result.terminated
    assert result.reward == 100.0
