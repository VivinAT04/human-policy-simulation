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


def test_combine_returns_discounts_rollout_once():
    """Rollout return must not be double-discounted."""

    env = GridEnvironment(
        rows=5,
        cols=5,
        goal=(1, 3),
    )

    policy = MCTSPolicy(
        env,
        simulations=20,
        gamma=0.99,
        seed=42,
    )

    result = policy._combine_returns(
        path_rewards=[-1.0, -1.0],
        rollout_return=100.0,
    )

    expected = (
        -1.0
        + 0.99 * (
            -1.0
            + 0.99 * 100.0
        )
    )

    assert abs(result - expected) < 1e-12


def test_combine_returns_without_tree_path():
    """No tree path means rollout return is unchanged."""

    env = GridEnvironment(
        rows=5,
        cols=5,
        goal=(1, 3),
    )

    policy = MCTSPolicy(
        env,
        simulations=20,
        gamma=0.99,
        seed=42,
    )

    result = policy._combine_returns(
        path_rewards=[],
        rollout_return=42.5,
    )

    assert result == 42.5


def test_combine_returns_single_transition():
    """One tree transition applies gamma exactly once."""

    env = GridEnvironment(
        rows=5,
        cols=5,
        goal=(1, 3),
    )

    policy = MCTSPolicy(
        env,
        simulations=20,
        gamma=0.9,
        seed=42,
    )

    result = policy._combine_returns(
        path_rewards=[-1.0],
        rollout_return=10.0,
    )

    assert abs(
        result - 8.0
    ) < 1e-12




def test_backpropagate_stores_action_returns():
    """
    Child statistics represent the action used to reach
    that child, including its immediate transition reward.
    """

    env = GridEnvironment(
        rows=5,
        cols=5,
        goal=(1, 3),
    )

    policy = MCTSPolicy(
        env,
        simulations=20,
        gamma=0.99,
        seed=42,
    )

    root = policy._new_node(
        State(
            4,
            0,
            Heading.NORTH,
        )
    )

    child = policy._new_node(
        State(
            3,
            0,
            Heading.NORTH,
        ),
        parent=root,
        action_from_parent=Action.FORWARD,
    )

    leaf = policy._new_node(
        State(
            2,
            0,
            Heading.NORTH,
        ),
        parent=child,
        action_from_parent=Action.FORWARD,
    )

    policy._backpropagate(
        leaf,
        path_rewards=[
            -1.0,
            -1.0,
        ],
        rollout_return=100.0,
    )

    leaf_action_return = (
        -1.0
        + 0.99 * 100.0
    )

    child_action_return = (
        -1.0
        + 0.99 * leaf_action_return
    )

    assert leaf.visits == 1
    assert child.visits == 1
    assert root.visits == 1

    assert abs(
        leaf.total_reward
        - leaf_action_return
    ) < 1e-12

    assert abs(
        child.total_reward
        - child_action_return
    ) < 1e-12


def test_terminal_goal_action_keeps_goal_reward():
    """
    An action that reaches the goal must retain its +100
    transition reward in the corresponding child value.
    """

    env = GridEnvironment(
        rows=20,
        cols=20,
        goal=(1, 18),
    )

    policy = MCTSPolicy(
        env,
        simulations=20,
        gamma=0.99,
        seed=42,
    )

    root = policy._new_node(
        State(
            1,
            17,
            Heading.EAST,
        )
    )

    result = env.step(
        root.state,
        Action.FORWARD,
    )

    assert result.terminated
    assert result.reward == 100.0

    goal_child = policy._new_node(
        result.state,
        parent=root,
        action_from_parent=Action.FORWARD,
    )

    policy._backpropagate(
        goal_child,
        path_rewards=[result.reward],
        rollout_return=0.0,
    )

    assert goal_child.visits == 1
    assert goal_child.mean_reward == 100.0


def test_backpropagate_single_action_return():
    env = GridEnvironment(
        rows=5,
        cols=5,
        goal=(1, 3),
    )

    policy = MCTSPolicy(
        env,
        simulations=20,
        gamma=0.9,
        seed=42,
    )

    root = policy._new_node(
        State(
            4,
            0,
            Heading.NORTH,
        )
    )

    child = policy._new_node(
        State(
            3,
            0,
            Heading.NORTH,
        ),
        parent=root,
        action_from_parent=Action.FORWARD,
    )

    policy._backpropagate(
        child,
        path_rewards=[-1.0],
        rollout_return=10.0,
    )

    assert abs(
        child.mean_reward - 8.0
    ) < 1e-12


def test_backpropagate_rejects_inconsistent_path():
    env = GridEnvironment(
        rows=5,
        cols=5,
        goal=(1, 3),
    )

    policy = MCTSPolicy(
        env,
        simulations=20,
        gamma=0.99,
        seed=42,
    )

    root = policy._new_node(
        State(
            4,
            0,
            Heading.NORTH,
        )
    )

    child = policy._new_node(
        State(
            3,
            0,
            Heading.NORTH,
        ),
        parent=root,
        action_from_parent=Action.FORWARD,
    )

    try:
        policy._backpropagate(
            child,
            path_rewards=[],
            rollout_return=10.0,
        )

    except RuntimeError:
        pass

    else:
        raise AssertionError(
            "Expected inconsistent MCTS path to fail."
        )
