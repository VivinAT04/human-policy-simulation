"""Generate browser trajectories from the real policy implementations."""

from __future__ import annotations

import json
from pathlib import Path

from human_policy_sim.environment import (
    GridEnvironment,
    Heading,
    State,
)
from policies.dynamic_programming import ValueIterationPolicy
from policies.q_learning import QLearningPolicy
from policies.mcts import MCTSPolicy


ROWS = 20
COLS = 20

START = (18, 1)
GOAL = (1, 18)
START_HEADING = Heading.NORTH

WEB_DIR = Path("web")
MODEL_PATH = Path("models/q_learning_table.npy")


def state_dict(state: State) -> dict:
    return {
        "row": state.row,
        "col": state.col,
        "heading": state.heading.name,
    }


def run_policy(
    env: GridEnvironment,
    policy,
    policy_id: str,
    display_name: str,
    algorithm: str,
    extra: dict,
    max_steps: int = 200,
) -> dict:

    state = State(
        START[0],
        START[1],
        START_HEADING,
    )

    trajectory = [
        {
            "step": 0,
            "state": state_dict(state),
            "action": "START",
            "reward": 0.0,
            "blocked": False,
        }
    ]

    total_reward = 0.0
    blocked_moves = 0
    planning_times = []

    for step_number in range(1, max_steps + 1):

        if env.is_goal(state):
            break

        action = policy.action(state)

        if policy_id == "mcts":
            planning_times.append(
                policy.last_planning_time * 1000.0
            )

        result = env.step(
            state,
            action,
        )

        total_reward += result.reward

        if result.blocked:
            blocked_moves += 1

        state = result.state

        trajectory.append(
            {
                "step": step_number,
                "state": state_dict(state),
                "action": action.name,
                "reward": result.reward,
                "blocked": result.blocked,
            }
        )

        if result.terminated:
            break

    success = env.is_goal(state)

    if not success:
        raise RuntimeError(
            f"{display_name} failed to reach the goal."
        )

    average_planning_ms = (
        sum(planning_times) / len(planning_times)
        if planning_times
        else None
    )

    data = {
        "policy_id": policy_id,
        "display_name": display_name,
        "algorithm": algorithm,
        "rows": ROWS,
        "cols": COLS,
        "start": {
            "row": START[0],
            "col": START[1],
        },
        "goal": {
            "row": GOAL[0],
            "col": GOAL[1],
        },
        "success": success,
        "total_actions": len(trajectory) - 1,
        "total_reward": total_reward,
        "blocked_moves": blocked_moves,
        "average_planning_ms": average_planning_ms,
        "extra": extra,
        "trajectory": trajectory,
    }

    return data


def main():

    WEB_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 64)
    print("GENERATING WEB POLICY TRAJECTORIES")
    print("=" * 64)

    # --------------------------------------------------------
    # DYNAMIC PROGRAMMING
    # --------------------------------------------------------

    print()
    print("1/3 Dynamic Programming...")

    dp_env = GridEnvironment(
        rows=ROWS,
        cols=COLS,
        goal=GOAL,
    )

    dp_policy = ValueIterationPolicy(
        dp_env
    )

    dp_policy.solve()
    dp_iterations = dp_policy.iterations

    dp_data = run_policy(
        env=dp_env,
        policy=dp_policy,
        policy_id="dp",
        display_name="Dynamic Programming",
        algorithm="Value Iteration",
        extra={
            "convergence_iterations": dp_iterations,
            "description": (
                "Policy calculated from the complete "
                "environment model."
            ),
        },
    )

    # --------------------------------------------------------
    # Q-LEARNING
    # --------------------------------------------------------

    print("2/3 Q-Learning...")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Missing models/q_learning_table.npy. "
            "Run train_q_learning.py first."
        )

    q_env = GridEnvironment(
        rows=ROWS,
        cols=COLS,
        goal=GOAL,
    )

    q_policy = QLearningPolicy(
        q_env,
        seed=42,
    )

    q_policy.load(
        MODEL_PATH
    )

    q_data = run_policy(
        env=q_env,
        policy=q_policy,
        policy_id="q_learning",
        display_name="Q-Learning",
        algorithm="Tabular Q-Learning",
        extra={
            "training_episodes": 30000,
            "description": (
                "Policy learned through interaction "
                "and reward."
            ),
        },
    )

    # --------------------------------------------------------
    # MCTS
    # --------------------------------------------------------

    print("3/3 MCTS...")

    mcts_env = GridEnvironment(
        rows=ROWS,
        cols=COLS,
        goal=GOAL,
    )

    mcts_policy = MCTSPolicy(
        env=mcts_env,
        simulations=400,
        rollout_depth=80,
        seed=42,
    )

    mcts_data = run_policy(
        env=mcts_env,
        policy=mcts_policy,
        policy_id="mcts",
        display_name="MCTS",
        algorithm="Monte Carlo Tree Search",
        extra={
            "simulations_per_action": 400,
            "rollout_depth": 80,
            "description": (
                "Policy chosen through online "
                "simulation and tree search."
            ),
        },
    )

    output = {
        "default_policy": "dp",
        "policies": {
            "dp": dp_data,
            "q_learning": q_data,
            "mcts": mcts_data,
        },
    }

    output_path = (
        WEB_DIR
        / "policy_trajectories.json"
    )

    output_path.write_text(
        json.dumps(
            output,
            indent=2,
        )
    )

    print()
    print("=" * 64)
    print("TRAJECTORIES GENERATED")
    print("=" * 64)

    for item in (
        dp_data,
        q_data,
        mcts_data,
    ):
        print(
            f"{item['display_name']:<22} "
            f"actions={item['total_actions']:<4} "
            f"reward={item['total_reward']:<6.1f} "
            f"success={item['success']}"
        )

    print()
    print(
        "Saved: "
        "web/policy_trajectories.json"
    )


if __name__ == "__main__":
    main()
