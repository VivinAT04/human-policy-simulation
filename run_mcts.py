"""Run MCTS on the dissertation-style 20x20 grid."""

import time

from human_policy_sim.environment import (
    GridEnvironment,
    Heading,
    State,
)
from policies.mcts import MCTSPolicy


START = (18, 1)
GOAL = (1, 18)

env = GridEnvironment(
    rows=20,
    cols=20,
    goal=GOAL,
)

policy = MCTSPolicy(
    env=env,
    simulations=400,
    rollout_depth=80,
    seed=42,
)

state = State(
    START[0],
    START[1],
    Heading.NORTH,
)

trajectory = [state]

total_reward = 0.0
blocked_moves = 0
planning_times = []

print("=" * 72)
print("MCTS — ONLINE MONTE CARLO TREE SEARCH")
print("=" * 72)

print()
print(f"START       : {START}")
print(f"GOAL        : {GOAL}")
print(
    f"SIMULATIONS : "
    f"{policy.simulations} per decision"
)

print()
print("-" * 72)

overall_start = time.perf_counter()

for step_number in range(1, 201):

    if env.is_goal(state):
        break

    action = policy.action(state)

    planning_times.append(
        policy.last_planning_time
    )

    result = env.step(
        state,
        action,
    )

    print(
        f"{step_number:03d} | "
        f"({state.row:02d},{state.col:02d}) "
        f"{state.heading.name:<5} "
        f"-> {action.name:<7} "
        f"-> "
        f"({result.state.row:02d},"
        f"{result.state.col:02d}) "
        f"{result.state.heading.name:<5} "
        f"| reward={result.reward:6.1f} "
        f"| plan={policy.last_planning_time * 1000:7.2f} ms"
    )

    total_reward += result.reward

    if result.blocked:
        blocked_moves += 1

    state = result.state

    trajectory.append(state)

overall_time = (
    time.perf_counter()
    - overall_start
)

success = env.is_goal(state)

average_planning_ms = (
    sum(planning_times)
    / len(planning_times)
    * 1000
    if planning_times
    else 0.0
)

print("-" * 72)

print()
print("RESULT")
print("=" * 72)

print(f"Success              : {success}")
print(
    f"Final position       : "
    f"({state.row}, {state.col})"
)
print(f"Goal                 : {GOAL}")
print(
    f"Actions              : "
    f"{len(trajectory) - 1}"
)
print(
    f"Total reward         : "
    f"{total_reward:.1f}"
)
print(
    f"Blocked moves        : "
    f"{blocked_moves}"
)
print(
    f"Avg planning / action: "
    f"{average_planning_ms:.2f} ms"
)
print(
    f"Total runtime        : "
    f"{overall_time:.2f} s"
)

print("=" * 72)

print()
print("POSITION TRAJECTORY")
print("=" * 72)

positions = []

for item in trajectory:

    position = (
        item.row,
        item.col,
    )

    if (
        not positions
        or positions[-1] != position
    ):
        positions.append(position)

print(
    " -> ".join(
        str(position)
        for position in positions
    )
)

print()
print("=" * 72)

if not success:
    raise RuntimeError(
        "MCTS failed to reach the goal."
    )

print("MCTS POLICY SUCCESS ✓")
