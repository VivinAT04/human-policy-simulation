"""Evaluate the trained Q-Learning policy."""

from pathlib import Path

from human_policy_sim.environment import (
    GridEnvironment,
    Heading,
    State,
)
from policies.q_learning import QLearningPolicy


START = (18, 1)
GOAL = (1, 18)

MODEL_PATH = Path(
    "models/q_learning_table.npy"
)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Q-table not found. "
        "Run train_q_learning.py first."
    )

env = GridEnvironment(
    rows=20,
    cols=20,
    goal=GOAL,
)

agent = QLearningPolicy(
    env=env,
    seed=42,
)

agent.load(str(MODEL_PATH))

state = State(
    START[0],
    START[1],
    Heading.NORTH,
)

trajectory = [state]

total_reward = 0.0
blocked_moves = 0

print("=" * 65)
print("Q-LEARNING — FROZEN GREEDY POLICY")
print("=" * 65)

print()
print(f"START: {START}")
print(f"GOAL : {GOAL}")
print()
print("-" * 65)

for step_number in range(1, 201):

    if env.is_goal(state):
        break

    action = agent.action(state)

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
        f"| reward={result.reward:6.1f}"
    )

    total_reward += result.reward

    if result.blocked:
        blocked_moves += 1

    state = result.state

    trajectory.append(state)

print("-" * 65)

success = env.is_goal(state)

print()
print("RESULT")
print("=" * 65)

print(f"Success        : {success}")
print(
    f"Final position : "
    f"({state.row}, {state.col})"
)
print(f"Goal           : {GOAL}")
print(
    f"Actions        : "
    f"{len(trajectory) - 1}"
)
print(
    f"Total reward   : "
    f"{total_reward:.1f}"
)
print(
    f"Blocked moves  : "
    f"{blocked_moves}"
)

print("=" * 65)

if not success:
    raise RuntimeError(
        "Q-Learning policy failed to reach the goal."
    )

print("Q-LEARNING POLICY SUCCESS ✓")
