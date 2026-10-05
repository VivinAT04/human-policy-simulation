"""Run the Dynamic Programming policy on the 20x20 grid."""

from human_policy_sim.environment import (
    GridEnvironment,
    Heading,
    State,
)
from policies.dynamic_programming import ValueIterationPolicy


START = (18, 1)
GOAL = (1, 18)

env = GridEnvironment(
    rows=20,
    cols=20,
    goal=GOAL,
)

policy = ValueIterationPolicy(
    env=env,
    gamma=0.99,
)

print("=" * 65)
print("DYNAMIC PROGRAMMING — VALUE ITERATION")
print("=" * 65)

print()
print("Solving policy...")

policy.solve()

print(f"Converged in: {policy.iterations} iterations")
print(f"Number of states: {len(policy.states)}")

state = State(
    row=START[0],
    col=START[1],
    heading=Heading.NORTH,
)

trajectory = [state]

total_reward = 0.0
blocked_moves = 0

print()
print(f"START: {START}")
print(f"GOAL : {GOAL}")
print()
print("-" * 65)

for step_number in range(1, 201):

    if env.is_goal(state):
        break

    action = policy.action(state)

    result = env.step(
        state,
        action,
    )

    print(
        f"{step_number:03d} | "
        f"({state.row:02d},{state.col:02d}) "
        f"{state.heading.name:<5} "
        f"-> {action.name:<7} "
        f"-> ({result.state.row:02d},{result.state.col:02d}) "
        f"{result.state.heading.name:<5} "
        f"| reward={result.reward:6.1f}"
    )

    total_reward += result.reward

    if result.blocked:
        blocked_moves += 1

    state = result.state
    trajectory.append(state)

print("-" * 65)
print()

success = env.is_goal(state)

print("RESULT")
print("=" * 65)
print(f"Success        : {success}")
print(f"Final position : ({state.row}, {state.col})")
print(f"Goal           : {GOAL}")
print(f"Actions        : {len(trajectory) - 1}")
print(f"Total reward   : {total_reward:.1f}")
print(f"Blocked moves  : {blocked_moves}")
print(f"DP iterations  : {policy.iterations}")

print()
print("POSITION TRAJECTORY")
print("=" * 65)

# Only print a position when the wheelchair actually changes cell.
positions = []

for s in trajectory:
    position = (s.row, s.col)

    if not positions or positions[-1] != position:
        positions.append(position)

print(" -> ".join(str(p) for p in positions))

print()
print("=" * 65)

if not success:
    raise RuntimeError(
        "DP policy failed to reach the goal."
    )

print("DP POLICY SUCCESS ✓")
