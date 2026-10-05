import json
from pathlib import Path

from human_policy_sim.environment import GridEnvironment, Heading, State
from policies.dynamic_programming import ValueIterationPolicy


START = (18, 1)
GOAL = (1, 18)

env = GridEnvironment(
    rows=20,
    cols=20,
    goal=GOAL,
)

policy = ValueIterationPolicy(env)
policy.solve()

state = State(
    START[0],
    START[1],
    Heading.NORTH,
)

steps = [{
    "step": 0,
    "row": state.row,
    "col": state.col,
    "heading": state.heading.name,
    "action": "START",
    "reward": 0.0,
}]

total_reward = 0.0

for step_number in range(1, 201):

    if env.is_goal(state):
        break

    action = policy.action(state)
    result = env.step(state, action)

    total_reward += result.reward
    state = result.state

    steps.append({
        "step": step_number,
        "row": state.row,
        "col": state.col,
        "heading": state.heading.name,
        "action": action.name,
        "reward": result.reward,
    })

data = {
    "algorithm": "Dynamic Programming — Value Iteration",
    "rows": env.rows,
    "cols": env.cols,
    "start": list(START),
    "goal": list(GOAL),
    "success": env.is_goal(state),
    "iterations": policy.iterations,
    "actions": len(steps) - 1,
    "total_reward": total_reward,
    "steps": steps,
}

Path("web").mkdir(exist_ok=True)

with open("web/dp_trajectory.json", "w") as f:
    json.dump(data, f, indent=2)

print("✓ web/dp_trajectory.json generated")
print(f"✓ Success: {data['success']}")
print(f"✓ Actions: {data['actions']}")
print(f"✓ DP iterations: {data['iterations']}")
