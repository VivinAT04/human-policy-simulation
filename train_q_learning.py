"""Train and save the Q-Learning wheelchair policy."""

from pathlib import Path

from human_policy_sim.environment import GridEnvironment
from policies.q_learning import QLearningPolicy


GOAL = (1, 18)

MODEL_PATH = Path(
    "models/q_learning_table.npy"
)

MODEL_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
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

print("=" * 65)
print("Q-LEARNING TRAINING")
print("=" * 65)

print()
print("Environment : 20 x 20")
print(f"Goal        : {GOAL}")
print("Episodes    : 30000")
print()
print("Training...")

rewards = agent.train(
    episodes=30_000,
    max_steps=200,
)

agent.save(str(MODEL_PATH))

recent_rewards = rewards[-1000:]

average_recent_reward = (
    sum(recent_rewards)
    / len(recent_rewards)
)

print()
print("TRAINING COMPLETE")
print("=" * 65)

print(
    f"Successful episodes : "
    f"{agent.training_successes} / "
    f"{agent.training_episodes}"
)

print(
    f"Last 1000 avg reward: "
    f"{average_recent_reward:.2f}"
)

print(
    f"Saved Q-table       : "
    f"{MODEL_PATH}"
)

print("=" * 65)
