"""Generate noisy browser trajectories for DP, Q-Learning and MCTS."""

from __future__ import annotations

import json
from pathlib import Path

from human_policy_sim.environment import (
    GridEnvironment,
    Heading,
    State,
)
from human_policy_sim.noise import ActionNoise
from policies.dynamic_programming import ValueIterationPolicy
from policies.q_learning import QLearningPolicy
from policies.mcts import MCTSPolicy


ROWS = 20
COLS = 20

START = (18, 1)
GOAL = (1, 18)
START_HEADING = Heading.NORTH

NOISE_LEVELS = [
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
]

SEED = 42
MAX_STEPS = 500

MODEL_PATH = Path("models/q_learning_table.npy")
OUTPUT_PATH = Path("web/noise_trajectories.json")


def state_dict(state: State) -> dict:
    return {
        "row": state.row,
        "col": state.col,
        "heading": state.heading.name,
    }


def run_noisy_policy(
    env,
    policy,
    noise_probability,
    seed,
):

    noise = ActionNoise(
        noise_probability=noise_probability,
        seed=seed,
    )

    state = State(
        START[0],
        START[1],
        START_HEADING,
    )

    trajectory = [
        {
            "step": 0,
            "state": state_dict(state),
            "intended_action": "START",
            "executed_action": "START",
            "corrupted": False,
            "reward": 0.0,
            "blocked": False,
        }
    ]

    total_reward = 0.0
    blocked_moves = 0
    corrupted_actions = 0

    for step_number in range(
        1,
        MAX_STEPS + 1,
    ):

        if env.is_goal(state):
            break

        intended_action = policy.action(state)

        executed_action = noise.apply(
            intended_action
        )

        corrupted = (
            executed_action != intended_action
        )

        if corrupted:
            corrupted_actions += 1

        result = env.step(
            state,
            executed_action,
        )

        if result.blocked:
            blocked_moves += 1

        total_reward += result.reward
        state = result.state

        trajectory.append(
            {
                "step": step_number,
                "state": state_dict(state),
                "intended_action":
                    intended_action.name,
                "executed_action":
                    executed_action.name,
                "corrupted": corrupted,
                "reward": result.reward,
                "blocked": result.blocked,
            }
        )

        if result.terminated:
            break

    return {
        "noise_probability":
            noise_probability,

        "noise_percent":
            int(noise_probability * 100),

        "seed":
            seed,

        "success":
            env.is_goal(state),

        "total_actions":
            len(trajectory) - 1,

        "total_reward":
            total_reward,

        "blocked_moves":
            blocked_moves,

        "corrupted_actions":
            corrupted_actions,

        "trajectory":
            trajectory,
    }


def main():

    print("=" * 72)
    print("GENERATING NOISY POLICY TRAJECTORIES")
    print("=" * 72)

    # --------------------------------------------------------
    # DP
    # --------------------------------------------------------

    dp_env = GridEnvironment(
        rows=ROWS,
        cols=COLS,
        goal=GOAL,
    )

    dp_policy = ValueIterationPolicy(
        dp_env
    )

    dp_policy.solve()

    # --------------------------------------------------------
    # Q-LEARNING
    # --------------------------------------------------------

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Missing models/q_learning_table.npy"
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

    # --------------------------------------------------------
    # MCTS
    # --------------------------------------------------------

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

    policies = {
        "dp": {
            "display_name":
                "Dynamic Programming",

            "algorithm":
                "Value Iteration",

            "env":
                dp_env,

            "policy":
                dp_policy,
        },

        "q_learning": {
            "display_name":
                "Q-Learning",

            "algorithm":
                "Tabular Q-Learning",

            "env":
                q_env,

            "policy":
                q_policy,
        },

        "mcts": {
            "display_name":
                "MCTS",

            "algorithm":
                "Monte Carlo Tree Search",

            "env":
                mcts_env,

            "policy":
                mcts_policy,
        },
    }

    output = {
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

        "max_steps":
            MAX_STEPS,

        "noise_levels":
            NOISE_LEVELS,

        "policies": {},
    }

    for policy_id, config in policies.items():

        print()
        print(config["display_name"])

        policy_output = {
            "display_name":
                config["display_name"],

            "algorithm":
                config["algorithm"],

            "noise_runs": {},
        }

        for index, noise_level in enumerate(
            NOISE_LEVELS
        ):

            # Separate deterministic seed for each
            # policy/noise combination.
            run_seed = (
                SEED
                + index
                + {
                    "dp": 0,
                    "q_learning": 100,
                    "mcts": 200,
                }[policy_id]
            )

            # MCTS is stateful because its RNG advances.
            # Re-create it for every experimental run.
            if policy_id == "mcts":

                run_env = GridEnvironment(
                    rows=ROWS,
                    cols=COLS,
                    goal=GOAL,
                )

                run_policy = MCTSPolicy(
                    env=run_env,
                    simulations=400,
                    rollout_depth=80,
                    seed=42,
                )

            else:

                run_env = config["env"]
                run_policy = config["policy"]

            result = run_noisy_policy(
                env=run_env,
                policy=run_policy,
                noise_probability=noise_level,
                seed=run_seed,
            )

            key = f"{noise_level:.1f}"

            policy_output[
                "noise_runs"
            ][key] = result

            print(
                f"  noise={noise_level:.1f} | "
                f"actions={result['total_actions']:>3} | "
                f"reward={result['total_reward']:>7.1f} | "
                f"corrupted={result['corrupted_actions']:>3} | "
                f"blocked={result['blocked_moves']:>3} | "
                f"success={result['success']}"
            )

        output["policies"][
            policy_id
        ] = policy_output

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            output,
            indent=2,
        )
    )

    print()
    print(
        "Saved:",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
