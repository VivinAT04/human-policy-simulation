"""Generate Policy × Noise × Control-Frequency browser trajectories.

Frequency represents the rate at which a new control command becomes
available.

A discrete LEFT, RIGHT, FORWARD or STOP command is executed exactly once
when a control update arrives. Between control updates, no additional
environment action is executed.

This avoids incorrectly treating discrete turn commands as continuous
motor commands.
"""

from __future__ import annotations

import json
from pathlib import Path

from human_policy_sim.environment import (
    GridEnvironment,
    Heading,
    State,
)
from human_policy_sim.noise import ActionNoise
from human_policy_sim.frequency import (
    ControlFrequency,
    SUPPORTED_FREQUENCIES,
)
from policies.dynamic_programming import ValueIterationPolicy
from policies.q_learning import QLearningPolicy
from policies.mcts import MCTSPolicy


ROWS = 20
COLS = 20

START = (18, 1)
GOAL = (1, 18)
START_HEADING = Heading.NORTH

NOISE_LEVELS = (
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
)

FREQUENCIES = SUPPORTED_FREQUENCIES

SIMULATION_HZ = 20

# Maximum number of actual policy/control decisions.
MAX_DECISIONS = 500

BASE_SEED = 42

MODEL_PATH = Path(
    "models/q_learning_table.npy"
)

OUTPUT_PATH = Path(
    "web/experiment_trajectories.json"
)


def state_dict(
    state: State,
) -> dict:

    return {
        "row": state.row,
        "col": state.col,
        "heading": state.heading.name,
    }


def build_policy(
    policy_id: str,
):

    env = GridEnvironment(
        rows=ROWS,
        cols=COLS,
        goal=GOAL,
    )

    if policy_id == "dp":

        policy = ValueIterationPolicy(
            env
        )

        policy.solve()

        return (
            env,
            policy,
            "Dynamic Programming",
            "Value Iteration",
        )

    if policy_id == "q_learning":

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "Missing models/q_learning_table.npy"
            )

        policy = QLearningPolicy(
            env,
            seed=42,
        )

        policy.load(
            MODEL_PATH
        )

        return (
            env,
            policy,
            "Q-Learning",
            "Tabular Q-Learning",
        )

    if policy_id == "mcts":

        policy = MCTSPolicy(
            env=env,
            simulations=400,
            rollout_depth=80,
            seed=42,
        )

        return (
            env,
            policy,
            "MCTS",
            "Monte Carlo Tree Search",
        )

    raise ValueError(
        f"Unknown policy: {policy_id}"
    )


def run_experiment(
    policy_id: str,
    noise_probability: float,
    control_hz: int,
    seed: int,
) -> dict:

    (
        env,
        policy,
        display_name,
        algorithm,
    ) = build_policy(
        policy_id
    )

    frequency = ControlFrequency(
        control_hz=control_hz,
        simulation_hz=SIMULATION_HZ,
    )

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
            "decision": 0,
            "simulation_tick": 0,
            "time_seconds": 0.0,
            "state": state_dict(state),
            "intended_action": "START",
            "executed_action": "START",
            "corrupted": False,
            "blocked": False,
            "reward": 0.0,
        }
    ]

    total_reward = 0.0
    blocked_moves = 0
    corrupted_decisions = 0
    policy_decisions = 0

    for decision_index in range(
        MAX_DECISIONS
    ):

        if env.is_goal(state):
            break

        intended_action = policy.action(
            state
        )

        executed_action = noise.apply(
            intended_action
        )

        corrupted = (
            executed_action
            != intended_action
        )

        if corrupted:
            corrupted_decisions += 1

        result = env.step(
            state,
            executed_action,
        )

        policy_decisions += 1

        total_reward += result.reward

        if result.blocked:
            blocked_moves += 1

        state = result.state

        # Decision 1 occurs at t = 0.
        #
        # The state resulting from that command is therefore
        # displayed at the end of one control interval.
        elapsed_seconds = (
            policy_decisions
            / control_hz
        )

        simulation_tick = (
            policy_decisions
            * frequency.hold_ticks
        )

        trajectory.append(
            {
                "decision":
                    policy_decisions,

                "simulation_tick":
                    simulation_tick,

                "time_seconds":
                    elapsed_seconds,

                "state":
                    state_dict(state),

                "intended_action":
                    intended_action.name,

                "executed_action":
                    executed_action.name,

                "corrupted":
                    corrupted,

                "blocked":
                    result.blocked,

                "reward":
                    result.reward,
            }
        )

        if result.terminated:
            break

    success = env.is_goal(
        state
    )

    elapsed_seconds = (
        policy_decisions
        / control_hz
    )

    simulation_ticks = (
        policy_decisions
        * frequency.hold_ticks
    )

    return {
        "policy_id":
            policy_id,

        "display_name":
            display_name,

        "algorithm":
            algorithm,

        "noise_probability":
            noise_probability,

        "noise_percent":
            int(
                round(
                    noise_probability * 100
                )
            ),

        "control_hz":
            control_hz,

        "simulation_hz":
            SIMULATION_HZ,

        "update_interval_seconds":
            frequency.update_interval_seconds,

        "ticks_between_updates":
            frequency.hold_ticks,

        "seed":
            seed,

        "success":
            success,

        "policy_decisions":
            policy_decisions,

        "simulation_ticks":
            simulation_ticks,

        "elapsed_seconds":
            elapsed_seconds,

        "corrupted_decisions":
            corrupted_decisions,

        "blocked_moves":
            blocked_moves,

        "total_reward":
            total_reward,

        "trajectory":
            trajectory,
    }


def main():

    print("=" * 78)

    print(
        "GENERATING POLICY × NOISE × "
        "CONTROL-FREQUENCY TRAJECTORIES"
    )

    print("=" * 78)

    output = {
        "rows":
            ROWS,

        "cols":
            COLS,

        "start": {
            "row": START[0],
            "col": START[1],
        },

        "goal": {
            "row": GOAL[0],
            "col": GOAL[1],
        },

        "simulation_hz":
            SIMULATION_HZ,

        "max_decisions":
            MAX_DECISIONS,

        "noise_levels":
            list(NOISE_LEVELS),

        "frequencies":
            list(FREQUENCIES),

        "frequency_semantics":
            (
                "A new discrete control command is "
                "generated and executed once at each "
                "control update. No repeated environment "
                "action is applied between updates."
            ),

        "experiment_interpretation": {
            "noise":
                (
                    "Action corruption affecting control "
                    "accuracy, trajectory and reward."
                ),

            "frequency":
                (
                    "Physiological/control command update "
                    "rate affecting response latency and "
                    "completion time in the discrete model."
                ),

            "interaction":
                (
                    "Noise can increase the number of "
                    "decisions required. Lower control "
                    "frequency increases the time cost of "
                    "those additional decisions."
                ),

            "future_extension":
                (
                    "Continuous dynamics can model "
                    "persistent linear and angular motion "
                    "between control updates."
                ),
        },

        "policies": {},
    }

    policy_ids = (
        "dp",
        "q_learning",
        "mcts",
    )

    policy_offsets = {
        "dp": 0,
        "q_learning": 10000,
        "mcts": 20000,
    }

    total_runs = (
        len(policy_ids)
        * len(NOISE_LEVELS)
        * len(FREQUENCIES)
    )

    completed = 0

    for policy_id in policy_ids:

        policy_data = {
            "runs": {}
        }

        for noise_index, noise_level in enumerate(
            NOISE_LEVELS
        ):

            noise_key = (
                f"{noise_level:.1f}"
            )

            policy_data[
                "runs"
            ][noise_key] = {}

            for frequency in FREQUENCIES:

                completed += 1

                # Keep the random corruption sequence identical
                # across frequencies for a given policy/noise level.
                #
                # This makes frequency comparisons fair:
                # frequency changes timing, not which random numbers
                # happen to be drawn.
                seed = (
                    BASE_SEED
                    + policy_offsets[policy_id]
                    + noise_index * 100
                )

                result = run_experiment(
                    policy_id=policy_id,
                    noise_probability=noise_level,
                    control_hz=frequency,
                    seed=seed,
                )

                policy_data[
                    "display_name"
                ] = result[
                    "display_name"
                ]

                policy_data[
                    "algorithm"
                ] = result[
                    "algorithm"
                ]

                policy_data[
                    "runs"
                ][noise_key][
                    str(frequency)
                ] = result

                print(
                    f"[{completed:>2}/{total_runs}] "
                    f"{result['display_name']:<20} | "
                    f"noise={result['noise_percent']:>2}% | "
                    f"{frequency:>2} Hz | "
                    f"decisions="
                    f"{result['policy_decisions']:>3} | "
                    f"time="
                    f"{result['elapsed_seconds']:>7.2f}s | "
                    f"corrupt="
                    f"{result['corrupted_decisions']:>3} | "
                    f"blocked="
                    f"{result['blocked_moves']:>3} | "
                    f"success={result['success']}"
                )

        output[
            "policies"
        ][policy_id] = policy_data

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
