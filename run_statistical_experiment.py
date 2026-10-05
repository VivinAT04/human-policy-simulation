"""
Multi-seed robustness experiment.

Design
------
Policies:
    - Dynamic Programming
    - Q-Learning
    - Monte Carlo Tree Search

Noise:
    0.0, 0.1, 0.2, 0.3, 0.4, 0.5

Control frequency:
    20, 10, 5, 2, 1 Hz

Independent stochastic seeds:
    30

Total reported conditions:
    3 × 6 × 5 × 30 = 2700 rows

Important discrete-frequency semantics
--------------------------------------
A command is generated and executed exactly once per control
update. The environment does not evolve between updates.

Therefore, for a fixed policy/noise/seed combination, spatial
behaviour is independent of control frequency. Frequency changes
elapsed completion time only:

    elapsed_seconds = decisions / control_hz

We therefore simulate each policy/noise/seed trajectory ONCE and
derive the five frequency conditions from that same trajectory.
This avoids recomputing identical spatial trials and preserves
matched/common randomness across frequencies.
"""

from __future__ import annotations

import csv
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path

from human_policy_sim.environment import (
    GridEnvironment,
    Heading,
    State,
)
from human_policy_sim.frequency import (
    SUPPORTED_FREQUENCIES,
)
from human_policy_sim.noise import ActionNoise
from policies.dynamic_programming import ValueIterationPolicy
from policies.mcts import MCTSPolicy
from policies.q_learning import QLearningPolicy


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

FREQUENCIES = tuple(
    SUPPORTED_FREQUENCIES
)

N_SEEDS = 30
BASE_SEED = 42
MAX_DECISIONS = 500

Q_TABLE_PATH = Path(
    "models/q_learning_table.npy"
)

RESULTS_DIR = Path("results")

RAW_PATH = RESULTS_DIR / (
    "statistical_experiment_raw.csv"
)

SUMMARY_PATH = RESULTS_DIR / (
    "statistical_experiment_summary.csv"
)


POLICY_INFO = {
    "dp": {
        "display_name":
            "Dynamic Programming",
        "offset":
            0,
    },
    "q_learning": {
        "display_name":
            "Q-Learning",
        "offset":
            100_000,
    },
    "mcts": {
        "display_name":
            "MCTS",
        "offset":
            200_000,
    },
}


def build_policy(
    policy_id: str,
    env: GridEnvironment,
    planning_seed: int,
):
    if policy_id == "dp":

        policy = ValueIterationPolicy(
            env=env,
            gamma=0.99,
        )

        policy.solve()

        return policy

    if policy_id == "q_learning":

        policy = QLearningPolicy(
            env=env,
        )

        policy.load(
            Q_TABLE_PATH
        )

        return policy

    if policy_id == "mcts":

        return MCTSPolicy(
            env=env,
            simulations=400,
            rollout_depth=80,
            gamma=0.99,
            seed=planning_seed,
        )

    raise ValueError(
        f"Unknown policy: {policy_id}"
    )


def run_spatial_trial(
    policy_id: str,
    noise_probability: float,
    trial_index: int,
) -> dict:
    """
    Run one spatial policy/noise/seed trial.

    This trajectory is later projected onto every control
    frequency because frequency does not alter spatial
    evolution in the current discrete model.
    """

    info = POLICY_INFO[policy_id]

    trial_seed = (
        BASE_SEED
        + info["offset"]
        + trial_index
    )

    # Keep the noise stream reproducible and independent
    # from MCTS's internal planning RNG.
    noise_seed = (
        trial_seed
        + 1_000_000
        + int(
            round(
                noise_probability
                * 1000
            )
        )
        * 10_000
    )

    planning_seed = (
        trial_seed
        + 2_000_000
    )

    env = GridEnvironment(
        rows=ROWS,
        cols=COLS,
        goal=GOAL,
    )

    policy = build_policy(
        policy_id,
        env,
        planning_seed,
    )

    noise = ActionNoise(
        noise_probability=
            noise_probability,
        seed=noise_seed,
    )

    state = State(
        START[0],
        START[1],
        START_HEADING,
    )

    total_reward = 0.0
    blocked_moves = 0
    corrupted_actions = 0
    decisions = 0

    visited_positions = {
        (
            state.row,
            state.col,
        )
    }

    planning_seconds = 0.0

    while (
        decisions < MAX_DECISIONS
        and not env.is_goal(state)
    ):

        planning_start = (
            time.perf_counter()
        )

        intended_action = (
            policy.action(state)
        )

        planning_seconds += (
            time.perf_counter()
            - planning_start
        )

        executed_action = noise.apply(
            intended_action
        )

        if (
            executed_action
            != intended_action
        ):
            corrupted_actions += 1

        result = env.step(
            state,
            executed_action,
        )

        decisions += 1

        total_reward += (
            result.reward
        )

        blocked_moves += int(
            result.blocked
        )

        state = result.state

        visited_positions.add(
            (
                state.row,
                state.col,
            )
        )

        if result.terminated:
            break

    success = env.is_goal(state)

    return {
        "trial_index":
            trial_index,

        "trial_seed":
            trial_seed,

        "noise_seed":
            noise_seed,

        "planning_seed":
            planning_seed,

        "success":
            int(success),

        "decisions":
            decisions,

        "total_reward":
            total_reward,

        "blocked_moves":
            blocked_moves,

        "corrupted_actions":
            corrupted_actions,

        "unique_positions":
            len(visited_positions),

        "planning_seconds":
            planning_seconds,
    }


def sample_std(
    values: list[float],
) -> float:
    if len(values) <= 1:
        return 0.0

    return statistics.stdev(
        values
    )


def mean_ci95(
    values: list[float],
) -> tuple[float, float, float]:
    """
    Normal-approximation 95% CI for the sample mean.

    With n=30 this is suitable for descriptive experiment
    summaries. Raw trial data are retained for any later
    bootstrap/non-parametric analysis.
    """

    mean = statistics.mean(
        values
    )

    if len(values) <= 1:
        return (
            mean,
            mean,
            mean,
        )

    std = sample_std(
        values
    )

    half_width = (
        1.96
        * std
        / math.sqrt(
            len(values)
        )
    )

    return (
        mean,
        mean - half_width,
        mean + half_width,
    )


def write_raw(
    rows: list[dict],
) -> None:
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "policy_id",
        "policy",
        "noise_probability",
        "control_hz",
        "trial_index",
        "trial_seed",
        "noise_seed",
        "planning_seed",
        "success",
        "decisions",
        "total_reward",
        "blocked_moves",
        "corrupted_actions",
        "unique_positions",
        "elapsed_seconds",
        "planning_seconds",
    ]

    with RAW_PATH.open(
        "w",
        newline="",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            lineterminator="\\n",
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


def write_summary(
    raw_rows: list[dict],
) -> list[dict]:

    grouped = defaultdict(list)

    for row in raw_rows:

        key = (
            row["policy_id"],
            row[
                "noise_probability"
            ],
            row["control_hz"],
        )

        grouped[key].append(
            row
        )

    summary_rows = []

    for key in sorted(
        grouped,
        key=lambda x: (
            list(
                POLICY_INFO
            ).index(x[0]),
            x[1],
            -x[2],
        ),
    ):

        policy_id, noise, hz = key

        rows = grouped[key]

        successes = [
            float(
                row["success"]
            )
            for row in rows
        ]

        decisions = [
            float(
                row["decisions"]
            )
            for row in rows
        ]

        rewards = [
            float(
                row["total_reward"]
            )
            for row in rows
        ]

        blocked = [
            float(
                row["blocked_moves"]
            )
            for row in rows
        ]

        corruptions = [
            float(
                row[
                    "corrupted_actions"
                ]
            )
            for row in rows
        ]

        elapsed = [
            float(
                row[
                    "elapsed_seconds"
                ]
            )
            for row in rows
        ]

        planning = [
            float(
                row[
                    "planning_seconds"
                ]
            )
            for row in rows
        ]

        unique_positions = [
            float(
                row[
                    "unique_positions"
                ]
            )
            for row in rows
        ]

        success_mean, (
            success_low
        ), success_high = (
            mean_ci95(
                successes
            )
        )

        decisions_mean, (
            decisions_low
        ), decisions_high = (
            mean_ci95(
                decisions
            )
        )

        reward_mean, (
            reward_low
        ), reward_high = (
            mean_ci95(
                rewards
            )
        )

        elapsed_mean, (
            elapsed_low
        ), elapsed_high = (
            mean_ci95(
                elapsed
            )
        )

        summary_rows.append(
            {
                "policy_id":
                    policy_id,

                "policy":
                    POLICY_INFO[
                        policy_id
                    ][
                        "display_name"
                    ],

                "noise_probability":
                    noise,

                "control_hz":
                    hz,

                "n_trials":
                    len(rows),

                "success_rate":
                    success_mean,

                "success_ci95_low":
                    max(
                        0.0,
                        success_low,
                    ),

                "success_ci95_high":
                    min(
                        1.0,
                        success_high,
                    ),

                "decisions_mean":
                    decisions_mean,

                "decisions_std":
                    sample_std(
                        decisions
                    ),

                "decisions_ci95_low":
                    decisions_low,

                "decisions_ci95_high":
                    decisions_high,

                "reward_mean":
                    reward_mean,

                "reward_std":
                    sample_std(
                        rewards
                    ),

                "reward_ci95_low":
                    reward_low,

                "reward_ci95_high":
                    reward_high,

                "blocked_mean":
                    statistics.mean(
                        blocked
                    ),

                "corrupted_mean":
                    statistics.mean(
                        corruptions
                    ),

                "unique_positions_mean":
                    statistics.mean(
                        unique_positions
                    ),

                "elapsed_seconds_mean":
                    elapsed_mean,

                "elapsed_seconds_std":
                    sample_std(
                        elapsed
                    ),

                "elapsed_ci95_low":
                    elapsed_low,

                "elapsed_ci95_high":
                    elapsed_high,

                "planning_seconds_mean":
                    statistics.mean(
                        planning
                    ),
            }
        )

    fieldnames = list(
        summary_rows[0].keys()
    )

    with SUMMARY_PATH.open(
        "w",
        newline="",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            lineterminator="\\n",
        )

        writer.writeheader()

        writer.writerows(
            summary_rows
        )

    return summary_rows


def verify_frequency_matching(
    raw_rows: list[dict],
) -> None:
    """
    Verify spatial outcomes are identical across frequency
    for each policy/noise/trial combination.
    """

    grouped = defaultdict(list)

    for row in raw_rows:

        key = (
            row["policy_id"],
            row[
                "noise_probability"
            ],
            row["trial_index"],
        )

        grouped[key].append(
            row
        )

    spatial_fields = (
        "success",
        "decisions",
        "total_reward",
        "blocked_moves",
        "corrupted_actions",
        "unique_positions",
    )

    for key, rows in grouped.items():

        assert (
            len(rows)
            == len(FREQUENCIES)
        ), key

        reference = rows[0]

        for row in rows[1:]:

            for field in spatial_fields:

                assert (
                    row[field]
                    == reference[field]
                ), (
                    key,
                    field,
                    reference[field],
                    row[field],
                )

        for row in rows:

            expected_time = (
                row["decisions"]
                / row["control_hz"]
            )

            assert math.isclose(
                row[
                    "elapsed_seconds"
                ],
                expected_time,
                rel_tol=0.0,
                abs_tol=1e-12,
            )

    print(
        "✓ Frequency matching verified"
    )


def main() -> None:

    expected_spatial_trials = (
        len(POLICY_INFO)
        * len(NOISE_LEVELS)
        * N_SEEDS
    )

    expected_rows = (
        expected_spatial_trials
        * len(FREQUENCIES)
    )

    print(
        "=" * 76
    )

    print(
        "MULTI-SEED POLICY ROBUSTNESS EXPERIMENT"
    )

    print(
        "=" * 76
    )

    print(
        f"Policies            : "
        f"{len(POLICY_INFO)}"
    )

    print(
        f"Noise levels        : "
        f"{len(NOISE_LEVELS)}"
    )

    print(
        f"Control frequencies : "
        f"{len(FREQUENCIES)}"
    )

    print(
        f"Seeds / condition   : "
        f"{N_SEEDS}"
    )

    print(
        f"Spatial simulations : "
        f"{expected_spatial_trials}"
    )

    print(
        f"Reported rows       : "
        f"{expected_rows}"
    )

    print()

    raw_rows = []

    completed = 0

    for policy_id, info in (
        POLICY_INFO.items()
    ):

        print()
        print(
            info["display_name"]
        )

        for noise in NOISE_LEVELS:

            noise_trials = []

            for trial_index in range(
                N_SEEDS
            ):

                trial = (
                    run_spatial_trial(
                        policy_id,
                        noise,
                        trial_index,
                    )
                )

                noise_trials.append(
                    trial
                )

                completed += 1

                print(
                    "\r"
                    f"  noise="
                    f"{noise:.1f} "
                    f"| trial "
                    f"{trial_index + 1:02d}"
                    f"/{N_SEEDS} "
                    f"| spatial "
                    f"{completed:03d}"
                    f"/"
                    f"{expected_spatial_trials}",
                    end="",
                    flush=True,
                )

            print()

            successes = sum(
                trial["success"]
                for trial in noise_trials
            )

            mean_decisions = (
                statistics.mean(
                    trial[
                        "decisions"
                    ]
                    for trial
                    in noise_trials
                )
            )

            print(
                f"    success="
                f"{successes}/{N_SEEDS}"
                f" | mean decisions="
                f"{mean_decisions:.2f}"
            )

            for trial in noise_trials:

                for hz in FREQUENCIES:

                    raw_rows.append(
                        {
                            "policy_id":
                                policy_id,

                            "policy":
                                info[
                                    "display_name"
                                ],

                            "noise_probability":
                                noise,

                            "control_hz":
                                hz,

                            "trial_index":
                                trial[
                                    "trial_index"
                                ],

                            "trial_seed":
                                trial[
                                    "trial_seed"
                                ],

                            "noise_seed":
                                trial[
                                    "noise_seed"
                                ],

                            "planning_seed":
                                trial[
                                    "planning_seed"
                                ],

                            "success":
                                trial[
                                    "success"
                                ],

                            "decisions":
                                trial[
                                    "decisions"
                                ],

                            "total_reward":
                                trial[
                                    "total_reward"
                                ],

                            "blocked_moves":
                                trial[
                                    "blocked_moves"
                                ],

                            "corrupted_actions":
                                trial[
                                    "corrupted_actions"
                                ],

                            "unique_positions":
                                trial[
                                    "unique_positions"
                                ],

                            "elapsed_seconds":
                                (
                                    trial[
                                        "decisions"
                                    ]
                                    / hz
                                ),

                            "planning_seconds":
                                trial[
                                    "planning_seconds"
                                ],
                        }
                    )

    assert (
        len(raw_rows)
        == expected_rows
    )

    verify_frequency_matching(
        raw_rows
    )

    write_raw(
        raw_rows
    )

    summary_rows = write_summary(
        raw_rows
    )

    expected_summary_rows = (
        len(POLICY_INFO)
        * len(NOISE_LEVELS)
        * len(FREQUENCIES)
    )

    assert (
        len(summary_rows)
        == expected_summary_rows
    )

    print()
    print(
        "=" * 76
    )

    print(
        "EXPERIMENT COMPLETE"
    )

    print(
        "=" * 76
    )

    print(
        "Raw rows    :",
        len(raw_rows),
    )

    print(
        "Summary rows:",
        len(summary_rows),
    )

    print(
        "Raw CSV     :",
        RAW_PATH,
    )

    print(
        "Summary CSV :",
        SUMMARY_PATH,
    )


if __name__ == "__main__":
    main()
