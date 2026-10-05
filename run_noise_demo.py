"""Demonstrate the configurable action-noise model."""

from collections import Counter

from human_policy_sim.environment import Action
from human_policy_sim.noise import ActionNoise


NOISE_LEVELS = [
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
]

TRIALS = 10000
SEED = 42


def main():

    intended = Action.FORWARD

    print("=" * 72)
    print("ACTION NOISE MODEL")
    print("=" * 72)

    print()
    print(
        "Intended action:",
        intended.name,
    )

    print(
        "Trials per level:",
        TRIALS,
    )

    print(
        "Seed:",
        SEED,
    )

    print()

    for level in NOISE_LEVELS:

        noise = ActionNoise(
            noise_probability=level,
            seed=SEED,
        )

        executed = [
            noise.apply(intended)
            for _ in range(TRIALS)
        ]

        counts = Counter(
            action.name
            for action in executed
        )

        changed = sum(
            action != intended
            for action in executed
        )

        observed = changed / TRIALS

        print("-" * 72)

        print(
            f"Requested noise: "
            f"{level:.1f}"
        )

        print(
            f"Observed noise:  "
            f"{observed:.3f}"
        )

        print(
            "Executed actions:"
        )

        for action in Action:

            print(
                f"  {action.name:<8} "
                f"{counts[action.name]:>5}"
            )

    print()
    print("=" * 72)
    print("DONE")
    print("=" * 72)


if __name__ == "__main__":
    main()
