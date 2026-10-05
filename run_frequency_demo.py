"""Demonstrate discrete control-update frequencies."""

from human_policy_sim.frequency import (
    ControlFrequency,
    DEFAULT_SIMULATION_HZ,
    SUPPORTED_FREQUENCIES,
)


def main():

    print("=" * 72)
    print("CONTROL UPDATE FREQUENCY MODEL")
    print("=" * 72)

    print()
    print(
        f"Simulation rate: "
        f"{DEFAULT_SIMULATION_HZ} Hz"
    )

    print()

    for frequency in SUPPORTED_FREQUENCIES:

        model = ControlFrequency(
            control_hz=frequency,
        )

        print("-" * 72)

        print(
            f"Control frequency: "
            f"{frequency} Hz"
        )

        print(
            f"Update interval:   "
            f"{model.update_interval_seconds:.2f} s"
        )

        print(
            f"Command hold:      "
            f"{model.hold_ticks} tick(s)"
        )

        update_ticks = [
            tick
            for tick in range(
                DEFAULT_SIMULATION_HZ + 1
            )
            if model.should_update(tick)
        ]

        print(
            "Updates in first second:",
            update_ticks,
        )

    print()
    print("=" * 72)
    print("DONE")
    print("=" * 72)


if __name__ == "__main__":
    main()
