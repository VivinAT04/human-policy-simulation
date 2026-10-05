"""Control-update frequency model for the discrete wheelchair simulation.

In the discrete-grid case study, control frequency represents the rate at
which a new physiological/BCI command becomes available.

Each received command produces exactly one discrete environment action.
The wheelchair does not repeatedly execute LEFT, RIGHT, FORWARD or STOP
between command arrivals.

Therefore, in this discrete model:

    noise     -> affects action accuracy, trajectory and reward
    frequency -> affects command latency and completion time

A future continuous-dynamics model can instead represent persistent
linear/angular velocity between control updates.
"""

from __future__ import annotations


DEFAULT_SIMULATION_HZ = 20

SUPPORTED_FREQUENCIES = (
    20,
    10,
    5,
    2,
    1,
)


class ControlFrequency:
    """
    Convert a control-update frequency into a discrete command hold.

    The environment is treated as running at ``simulation_hz`` ticks
    per second. A new policy decision is produced at ``control_hz``.

    Examples with a 20 Hz simulation:

        20 Hz -> hold command for 1 tick
        10 Hz -> hold command for 2 ticks
         5 Hz -> hold command for 4 ticks
         2 Hz -> hold command for 10 ticks
         1 Hz -> hold command for 20 ticks
    """

    def __init__(
        self,
        control_hz: int,
        simulation_hz: int = DEFAULT_SIMULATION_HZ,
    ) -> None:

        if simulation_hz <= 0:
            raise ValueError(
                "simulation_hz must be greater than zero"
            )

        if control_hz <= 0:
            raise ValueError(
                "control_hz must be greater than zero"
            )

        if control_hz > simulation_hz:
            raise ValueError(
                "control_hz cannot exceed simulation_hz"
            )

        if simulation_hz % control_hz != 0:
            raise ValueError(
                "simulation_hz must be exactly divisible "
                "by control_hz"
            )

        self.control_hz = control_hz
        self.simulation_hz = simulation_hz

        self.hold_ticks = (
            simulation_hz // control_hz
        )

    def should_update(
        self,
        tick: int,
    ) -> bool:
        """Return True when a new policy decision is required."""

        if tick < 0:
            raise ValueError(
                "tick cannot be negative"
            )

        return (
            tick % self.hold_ticks == 0
        )

    def decision_index(
        self,
        tick: int,
    ) -> int:
        """Return the control-decision index for a simulation tick."""

        if tick < 0:
            raise ValueError(
                "tick cannot be negative"
            )

        return tick // self.hold_ticks

    @property
    def update_interval_seconds(
        self,
    ) -> float:
        """Time between control updates in seconds."""

        return 1.0 / self.control_hz
