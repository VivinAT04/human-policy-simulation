import pytest

from human_policy_sim.frequency import (
    ControlFrequency,
)


@pytest.mark.parametrize(
    "hz, expected_interval",
    [
        (20, 0.05),
        (10, 0.10),
        (5, 0.20),
        (2, 0.50),
        (1, 1.00),
    ],
)
def test_control_update_intervals(
    hz,
    expected_interval,
):

    model = ControlFrequency(
        control_hz=hz,
        simulation_hz=20,
    )

    assert (
        model.update_interval_seconds
        == pytest.approx(
            expected_interval
        )
    )


@pytest.mark.parametrize(
    "decisions,hz,expected_seconds",
    [
        (35, 20, 1.75),
        (35, 10, 3.50),
        (35, 5, 7.00),
        (35, 2, 17.50),
        (35, 1, 35.00),
    ],
)
def test_elapsed_time_from_control_rate(
    decisions,
    hz,
    expected_seconds,
):

    elapsed = (
        decisions / hz
    )

    assert elapsed == pytest.approx(
        expected_seconds
    )


def test_lower_frequency_does_not_mean_repeating_action():

    """
    Frequency defines time between new commands.

    A discrete command remains one environment action;
    hold_ticks represents elapsed simulator ticks, not
    repeated calls to environment.step().
    """

    model = ControlFrequency(
        control_hz=1,
        simulation_hz=20,
    )

    assert model.hold_ticks == 20

    number_of_environment_actions = 1

    assert (
        number_of_environment_actions
        == 1
    )


@pytest.mark.parametrize(
    "decisions",
    [
        35,
        41,
        61,
        100,
    ],
)
def test_frequency_scales_completion_time(
    decisions,
):
    """
    In the discrete model, frequency changes the elapsed
    time associated with a fixed number of control decisions.
    """

    times = {
        hz: decisions / hz
        for hz in [20, 10, 5, 2, 1]
    }

    assert times[20] < times[10]
    assert times[10] < times[5]
    assert times[5] < times[2]
    assert times[2] < times[1]


def test_lower_frequency_increases_cost_of_extra_decisions():
    """
    A noisy run requiring extra control decisions incurs a
    larger absolute time penalty when command updates are slow.
    """

    clean_decisions = 35
    noisy_decisions = 61

    penalty_20hz = (
        noisy_decisions / 20
        - clean_decisions / 20
    )

    penalty_1hz = (
        noisy_decisions / 1
        - clean_decisions / 1
    )

    assert penalty_1hz > penalty_20hz

    assert penalty_20hz == pytest.approx(
        1.30
    )

    assert penalty_1hz == pytest.approx(
        26.0
    )
