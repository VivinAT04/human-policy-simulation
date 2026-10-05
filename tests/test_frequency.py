import pytest

from human_policy_sim.frequency import (
    ControlFrequency,
    DEFAULT_SIMULATION_HZ,
    SUPPORTED_FREQUENCIES,
)


@pytest.mark.parametrize(
    "frequency, expected_ticks",
    [
        (20, 1),
        (10, 2),
        (5, 4),
        (2, 10),
        (1, 20),
    ],
)
def test_hold_ticks(
    frequency,
    expected_ticks,
):

    model = ControlFrequency(
        control_hz=frequency,
        simulation_hz=20,
    )

    assert (
        model.hold_ticks
        == expected_ticks
    )


def test_default_simulation_rate():

    assert DEFAULT_SIMULATION_HZ == 20


def test_supported_frequencies():

    assert SUPPORTED_FREQUENCIES == (
        20,
        10,
        5,
        2,
        1,
    )


def test_twenty_hz_updates_every_tick():

    model = ControlFrequency(
        control_hz=20,
    )

    updates = [
        model.should_update(tick)
        for tick in range(10)
    ]

    assert updates == [
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
    ]


def test_ten_hz_updates_every_two_ticks():

    model = ControlFrequency(
        control_hz=10,
    )

    updates = [
        model.should_update(tick)
        for tick in range(8)
    ]

    assert updates == [
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
    ]


def test_five_hz_updates_every_four_ticks():

    model = ControlFrequency(
        control_hz=5,
    )

    update_ticks = [
        tick
        for tick in range(13)
        if model.should_update(tick)
    ]

    assert update_ticks == [
        0,
        4,
        8,
        12,
    ]


def test_two_hz_updates_every_ten_ticks():

    model = ControlFrequency(
        control_hz=2,
    )

    update_ticks = [
        tick
        for tick in range(31)
        if model.should_update(tick)
    ]

    assert update_ticks == [
        0,
        10,
        20,
        30,
    ]


def test_one_hz_updates_every_twenty_ticks():

    model = ControlFrequency(
        control_hz=1,
    )

    update_ticks = [
        tick
        for tick in range(61)
        if model.should_update(tick)
    ]

    assert update_ticks == [
        0,
        20,
        40,
        60,
    ]


def test_decision_index():

    model = ControlFrequency(
        control_hz=5,
    )

    assert model.decision_index(0) == 0
    assert model.decision_index(1) == 0
    assert model.decision_index(3) == 0

    assert model.decision_index(4) == 1
    assert model.decision_index(7) == 1

    assert model.decision_index(8) == 2


def test_update_interval_seconds():

    assert (
        ControlFrequency(20)
        .update_interval_seconds
        == pytest.approx(0.05)
    )

    assert (
        ControlFrequency(10)
        .update_interval_seconds
        == pytest.approx(0.1)
    )

    assert (
        ControlFrequency(5)
        .update_interval_seconds
        == pytest.approx(0.2)
    )

    assert (
        ControlFrequency(2)
        .update_interval_seconds
        == pytest.approx(0.5)
    )

    assert (
        ControlFrequency(1)
        .update_interval_seconds
        == pytest.approx(1.0)
    )


@pytest.mark.parametrize(
    "frequency",
    [
        0,
        -1,
    ],
)
def test_invalid_control_frequency(
    frequency,
):

    with pytest.raises(ValueError):
        ControlFrequency(
            control_hz=frequency
        )


def test_control_frequency_cannot_exceed_simulation_rate():

    with pytest.raises(ValueError):
        ControlFrequency(
            control_hz=21,
            simulation_hz=20,
        )


def test_frequency_must_divide_simulation_rate():

    with pytest.raises(ValueError):
        ControlFrequency(
            control_hz=3,
            simulation_hz=20,
        )


def test_invalid_simulation_frequency():

    with pytest.raises(ValueError):
        ControlFrequency(
            control_hz=1,
            simulation_hz=0,
        )


def test_negative_tick_rejected():

    model = ControlFrequency(
        control_hz=10,
    )

    with pytest.raises(ValueError):
        model.should_update(-1)

    with pytest.raises(ValueError):
        model.decision_index(-1)
