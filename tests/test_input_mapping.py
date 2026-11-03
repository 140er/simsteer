import pytest

from simsteer.runtime.input.mapping import AxisMap, PedalMapper, resolve_slots, validate_axes


@pytest.mark.parametrize("raw,expected", [(-1, -1), (0, -1), (32767.5, 0), (65535, 1), (99999, 1)])
def test_steering_identity_clamped(raw, expected):
    assert AxisMap().steering([raw]) == expected


@pytest.mark.parametrize("rest,full", [(0, 32767), (32767, 0), (65535, 32767)])
def test_pedal_direction(rest, full):
    mapping = {"slot": 0, "rest": rest, "full": full}
    for fraction in (0, .25, 1):
        assert AxisMap.mapped_pedal([rest + fraction * (full - rest)], mapping) == pytest.approx(fraction)
    assert AxisMap.mapped_pedal([full], None) is None


@pytest.mark.parametrize("names,expected", [
    ({0: "Wheel Axis", 1: "Accelerator", 2: "Brake", 3: "Clutch", 4: "Combined Pedals"},
     dict(steer=0, throttle=1, brake=2, clutch=3)),
    ({0: "Wheel Axis", 1: "Z Axis", 4: "Accelerator", 5: "Brake", 6: "Clutch"},
     dict(steer=0, throttle=4, brake=5, clutch=6)),
    ({1: "Combined Pedals"}, dict(steer=0)),
])
def test_names_win_over_slots(names, expected):
    assert resolve_slots(names) == expected


def test_explicit_mapping_requires_a_press_per_required_pedal():
    names = {0: "Wheel Axis", 1: "Accelerator", 2: "Brake", 3: "Clutch", 4: "Combined Pedals"}
    ranges = {i: (0, 100) for i in range(8)}
    mapper = PedalMapper(0, [0] * 8, names, ranges, 0)
    mapper.step(1, [0, 100, 0, 0, 0, 0, 0, 0])
    mapper.step(3, [0] * 8)
    mapper.step(4, [0, 0, 100, 0, 0, 0, 0, 0])
    mapper.step(6, [0] * 8)
    mapper.step(9, [0] * 8)  # Optional clutch absent.
    assert mapper.done and set(mapper.result) == {"steer", "throttle", "brake"}
    assert validate_axes(mapper.result) == mapper.result
    failed = PedalMapper(0, [0] * 8, names, ranges, 0)
    failed.step(3, [0] * 8)
    assert failed.done and failed.result == {}


@pytest.mark.parametrize("mapping", [None, [], {"throttle": {"slot": 0, "rest": 0, "full": 1}},
    {"steer": []}, {"throttle": {"slot": 1, "rest": 1, "full": 1}},
    {"brake": {"slot": 1, "rest": float("nan"), "full": 0}},
    {"brake": {"slot": 8, "rest": 1, "full": 0}}, {"throttle": {"slot": 1}}])
def test_corrupt_mapping_rejected(mapping):
    assert validate_axes(mapping) == {}
