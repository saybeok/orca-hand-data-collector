"""Encoder centering must agree with the driver's writable position frame."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "recenter_motor", Path(__file__).resolve().parents[2] / "scripts/recenter_motor_offset.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Motor:
    id = 15

    def __init__(self, position, after=2048):
        self.value = position
        self.after = after
        self.writes = []

    def position(self):
        return self.value

    def set_current_as(self, position):
        self.writes.append(position)
        self.value = self.after


def test_native_center_needs_no_eeprom_write():
    motor = Motor(2048)
    assert module.recenter(motor, apply=True)
    assert not motor.writes


def test_negative_center_alias_is_not_accepted_as_writable_center():
    motor = Motor(-2048, after=-2048)
    assert not module.recenter(motor, apply=True)
    assert motor.writes == [2048]


def test_recenter_can_recover_negative_reading_into_native_frame():
    motor = Motor(-3968)
    assert module.recenter(motor, apply=True)
    assert motor.value == 2048
