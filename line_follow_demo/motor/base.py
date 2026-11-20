from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MotorMapping:
    fl: int = 1
    fr: int = 2
    rl: int = 3
    rr: int = 4


@dataclass(frozen=True)
class MotorInversion:
    fl: bool = False
    fr: bool = False
    rl: bool = False
    rr: bool = False


class MotorDriver:
    def open(self) -> None:
        raise NotImplementedError

    def read_battery_voltage_v(self) -> float | None:
        return None

    def set_wheel_duty(self, *, fl: float, fr: float, rl: float, rr: float) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        self.set_wheel_duty(fl=0.0, fr=0.0, rl=0.0, rr=0.0)

    def close(self) -> None:
        pass
