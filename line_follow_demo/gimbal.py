from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

PWM_PULSE_MIN_US = 500
PWM_PULSE_MAX_US = 2500
TILT_PULSE_MIN_US = 500
TILT_PULSE_MAX_US = 1944
MANUAL_MOVE_DURATION_S = 0.08


@dataclass(frozen=True)
class GimbalConfig:
    enabled: bool = False
    pan_servo_id: int = 1
    tilt_servo_id: int = 2
    pan_pulse_us: int = 1500
    tilt_pulse_us: int = 1500
    move_duration_s: float = 1.0
    settle_time_s: float = 0.3

    @property
    def home_positions(self) -> list[list[int]]:
        return [
            [self.pan_servo_id, self.pan_pulse_us],
            [self.tilt_servo_id, self.tilt_pulse_us],
        ]


def _bounded_int(value: Any, name: str, lower: int, upper: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"gimbal.{name} must be an integer")
    if not lower <= value <= upper:
        raise ValueError(f"gimbal.{name} must be between {lower} and {upper}")
    return value


def _bounded_float(value: Any, name: str, lower: float, upper: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"gimbal.{name} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"gimbal.{name} must be numeric") from exc
    if not math.isfinite(number) or not lower <= number <= upper:
        raise ValueError(f"gimbal.{name} must be between {lower} and {upper}")
    return number


def validate_gimbal_position(pan_pulse_us: Any, tilt_pulse_us: Any) -> tuple[int, int]:
    pan = _bounded_int(pan_pulse_us, "pan_pulse_us", PWM_PULSE_MIN_US, PWM_PULSE_MAX_US)
    tilt = _bounded_int(tilt_pulse_us, "tilt_pulse_us", TILT_PULSE_MIN_US, TILT_PULSE_MAX_US)
    return pan, tilt


def gimbal_config_from_dict(values: Any) -> GimbalConfig:
    if values is None:
        values = {}
    if not isinstance(values, dict):
        raise ValueError("gimbal must be an object")

    enabled = values.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("gimbal.enabled must be a boolean")

    pan_servo_id = _bounded_int(values.get("pan_servo_id", 1), "pan_servo_id", 1, 4)
    tilt_servo_id = _bounded_int(values.get("tilt_servo_id", 2), "tilt_servo_id", 1, 4)
    if pan_servo_id == tilt_servo_id:
        raise ValueError("pan_servo_id and tilt_servo_id must be different PWM channels")

    return GimbalConfig(
        enabled=enabled,
        pan_servo_id=pan_servo_id,
        tilt_servo_id=tilt_servo_id,
        pan_pulse_us=_bounded_int(
            values.get("pan_pulse_us", 1500),
            "pan_pulse_us",
            PWM_PULSE_MIN_US,
            PWM_PULSE_MAX_US,
        ),
        tilt_pulse_us=_bounded_int(
            values.get("tilt_pulse_us", 1500),
            "tilt_pulse_us",
            TILT_PULSE_MIN_US,
            TILT_PULSE_MAX_US,
        ),
        move_duration_s=_bounded_float(
            values.get("move_duration_s", 1.0), "move_duration_s", 0.05, 10.0
        ),
        settle_time_s=_bounded_float(
            values.get("settle_time_s", 0.3), "settle_time_s", 0.0, 5.0
        ),
    )


class PwmGimbalController:
    def __init__(self, config: GimbalConfig):
        self._config = config

    @property
    def config(self) -> GimbalConfig:
        return self._config

    def command_home(self, board: Any) -> None:
        if not self._config.enabled:
            raise RuntimeError("Camera gimbal homing is disabled")
        set_positions = getattr(board, "set_pwm_servo_positions", None)
        if not callable(set_positions):
            raise RuntimeError("The selected motor driver does not support PWM servos")
        set_positions(
            duration_s=self._config.move_duration_s,
            positions=self._config.home_positions,
        )

    def command_position(
        self, board: Any, *, pan_pulse_us: Any, tilt_pulse_us: Any
    ) -> tuple[int, int]:
        if not self._config.enabled:
            raise RuntimeError("Camera gimbal control is disabled")
        pan, tilt = validate_gimbal_position(pan_pulse_us, tilt_pulse_us)
        set_positions = getattr(board, "set_pwm_servo_positions", None)
        if not callable(set_positions):
            raise RuntimeError("The selected motor driver does not support PWM servos")
        set_positions(
            duration_s=MANUAL_MOVE_DURATION_S,
            positions=[
                [self._config.pan_servo_id, pan],
                [self._config.tilt_servo_id, tilt],
            ],
        )
        return pan, tilt
