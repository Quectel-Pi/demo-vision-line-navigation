from __future__ import annotations

from dataclasses import dataclass
import logging
import hashlib
import math
from pathlib import Path
import sys
import time

from .base import MotorDriver, MotorInversion, MotorMapping


log = logging.getLogger(__name__)
MODEL_A_COMMAND_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _load_vendor_sdk():
    here = Path(__file__).resolve()
    vendor_dir = here.parents[2] / "vendor" / "hiwonder"
    sys.path.insert(0, str(vendor_dir))
    try:
        import ros_robot_controller_sdk as rrc  # type: ignore
    finally:
        # 仅为 import 临时插入；避免污染全局 import 顺序
        try:
            sys.path.remove(str(vendor_dir))
        except ValueError:
            pass
    return rrc


@dataclass(frozen=True)
class HiwonderModelAConfig:
    serial_device: str = "/dev/ttyS1"
    baudrate: int = 1_000_000
    duty_limit: float = 80.0
    deadzone_pct: float = 15.0
    apply_deadzone: bool = True
    mapping: MotorMapping = MotorMapping()
    invert: MotorInversion = MotorInversion()
    trace_commands: bool = False
    trace_interval_s: float = 0.25


def model_a_wheel_commands(cfg, values):
    commands, clipped = [], False
    for name, requested in zip(("fl", "fr", "rl", "rr"), values):
        duty = -float(requested) if getattr(cfg.invert, name) else float(requested)
        bounded = max(-100., min(100., duty))
        clipped |= bounded != duty
        duty = bounded
        if cfg.apply_deadzone and duty != 0.:
            deadzone = max(0., min(95., float(cfg.deadzone_pct)))
            duty = math.copysign(deadzone+abs(duty)*(100.-deadzone)/100., duty)
        if cfg.duty_limit > 0:
            bounded = max(-float(cfg.duty_limit), min(float(cfg.duty_limit), duty))
            clipped |= bounded != duty
            duty = bounded
        commands.append([int(getattr(cfg.mapping, name)), float(duty)])
    return commands, clipped


class HiwonderModelADriver(MotorDriver):
    def __init__(self, cfg: HiwonderModelAConfig):
        self._cfg = cfg
        self._rrc = _load_vendor_sdk()
        self._board = None
        self._last_trace_at = 0.0

    def open(self) -> None:
        self._board = self._rrc.Board(device=self._cfg.serial_device, baudrate=self._cfg.baudrate)
        self._board.enable_reception()

    def read_battery_voltage_v(self) -> float | None:
        if self._board is None:
            return None
        millivolts = self._board.get_battery()
        if millivolts is None or millivolts <= 0:
            return None
        return float(millivolts) / 1000.0

    def set_wheel_duty(self, *, fl: float, fr: float, rl: float, rr: float) -> None:
        if self._board is None:
            raise RuntimeError("Motor driver not opened")

        requested_values = (float(fl), float(fr), float(rl), float(rr))
        board_commands, _ = model_a_wheel_commands(self._cfg, requested_values)
        self._board.set_motor_duty(board_commands)

        if self._cfg.trace_commands:
            now = time.monotonic()
            interval = max(0.1, float(self._cfg.trace_interval_s))
            if now - self._last_trace_at >= interval:
                self._last_trace_at = now
                wheel_names = ("FL", "FR", "RL", "RR")
                requested_text = ",".join(
                    f"{name}:{value:+.1f}"
                    for name, value in zip(wheel_names, requested_values)
                )
                board_text = ",".join(
                    f"M{channel}:{value:+.1f}" for channel, value in board_commands
                )
                log.info(
                    "MOTOR_TRACE requested=[%s] board=[%s] invert=[FL:%s,FR:%s,RL:%s,RR:%s]",
                    requested_text,
                    board_text,
                    self._cfg.invert.fl,
                    self._cfg.invert.fr,
                    self._cfg.invert.rl,
                    self._cfg.invert.rr,
                )

    def stop(self) -> None:
        for attempt in range(4):
            self.set_wheel_duty(fl=0.0, fr=0.0, rl=0.0, rr=0.0)
            if attempt < 3:
                time.sleep(0.03)

    def set_pwm_servo_positions(
        self, *, duration_s: float, positions: list[list[int]]
    ) -> None:
        if self._board is None:
            raise RuntimeError("Motor driver not opened")
        self._board.pwm_servo_set_position(float(duration_s), positions)

    def close(self) -> None:
        try:
            if self._board is not None:
                self.stop()
        finally:
            self._board = None
