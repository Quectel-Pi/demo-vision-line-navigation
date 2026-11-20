from __future__ import annotations

from .base import MotorDriver, MotorInversion, MotorMapping
from .dry_run import DryRunDriver
from .hiwonder_model_a import HiwonderModelAConfig, HiwonderModelADriver


def _mapping_from_dict(d: dict) -> MotorMapping:
    if not isinstance(d, dict):
        return MotorMapping()
    return MotorMapping(
        fl=int(d.get("fl", 1)),
        fr=int(d.get("fr", 2)),
        rl=int(d.get("rl", 3)),
        rr=int(d.get("rr", 4)),
    )


def _invert_from_dict(d: dict) -> MotorInversion:
    if not isinstance(d, dict):
        return MotorInversion()
    return MotorInversion(
        fl=bool(d.get("fl", False)),
        fr=bool(d.get("fr", False)),
        rl=bool(d.get("rl", False)),
        rr=bool(d.get("rr", False)),
    )


def model_a_config_from_dict(cfg: dict) -> HiwonderModelAConfig:
    return HiwonderModelAConfig(
        serial_device=str(cfg.get("serial_device", "/dev/ttyS1")),
        baudrate=int(cfg.get("baudrate", 1_000_000)),
        duty_limit=float(cfg.get("duty_limit", 80.0)),
        deadzone_pct=float(cfg.get("deadzone_pct", 15.0)),
        apply_deadzone=bool(cfg.get("apply_deadzone", True)),
        mapping=_mapping_from_dict(cfg.get("mapping", {})),
        invert=_invert_from_dict(cfg.get("invert", {})),
        trace_commands=bool(cfg.get("trace_commands", True)),
        trace_interval_s=max(0.1, float(cfg.get("trace_interval_s", 0.25))),
    )


def create_motor_driver(cfg: dict, *, force_dry_run: bool = False) -> MotorDriver:
    if force_dry_run:
        return DryRunDriver()

    driver = str(cfg.get("driver", "hiwonder_model_a"))
    if driver == "hiwonder_model_a":
        return HiwonderModelADriver(model_a_config_from_dict(cfg))

    if driver == "dry_run":
        return DryRunDriver()

    raise ValueError(f"Unknown motor driver: {driver}")
