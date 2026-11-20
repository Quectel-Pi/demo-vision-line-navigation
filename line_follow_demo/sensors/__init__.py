from __future__ import annotations

from .base import DistanceSensor
from .i2c_line_follower import I2CLineFollowerSensor
from .i2c_ultrasonic import I2CUltrasonicSensor
from .serial_ultrasonic import SerialUltrasonicSensor


class NullDistanceSensor(DistanceSensor):
    def open(self) -> None:
        pass

    def read_distance_mm(self) -> int | None:
        return None


def create_distance_sensor(cfg: dict) -> DistanceSensor:
    enabled = bool(cfg.get("enabled", False))
    if not enabled:
        return NullDistanceSensor()

    kind = str(cfg.get("type", "serial_ultrasonic"))
    if kind == "serial_ultrasonic":
        device = str(cfg.get("serial_device", "/dev/ttyUSB0"))
        baudrate = int(cfg.get("baudrate", 9600))
        return SerialUltrasonicSensor(device=device, baudrate=baudrate)
    if kind == "i2c_ultrasonic":
        if cfg.get("bus") is None:
            raise ValueError("ultrasonic.bus must be set to the M2 I2C bus number")
        return I2CUltrasonicSensor(bus=int(cfg["bus"]), address=int(cfg.get("address", 0x77)))

    raise ValueError(f"Unknown distance sensor type: {kind}")


def create_line_sensor(cfg: dict) -> I2CLineFollowerSensor | None:
    if not cfg.get("enabled", False):
        return None
    if cfg.get("type", "i2c_line_follower") != "i2c_line_follower":
        raise ValueError("Unknown line sensor type")
    if cfg.get("bus") is None:
        raise ValueError("line_sensor.bus must be set to the M2 I2C bus number")
    return I2CLineFollowerSensor(bus=int(cfg["bus"]), address=int(cfg.get("address", 0x78)))
