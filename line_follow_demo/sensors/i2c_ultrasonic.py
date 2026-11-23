from __future__ import annotations

from .base import DistanceSensor


class I2CUltrasonicSensor(DistanceSensor):
    def __init__(self, bus: int, address: int = 0x77):
        self._bus_number = bus
        self._address = address
        self._bus = None

    def open(self) -> None:
        from smbus2 import SMBus

        self._bus = SMBus(self._bus_number)
        self._bus.write_byte_data(self._address, 2, 0)
        for register in range(3, 9):
            self._bus.write_byte_data(self._address, register, 0)

    def read_distance_mm(self) -> int | None:
        if self._bus is None:
            return None
        from smbus2 import i2c_msg

        try:
            select = i2c_msg.write(self._address, [0])
            self._bus.i2c_rdwr(select)
            value = i2c_msg.read(self._address, 2)
            self._bus.i2c_rdwr(value)
            return int.from_bytes(bytes(list(value)), "little")
        except OSError:
            return None

    def close(self) -> None:
        if self._bus is not None:
            self._bus.close()
            self._bus = None
