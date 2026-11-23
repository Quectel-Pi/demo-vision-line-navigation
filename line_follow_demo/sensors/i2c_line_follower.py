from __future__ import annotations


class I2CLineFollowerSensor:
    def __init__(self, bus: int, address: int = 0x78):
        self._bus_number = bus
        self._address = address
        self._bus = None

    def open(self) -> None:
        from smbus2 import SMBus

        self._bus = SMBus(self._bus_number)

    def read_channels(self) -> tuple[bool, bool, bool, bool] | None:
        if self._bus is None:
            return None
        try:
            value = self._bus.read_byte_data(self._address, 1)
        except OSError:
            return None
        return tuple(bool(value & (1 << bit)) for bit in range(4))

    def close(self) -> None:
        if self._bus is not None:
            self._bus.close()
            self._bus = None
