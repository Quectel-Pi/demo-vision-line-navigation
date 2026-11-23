"""MPU6050 sampling and short-term relative yaw measurements.

Angles use degrees, angular rates use degrees/second, and timestamps use
``time.monotonic()`` seconds. The integrated angle is relative to the most
recent explicit ``reset_turn()``; it is never an absolute heading.
"""

from __future__ import annotations

from collections import deque
import ctypes
from dataclasses import dataclass
import glob
import math
import statistics
import time
from typing import Callable

try:
    import fcntl
except ImportError:  # Linux board only; pure calculation tests also run on Windows.
    fcntl = None


I2C_SLAVE = 0x0703
I2C_FUNCS = 0x0705
I2C_RDWR = 0x0707
I2C_FUNC_I2C = 0x0001
I2C_M_RD = 0x0001
WHO_AM_I = 0x75
PWR_MGMT_1 = 0x6B
SMPLRT_DIV = 0x19
CONFIG = 0x1A
GYRO_CONFIG = 0x1B
ACCEL_CONFIG = 0x1C
ACCEL_XOUT_H = 0x3B

EXPECTED_CONFIGURATION = {
    PWR_MGMT_1: 0x01,
    SMPLRT_DIV: 0x04,
    CONFIG: 0x03,
    GYRO_CONFIG: 0x00,
    ACCEL_CONFIG: 0x00,
}


class _I2CMessage(ctypes.Structure):
    # Linux UAPI: struct i2c_msg, including native pointer alignment.
    _fields_ = [("addr", ctypes.c_uint16), ("flags", ctypes.c_uint16),
                ("len", ctypes.c_uint16), ("buf", ctypes.POINTER(ctypes.c_ubyte))]


class _I2CTransfer(ctypes.Structure):
    # Linux UAPI: struct i2c_rdwr_ioctl_data.
    _fields_ = [("msgs", ctypes.POINTER(_I2CMessage)), ("nmsgs", ctypes.c_uint32)]


class IMUError(RuntimeError):
    """The IMU cannot be read or its measurement is not trustworthy."""


class IMUSampleRejected(IMUError):
    """A rate/timing limit rejected a sample; later samples may be revalidated."""


@dataclass(frozen=True)
class IMUSample:
    timestamp_s: float
    accel_g: tuple[float, float, float]
    gyro_dps: tuple[float, float, float]


@dataclass(frozen=True)
class GyroBias:
    xyz_dps: tuple[float, float, float]
    max_spread_dps: float
    samples: int
    mean_accel_norm_g: float = 1.0
    max_accel_spread_g: float = 0.0


@dataclass(frozen=True)
class YawReading:
    timestamp_s: float
    gyro_axis_dps: float
    yaw_rate_dps: float
    delta_yaw_deg: float
    window_yaw_deg: float
    dt_s: float | None


class MPU6050:
    """Read an MPU6050 directly through Linux ``/dev/i2c-N``.

    The configured ranges are ±250 °/s and ±2 g. Address and bus are supplied
    by the caller; neither the board bus nor the vehicle yaw axis is assumed.
    """

    def __init__(
        self,
        bus: int,
        address: int = 0x68,
        *,
        gyro_scale_lsb_per_dps: float = 131.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if bus < 0 or address not in (0x68, 0x69):
            raise ValueError("bus must be non-negative and address must be 0x68 or 0x69")
        if not math.isfinite(gyro_scale_lsb_per_dps) or gyro_scale_lsb_per_dps <= 0:
            raise ValueError("gyro scale must be finite and positive")
        self.bus = bus
        self.address = address
        self.gyro_scale_lsb_per_dps = gyro_scale_lsb_per_dps
        self._clock = clock
        self._file = None
        self._configuration_fault: str | None = None

    def open(self) -> None:
        if fcntl is None:
            raise IMUError("MPU6050 I2C access requires Linux /dev/i2c-N")
        try:
            self._file = open(f"/dev/i2c-{self.bus}", "r+b", buffering=0)
            functions = ctypes.c_ulong()
            fcntl.ioctl(self._file.fileno(), I2C_FUNCS, functions)
            if not functions.value & I2C_FUNC_I2C:
                raise IMUError(f"/dev/i2c-{self.bus} does not support combined I2C transfers")
            fcntl.ioctl(self._file.fileno(), I2C_SLAVE, self.address)
        except IMUError:
            self.close()
            raise
        except OSError as exc:
            self.close()
            raise IMUError(f"cannot open /dev/i2c-{self.bus} at 0x{self.address:02x}: {exc}") from exc

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> "MPU6050":
        self.open()
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.close()

    def _write(self, register: int, value: int) -> None:
        if self._file is None:
            raise IMUError("MPU6050 is not open")
        try:
            written = self._file.write(bytes((register, value)))
        except OSError as exc:
            raise IMUError(f"I2C write at 0x{register:02x} failed: {exc}") from exc
        if written != 2:
            raise IMUError(f"I2C write at 0x{register:02x} was incomplete")

    def _read(self, register: int, length: int) -> bytes:
        if self._file is None:
            raise IMUError("MPU6050 is not open")
        if fcntl is None:
            raise IMUError("MPU6050 I2C access requires Linux /dev/i2c-N")
        if not 0 <= register <= 0xff or not 1 <= length <= 8192:
            raise ValueError("invalid I2C register or read length")
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                selected = (ctypes.c_ubyte * 1)(register)
                data = (ctypes.c_ubyte * length)()
                messages = (_I2CMessage * 2)(
                    _I2CMessage(self.address, 0, 1, selected),
                    _I2CMessage(self.address, I2C_M_RD, length, data),
                )
                request = _I2CTransfer(messages, 2)
                # One kernel transfer holds the bus through register selection
                # and the repeated-START read; another client cannot interleave.
                completed = fcntl.ioctl(self._file.fileno(), I2C_RDWR, request)
                if completed != 2:
                    raise IMUError(f"I2C combined transfer returned {completed}/2 messages")
                return bytes(data)
            except (OSError, IMUError) as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(0.002 * (attempt + 1))
        raise IMUError(f"I2C read at 0x{register:02x} failed: {last_error}") from last_error

    def who_am_i(self) -> int:
        return self._read(WHO_AM_I, 1)[0]

    def read_configuration(self) -> dict[int, int]:
        """Read settings without modifying hardware or clearing a latched fault."""
        block = self._read(SMPLRT_DIV, 4)
        power = self._read(PWR_MGMT_1, 1)
        if len(block) != 4 or len(power) != 1:
            raise IMUError("MPU6050 configuration readback was incomplete")
        return {PWR_MGMT_1: power[0], **{SMPLRT_DIV + index: value for index, value in enumerate(block)}}

    def verify_configuration(self) -> dict[int, int]:
        """Reject wrong range/timing/power settings; do not silently repair them."""
        if self._configuration_fault is not None:
            raise IMUError(f"MPU6050 configuration fault latched; initialize and recalibrate: {self._configuration_fault}")
        actual = self.read_configuration()
        mismatch = [f"0x{register:02x}: expected 0x{value:02x}, got 0x{actual[register]:02x}"
                    for register, value in EXPECTED_CONFIGURATION.items() if actual[register] != value]
        if mismatch:
            self._configuration_fault = "; ".join(mismatch)
            raise IMUError(f"MPU6050 configuration mismatch: {self._configuration_fault}")
        return actual

    def initialize(self) -> None:
        identity = self.who_am_i()
        if identity != 0x68:
            raise IMUError(f"WHO_AM_I=0x{identity:02x}; expected MPU6050 value 0x68")
        self._write(PWR_MGMT_1, 0x01)
        time.sleep(0.05)
        self._write(CONFIG, 0x03)  # DLPF, approximately 44 Hz gyro bandwidth.
        self._write(SMPLRT_DIV, 0x04)
        self._write(GYRO_CONFIG, 0x00)  # ±250 °/s: 131 LSB/(°/s).
        self._write(ACCEL_CONFIG, 0x00)  # ±2 g: 16384 LSB/g.
        time.sleep(0.05)
        self._configuration_fault = None
        self.verify_configuration()

    @staticmethod
    def _signed16(high: int, low: int) -> int:
        value = (high << 8) | low
        return value - 65536 if value >= 32768 else value

    def read_sample(self) -> IMUSample:
        """Read one burst: accelerometer in g and gyroscope in °/s."""
        self.verify_configuration()
        data = self._read(ACCEL_XOUT_H, 14)
        timestamp_s = self._clock()
        self.verify_configuration()
        raw = tuple(self._signed16(data[i], data[i + 1]) for i in range(0, 14, 2))
        accel = tuple(value / 16384.0 for value in raw[:3])
        gyro = tuple(value / self.gyro_scale_lsb_per_dps for value in raw[4:7])
        return IMUSample(timestamp_s=timestamp_s, accel_g=accel, gyro_dps=gyro)


def available_i2c_buses() -> tuple[int, ...]:
    """Return Linux I2C bus numbers; an empty tuple means none are visible."""
    buses: list[int] = []
    for path in glob.glob("/dev/i2c-*"):
        try:
            buses.append(int(path.rsplit("-", 1)[1]))
        except ValueError:
            continue
    return tuple(sorted(set(buses)))


def probe_mpu6050(buses: tuple[int, ...] | None = None) -> tuple[int, int]:
    """Find an MPU6050 by its exact WHO_AM_I value; never move motors."""
    candidates = available_i2c_buses() if buses is None else buses
    for bus in candidates:
        for address in (0x68, 0x69):
            try:
                with MPU6050(bus, address) as device:
                    if device.who_am_i() == 0x68:
                        return bus, address
            except IMUError:
                continue
    raise IMUError(f"MPU6050 not found on I2C buses {candidates}")


def scan_mpu6050(buses: tuple[int, ...] | None = None) -> tuple[tuple[int, int], ...]:
    """List every visible MPU6050 address instead of stopping at the first."""
    candidates = available_i2c_buses() if buses is None else buses
    found: list[tuple[int, int]] = []
    for bus in candidates:
        for address in (0x68, 0x69):
            try:
                with MPU6050(bus, address) as device:
                    if device.who_am_i() == 0x68:
                        found.append((bus, address))
            except IMUError:
                continue
    return tuple(found)


def calibrate_gyro_bias(
    device: MPU6050,
    *,
    samples: int = 200,
    sample_period_s: float = 0.01,
    max_abs_gyro_dps: float = 8.0,
    max_spread_dps: float = 3.0,
) -> GyroBias:
    """Measure XYZ gyro bias while the whole car is stationary.

    Any conspicuous motion rejects the calibration. Accelerometer stability is
    checked without assuming its uncalibrated norm equals exactly 1 g; it does
    not correct yaw drift.
    """
    if samples < 2 or not all(
        math.isfinite(value) for value in (sample_period_s, max_abs_gyro_dps, max_spread_dps)
    ) or sample_period_s < 0 or max_abs_gyro_dps <= 0 or max_spread_dps <= 0:
        raise ValueError("invalid calibration sample count, period, or motion limit")
    gyro_rows: list[tuple[float, float, float]] = []
    accel_rows: list[tuple[float, float, float]] = []
    accel_norms: list[float] = []
    for index in range(samples):
        sample = device.read_sample()
        if not all(math.isfinite(value) for value in (*sample.gyro_dps, *sample.accel_g)):
            raise IMUError("non-finite sample during gyro calibration")
        norm = math.sqrt(sum(value * value for value in sample.accel_g))
        if not 0.5 <= norm <= 1.5 or any(abs(value) > max_abs_gyro_dps for value in sample.gyro_dps):
            raise IMUError(
                f"calibration sample {index + 1}: gyro={sample.gyro_dps} °/s, "
                f"accel={sample.accel_g} g (norm={norm:.3f}); car moved or sensor data is invalid"
            )
        gyro_rows.append(sample.gyro_dps)
        accel_rows.append(sample.accel_g)
        accel_norms.append(norm)
        if index + 1 < samples and sample_period_s:
            time.sleep(sample_period_s)
    columns = tuple(tuple(row[axis] for row in gyro_rows) for axis in range(3))
    spread = max(max(column) - min(column) for column in columns)
    if spread > max_spread_dps:
        raise IMUError(f"gyro calibration spread {spread:.2f} °/s exceeds {max_spread_dps:.2f} °/s")
    accel_columns = tuple(tuple(row[axis] for row in accel_rows) for axis in range(3))
    accel_spread = max(max(column) - min(column) for column in accel_columns)
    if accel_spread > 0.1:
        raise IMUError(f"accelerometer changed {accel_spread:.3f} g during gyro calibration")
    return GyroBias(
        xyz_dps=tuple(statistics.median(column) for column in columns),
        max_spread_dps=spread,
        samples=samples,
        mean_accel_norm_g=statistics.mean(accel_norms),
        max_accel_spread_g=accel_spread,
    )


class YawEstimator:
    """Estimate yaw rate and short-term relative angle from a verified gyro axis."""

    def __init__(
        self,
        bias: GyroBias,
        *,
        axis: str = "z",
        positive_sign: int = 1,
        filter_alpha: float = 0.3,
        max_rate_dps: float = 220.0,
        max_dt_s: float = 0.2,
        window_s: float = 0.5,
    ) -> None:
        if axis not in ("x", "y", "z") or positive_sign not in (-1, 1):
            raise ValueError("axis must be x/y/z and positive_sign must be -1 or 1")
        if not all(math.isfinite(value) for value in (filter_alpha, max_rate_dps, max_dt_s, window_s)) or not (
            0 < filter_alpha <= 1 and max_rate_dps > 0 and max_dt_s > 0 and window_s > 0
        ):
            raise ValueError("filter and measurement limits must be positive")
        self.bias = bias
        self.axis = axis
        self.positive_sign = positive_sign
        self.filter_alpha = filter_alpha
        self.max_rate_dps = max_rate_dps
        self.max_dt_s = max_dt_s
        self.window_s = window_s
        self.delta_yaw_deg = 0.0
        self._last_timestamp_s: float | None = None
        self._filtered_rate_dps: float | None = None
        self._window: deque[tuple[float, float, float]] = deque()

    def reset_turn(self) -> None:
        """Zero only an explicitly started independent turn measurement."""
        self.delta_yaw_deg = 0.0

    def update(self, sample: IMUSample) -> YawReading:
        axis_index = ("x", "y", "z").index(self.axis)
        raw = sample.gyro_dps[axis_index]
        timestamp = sample.timestamp_s
        if not math.isfinite(timestamp) or not math.isfinite(raw) or not all(
            math.isfinite(value) for value in sample.accel_g
        ):
            self._last_timestamp_s = None
            self._filtered_rate_dps = None
            self._window.clear()
            raise IMUError("non-finite IMU sample")
        rate = (raw - self.bias.xyz_dps[axis_index]) * self.positive_sign
        if abs(rate) > self.max_rate_dps:
            self._last_timestamp_s = None
            self._filtered_rate_dps = None
            self._window.clear()
            raise IMUSampleRejected(f"gyro rate {rate:.1f} °/s exceeds configured maximum")

        previous = self._last_timestamp_s
        self._last_timestamp_s = timestamp
        if previous is not None:
            dt = timestamp - previous
            if dt <= 0 or dt > self.max_dt_s:
                self._filtered_rate_dps = None
                self._window.clear()
                raise IMUSampleRejected(f"invalid IMU sampling interval {dt:.3f} s")
        else:
            dt = None

        alpha = self.filter_alpha
        self._filtered_rate_dps = (
            rate
            if self._filtered_rate_dps is None
            else alpha * rate + (1.0 - alpha) * self._filtered_rate_dps
        )
        if dt is not None:
            increment = self._filtered_rate_dps * dt
            self.delta_yaw_deg += increment
            self._window.append((previous, timestamp, self._filtered_rate_dps))
        cutoff = timestamp - self.window_s
        while self._window and self._window[0][1] <= cutoff:
            self._window.popleft()
        return YawReading(
            timestamp_s=timestamp,
            gyro_axis_dps=raw,
            yaw_rate_dps=self._filtered_rate_dps,
            delta_yaw_deg=self.delta_yaw_deg,
            window_yaw_deg=sum(
                rate_dps * (end_s - max(start_s, cutoff))
                for start_s, end_s, rate_dps in self._window
            ),
            dt_s=dt,
        )
