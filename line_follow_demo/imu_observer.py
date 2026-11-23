"""Independent IMU sampling and queued logs; the runtime owns motor feedback."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import math
import logging
from pathlib import Path
import queue
import threading
import time
from typing import Any, Callable
from uuid import uuid4

from .imu import GyroBias, IMUError, IMUSample, IMUSampleRejected, MPU6050, YawEstimator, YawReading, calibrate_gyro_bias, probe_mpu6050


RECOVERY_VALID_SAMPLES = 10


@dataclass(frozen=True)
class IMUObserverConfig:
    mode: str = "vision"
    bus: int | None = None
    address: int = 0x68
    axis: str = "z"
    positive_sign: int = 1
    axis_verified: bool = False
    gyro_scale_lsb_per_dps: float = 131.0
    calibration_samples: int = 200
    filter_alpha: float = 0.3
    sample_hz: float = 50.0
    max_rate_dps: float = 220.0
    max_dt_s: float = 0.2
    stale_after_s: float = 0.2
    window_s: float = 0.5
    log_dir: str = "../logs"

    @property
    def enabled(self) -> bool:
        return self.mode in {"observe", "feedback"}

    def __post_init__(self) -> None:
        if self.mode not in {"vision", "observe", "feedback"}:
            raise ValueError("imu.mode must be vision, observe or feedback")
        if self.axis not in {"x", "y", "z"} or self.positive_sign not in (-1, 1):
            raise ValueError("invalid IMU yaw axis or sign")
        if self.enabled and not self.axis_verified:
            raise ValueError("verify IMU axis/sign by hand before enabling IMU mode")
        if self.address not in (0x68, 0x69) or (self.bus is not None and self.bus < 0):
            raise ValueError("invalid IMU bus/address")
        numbers = (self.gyro_scale_lsb_per_dps, self.filter_alpha, self.sample_hz,
                   self.max_rate_dps, self.max_dt_s, self.stale_after_s, self.window_s)
        if not all(math.isfinite(value) and value > 0 for value in numbers) or not (
            self.filter_alpha <= 1 and self.sample_hz <= 200 and self.calibration_samples >= 2
            and self.stale_after_s >= 1.0 / self.sample_hz
        ):
            raise ValueError("invalid IMU sampling, calibration or freshness limits")
        if not self.log_dir:
            raise ValueError("imu.log_dir must not be empty")


def imu_observer_config_from_dict(data: dict[str, Any]) -> IMUObserverConfig:
    """Load known sampler settings; feedback additionally requires runtime gates."""
    allowed = IMUObserverConfig.__dataclass_fields__
    unknown = set(data) - set(allowed)
    if unknown:
        raise ValueError(f"unknown IMU settings: {', '.join(sorted(unknown))}")
    values = dict(data)
    for key in ("axis_verified",):
        if key in values and not isinstance(values[key], bool):
            raise ValueError(f"imu.{key} must be a boolean")
    for key in ("bus", "address", "positive_sign", "calibration_samples"):
        if key in values and values[key] is not None:
            if isinstance(values[key], bool) or not isinstance(values[key], int):
                raise ValueError(f"imu.{key} must be an integer")
    return IMUObserverConfig(**values)


OBSERVATION_FIELDS = (
    "timestamp", "vision_timestamp_monotonic_s", "imu_timestamp_monotonic_s", "imu_age_s",
    "lateral_error", "heading_error", "curvature", "visual_turn", "visual_turn_hint",
    "gyro_x", "gyro_y", "gyro_z", "yaw_rate", "delta_yaw", "window_yaw",
    "target_yaw_rate", "yaw_rate_error", "final_turn",
    "motor_1", "motor_2", "motor_3", "motor_4",
    "wheel_fl_command", "wheel_fr_command", "wheel_rl_command", "wheel_rr_command",
    "forward_duty_command", "turn_duty_command", "path_state", "vision_state",
    "runtime_state", "armed", "imu_state", "imu_error", "imu_dt_s",
    "imu_recovery_count", "imu_recovery_valid_samples", "imu_last_recovery_reason",
)


class IMUObserver:
    """Sample independently of camera FPS; return fresh data or explicit blanks.

    Relative angles start at service initialization. The window angle is the
    useful short-term quantity; accumulated angle is diagnostic, not heading.
    CSV writes run on another thread, never in the vision/control loop.
    """

    def __init__(
        self, config: IMUObserverConfig, *, config_directory: Path,
        device_factory: Callable[..., MPU6050] = MPU6050,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self._directory = config_directory
        self._device_factory = device_factory
        self._clock = clock
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._sampler: threading.Thread | None = None
        self._logger: threading.Thread | None = None
        self._rows: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=256)
        self._state = "STOPPED" if config.enabled else "DISABLED"
        self._error = ""
        self._latest: tuple[IMUSample, YawReading] | None = None
        self._bias: GyroBias | None = None
        self._log_path: Path | None = None
        self._recovery_count = 0
        self._recovery_valid_samples = 0
        self._last_recovery_reason = ""

    def start(self) -> None:
        if not self.config.enabled or (self._sampler is not None and self._sampler.is_alive()):
            return
        self._stop.clear()
        self._rows = queue.Queue(maxsize=256)
        with self._lock:
            self._state, self._error, self._latest, self._bias = "CALIBRATING", "", None, None
            self._recovery_count = self._recovery_valid_samples = 0
            self._last_recovery_reason = ""
            suffix = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8]
            self._log_path = (self._directory / self.config.log_dir / f"vision_imu_{suffix}.csv").resolve()
        self._logger = threading.Thread(target=self._write_rows, name="imu-csv", daemon=True)
        self._sampler = threading.Thread(target=self._sample, name="imu-sampling", daemon=True)
        self._logger.start()
        self._sampler.start()

    def stop(self) -> None:
        self._stop.set()
        for worker in (self._sampler, self._logger):
            if worker is not None and worker is not threading.current_thread():
                worker.join(timeout=2.0)
        with self._lock:
            self._latest = None
            if self._state != "ERROR":
                self._state = "STOPPED" if self.config.enabled else "DISABLED"

    def _fail(self, exc: Exception) -> None:
        with self._lock:
            self._stop.set()
            self._state, self._error, self._latest = "ERROR", str(exc), None
        logging.getLogger(__name__).error("IMU observation failed: %s", exc)
        try:
            self._rows.put_nowait({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "imu_state": "ERROR", "imu_error": str(exc),
            })
        except queue.Full:
            pass

    def _sample(self) -> None:
        cfg = self.config
        try:
            bus, address = probe_mpu6050() if cfg.bus is None else (cfg.bus, cfg.address)
            with self._device_factory(bus, address, gyro_scale_lsb_per_dps=cfg.gyro_scale_lsb_per_dps) as device:
                device.initialize()
                bias = calibrate_gyro_bias(device, samples=cfg.calibration_samples)
                def new_estimator():
                    return YawEstimator(
                        bias, axis=cfg.axis, positive_sign=cfg.positive_sign,
                        filter_alpha=cfg.filter_alpha, max_rate_dps=cfg.max_rate_dps,
                        max_dt_s=cfg.max_dt_s, window_s=cfg.window_s,
                    )

                estimator = new_estimator()
                recovering = False
                with self._lock:
                    self._bias = bias
                while not self._stop.is_set():
                    sample = device.read_sample()
                    if not all(math.isfinite(value) for value in (*sample.gyro_dps, *sample.accel_g)):
                        raise IMUError("non-finite IMU data")
                    try:
                        reading = estimator.update(sample)
                    except IMUSampleRejected as exc:
                        # Keep sampling verified hardware. Never integrate or
                        # publish the rejected sample, or bridge its angle gap.
                        with self._lock:
                            if self._stop.is_set():
                                break
                            if not recovering:
                                self._recovery_count += 1
                            self._recovery_valid_samples = 0
                            self._last_recovery_reason = str(exc)
                            self._state, self._error, self._latest = "RECOVERING", str(exc), None
                        if not recovering:
                            logging.getLogger(__name__).warning("IMU sample rejected; revalidating: %s", exc)
                            self._record_recovery_event("RECOVERING", str(exc))
                        recovering = True
                        estimator = new_estimator()
                        self._stop.wait(1.0 / cfg.sample_hz)
                        continue
                    recovered = False
                    with self._lock:
                        if not self._stop.is_set():
                            if recovering:
                                self._recovery_valid_samples += 1
                                if self._recovery_valid_samples >= RECOVERY_VALID_SAMPLES:
                                    recovering = False
                                    recovered = True
                            if not recovering:
                                self._latest = (sample, reading)
                                self._state, self._error = "READY", ""
                    if recovered:
                        self._record_recovery_event("READY", "")
                        logging.getLogger(__name__).info("IMU recovered after %d valid samples; angle restarted, tracking requires explicit Start", RECOVERY_VALID_SAMPLES)
                    self._stop.wait(1.0 / cfg.sample_hz)
        except Exception as exc:
            self._fail(exc)

    def _record_recovery_event(self, state: str, error: str) -> None:
        try:
            self._rows.put_nowait({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "imu_state": state, "imu_error": error,
                "imu_recovery_count": self._recovery_count,
                "imu_recovery_valid_samples": self._recovery_valid_samples,
                "imu_last_recovery_reason": self._last_recovery_reason,
            })
        except queue.Full as exc:
            raise IMUError("IMU recovery event log queue is full") from exc

    def snapshot(self) -> dict[str, Any]:
        """Return serializable telemetry; invalid/stale rates and angles are None."""
        with self._lock:
            latest, bias, state, error, path = self._latest, self._bias, self._state, self._error, self._log_path
            recovery_count = self._recovery_count
            valid_samples = self._recovery_valid_samples
            last_reason = self._last_recovery_reason
        now = self._clock()
        age = None if latest is None else now - latest[0].timestamp_s
        if state == "READY" and (age is None or not 0 <= age <= self.config.stale_after_s):
            state, error = "STALE", "IMU samples are stale or timestamps are invalid"
        healthy = state == "READY"
        sample, reading = latest if healthy and latest is not None else (None, None)
        return {
            "imu_mode": self.config.mode, "imu_feedback_enabled": False,
            "imu_state": state, "imu_healthy": healthy, "imu_error": error,
            "imu_axis": self.config.axis, "imu_positive_sign": self.config.positive_sign,
            "imu_timestamp_monotonic_s": None if latest is None else latest[0].timestamp_s,
            "imu_age_s": age,
            "imu_gyro_dps": None if sample is None else sample.gyro_dps,
            "imu_accel_g": None if sample is None else sample.accel_g,
            "imu_yaw_rate_dps": None if reading is None else reading.yaw_rate_dps,
            "imu_delta_yaw_deg": None if reading is None else reading.delta_yaw_deg,
            "imu_window_yaw_deg": None if reading is None else reading.window_yaw_deg,
            "imu_dt_s": None if reading is None else reading.dt_s,
            "imu_bias_dps": None if bias is None else bias.xyz_dps,
            "imu_calibration_samples": None if bias is None else bias.samples,
            "imu_accel_norm_g": None if bias is None else bias.mean_accel_norm_g,
            "imu_log_path": None if path is None else str(path),
            "imu_recovery_count": recovery_count,
            "imu_recovery_valid_samples": valid_samples,
            "imu_recovery_required_samples": RECOVERY_VALID_SAMPLES,
            "imu_last_recovery_reason": last_reason,
        }

    def record_frame(self, status: dict[str, Any], imu: dict[str, Any]) -> None:
        if not self.config.enabled or self._stop.is_set():
            return
        gyro = imu.get("imu_gyro_dps") or (None, None, None)
        wheels = status.get("wheel_duty_commands", {})
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "vision_timestamp_monotonic_s": status.get("vision_timestamp_monotonic_s"),
            "imu_timestamp_monotonic_s": imu.get("imu_timestamp_monotonic_s"),
            "imu_age_s": imu.get("imu_age_s"),
            "lateral_error": status.get("error"),
            "heading_error": status.get("track_heading_deg"),
            "curvature": status.get("path_curvature_px_inv"),
            "visual_turn": status.get("visual_turn", status.get("steering_output")),
            "visual_turn_hint": status.get("visual_turn_hint"),
            "gyro_x": gyro[0], "gyro_y": gyro[1], "gyro_z": gyro[2],
            "yaw_rate": imu.get("imu_yaw_rate_dps"),
            "delta_yaw": imu.get("imu_delta_yaw_deg"),
            "window_yaw": imu.get("imu_window_yaw_deg"),
            "target_yaw_rate": status.get("target_yaw_rate_dps"),
            "yaw_rate_error": status.get("yaw_rate_error_dps"),
            "final_turn": status.get("steering_output"),
            "forward_duty_command": status.get("forward_duty_command"),
            "turn_duty_command": status.get("turn_duty_command"),
            "path_state": status.get("track_state"), "vision_state": status.get("vision_state"),
            "runtime_state": status.get("state"), "armed": status.get("armed"),
            "imu_state": imu.get("imu_state"), "imu_error": imu.get("imu_error"),
            "imu_dt_s": imu.get("imu_dt_s"),
            "imu_recovery_count": imu.get("imu_recovery_count"),
            "imu_recovery_valid_samples": imu.get("imu_recovery_valid_samples"),
            "imu_last_recovery_reason": imu.get("imu_last_recovery_reason"),
            **{f"wheel_{key.lower()}_command": wheels.get(key) for key in ("FL", "FR", "RL", "RR")},
        }
        try:
            self._rows.put_nowait(row)
        except queue.Full:
            self._fail(IMUError("IMU observation log queue is full"))

    def _write_rows(self) -> None:
        try:
            assert self._log_path is not None
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._log_path.open("x", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=OBSERVATION_FIELDS)
                writer.writeheader()
                handle.flush()
                last_flush = time.monotonic()
                while not self._stop.is_set() or not self._rows.empty():
                    try:
                        row = self._rows.get(timeout=0.1)
                    except queue.Empty:
                        row = None
                    if row is not None:
                        writer.writerow(row)
                    if time.monotonic() - last_flush >= 1.0:
                        handle.flush()
                        last_flush = time.monotonic()
        except Exception as exc:
            self._fail(exc)
