"""Bounded, timestamped line-sensor reads independent of the camera loop."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import math
import threading
import time


@dataclass(frozen=True)
class LineSamplingConfig:
    frequency_hz: float = 100.
    buffer_size: int = 512
    max_read_duration_s: float = .01
    stale_after_s: float = .10

    def __post_init__(self):
        for name in ("frequency_hz", "max_read_duration_s", "stale_after_s"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"invalid line sampling {name}")
        if not 1 <= self.frequency_hz <= 200:
            raise ValueError("line sampling frequency must be within 1..200 Hz")
        if isinstance(self.buffer_size, bool) or not isinstance(self.buffer_size, int) or not 2 <= self.buffer_size <= 10000:
            raise ValueError("invalid line sampling buffer size")


@dataclass(frozen=True)
class LineSensorSample:
    sequence: int
    started_s: float
    completed_s: float
    raw_channels: tuple[bool, bool, bool, bool] | None
    error: str
    imu: dict
    imu_error: str
    context: tuple
    active: bool

    @property
    def channels(self):
        return self.raw_channels if not self.error else None


class LineSensorSampler:
    def __init__(self, sensor, config: LineSamplingConfig, *, imu_snapshot=None,
                 context_snapshot=None, on_sample=None, clock=time.monotonic):
        self.sensor, self.config = sensor, config
        self._imu_snapshot, self._context_snapshot = imu_snapshot, context_snapshot
        self._on_sample = on_sample
        self._clock = clock
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._buffer = deque()
        self._latest = None
        self._sequence = self._dropped = self._reported_dropped = self._failed = 0

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            if self._stop.is_set():
                raise RuntimeError("Line sampler is still stopping")
            return
        with self._lock:
            self._buffer.clear()
            self._latest = None
            self._reported_dropped = self._dropped
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="line-sensor-sampling", daemon=True)
        self._thread.start()

    def stop(self, timeout=1.):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
        return self._thread is None or not self._thread.is_alive()

    def _run(self):
        period = 1/self.config.frequency_hz
        deadline = self._clock()
        while not self._stop.is_set():
            self._poll_once()
            deadline += period
            now = self._clock()
            if deadline <= now:
                deadline = now+period
            self._stop.wait(max(0., deadline-now))

    def _poll_once(self):
        before = self._context_snapshot() if self._context_snapshot is not None else ((), False)
        started = self._clock()
        error, raw = "", None
        try:
            raw = self.sensor.read_channels()
        except Exception:
            error = "sensor_read_failed"
        completed = self._clock()
        if raw is None or len(raw) != 4 or any(type(v) is not bool for v in raw):
            raw, error = None, error or "sensor_sample_unavailable"
        else:
            raw = tuple(raw)
        if completed < started or completed-started > self.config.max_read_duration_s:
            error = "sensor_read_duration_invalid"
        imu, imu_error = {}, ""
        if self._imu_snapshot is not None:
            try:
                values = self._imu_snapshot()
                imu = {k: values.get(k) for k in ("imu_healthy", "imu_delta_yaw_deg", "imu_yaw_rate_dps",
                    "imu_timestamp_monotonic_s", "imu_age_s", "imu_recovery_count")}
            except Exception:
                imu_error = "imu_snapshot_failed"
        after = self._context_snapshot() if self._context_snapshot is not None else ((), False)
        if before != after:
            error = "context_changed_during_read"
        with self._lock:
            self._sequence += 1
            sample = LineSensorSample(self._sequence, started, completed, raw, error,
                                      imu, imu_error, before[0], before[1])
            if len(self._buffer) >= self.config.buffer_size:
                self._buffer.popleft()
                self._dropped += 1
            self._buffer.append(sample)
            self._latest = sample
            self._failed += bool(error)
        if self._on_sample is not None:
            self._on_sample(sample)
        return sample

    def drain_before(self, cutoff_s):
        with self._lock:
            samples = []
            while self._buffer and self._buffer[0].completed_s <= cutoff_s:
                samples.append(self._buffer.popleft())
            dropped = self._dropped-self._reported_dropped
            self._reported_dropped = self._dropped
        return tuple(samples), dropped

    def snapshot(self):
        with self._lock:
            latest = self._latest
            now = self._clock()
            age = None if latest is None else now-latest.completed_s
            alive = self._thread is not None and self._thread.is_alive()
            state = "STOPPING" if alive and self._stop.is_set() else "RUNNING" if alive else "STOPPED"
            return dict(state=state, requested_hz=self.config.frequency_hz, samples=self._sequence,
                failed_samples=self._failed, dropped_samples=self._dropped, queued_samples=len(self._buffer),
                latest_age_s=age,
                latest_channels=(latest.channels if latest is not None and age is not None
                                 and 0 <= age and now-latest.started_s <= self.config.stale_after_s else None),
                latest_observation_age_max_s=None if latest is None else now-latest.started_s,
                latest_sample=None if latest is None else asdict(latest))
