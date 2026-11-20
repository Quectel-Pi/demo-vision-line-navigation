"""Ultrasonic obstacle bypass state machine.

This module is deliberately hardware-free: it consumes filtered distance,
IMU, and vision status, and returns a motion request. TrackingRuntime owns the
sensor reads, wheel dispatch, stop calls, and status publication.

The controller implements the user-approved sequence:
stop -> reverse a short bounded distance -> left arc while seeking a line ->
optional short straight seek -> right arc while seeking a line -> optional
short straight seek. Any stable line in front may be adopted; the car does not
need to return to the original tape. Every distance-based leg is a time
estimate because the chassis has no wheel encoders.
Zero turn speed scale selects an in-place turn before the straight seek.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import median
from typing import Any, Iterable


INACTIVE = "INACTIVE"
SLOW = "SLOW"
STOP = "STOP"
ACTIVE = "ACTIVE"

PHASE_IDLE = "IDLE"
PHASE_STOP = "STOP"
PHASE_REVERSE = "REVERSE"
PHASE_LEFT_ARC = "LEFT_ARC"
PHASE_LEFT_SEEK = "LEFT_SEEK"
PHASE_RIGHT_ARC = "RIGHT_ARC"
PHASE_RIGHT_SEEK = "RIGHT_SEEK"
PHASE_SEEK_RECOVER = "SEEK_RECOVER"
PHASE_CAPTURE_HOLD = "CAPTURE_HOLD"
PHASE_FAILED = "FAILED"

_ACTIVE_PHASES = {
    PHASE_STOP,
    PHASE_REVERSE,
    PHASE_LEFT_ARC,
    PHASE_LEFT_SEEK,
    PHASE_RIGHT_ARC,
    PHASE_RIGHT_SEEK,
    PHASE_SEEK_RECOVER,
    PHASE_CAPTURE_HOLD,
}

_CORNER_ACTIVE_STATES = {"APPROACH", "ADVANCE", "SETTLE", "TURN", "SETTLE_TURN"}


@dataclass
class DistanceFilter:
    """Median of several raw reads in one control frame.

    A sudden real obstacle is accepted because the three reads agree with each
    other; a time-based jump filter is intentionally not used. A single bad
    read cannot form a median, and a wide spread is rejected as unstable.
    """

    min_valid_mm: float = 1.0
    max_valid_mm: float = 4000.0
    max_spread_mm: float = 50.0
    required_samples: int = 2

    def update(self, samples: Iterable[Any]) -> int | None:
        valid: list[float] = []
        for sample in samples:
            if sample is None or isinstance(sample, bool):
                continue
            try:
                value = float(sample)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(value):
                continue
            if value < self.min_valid_mm or value > self.max_valid_mm:
                continue
            valid.append(value)
        if len(valid) < max(1, int(self.required_samples)):
            return None
        if max(valid) - min(valid) > self.max_spread_mm:
            return None
        return int(round(median(valid)))


def _number(value: Any, name: str, lower: float, upper: float, *, allow_equal: bool = True) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if allow_equal:
        inside = lower <= number <= upper
    else:
        inside = lower < number < upper
    if not inside:
        raise ValueError(f"{name} must be between {lower} and {upper}")
    return number


def _integer(value: Any, name: str, lower: int, upper: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    number = int(value)
    if number < lower or number > upper:
        raise ValueError(f"{name} must be between {lower} and {upper}")
    return number


@dataclass(frozen=True)
class ObstacleBypassConfig:
    enabled: bool = False
    stop_distance_mm: float = 250.0
    hard_stop_distance_mm: float = 60.0
    pre_slow_distance_mm: float = 250.0
    pre_slow_scale: float = 0.4
    confirm_frames: int = 3
    min_valid_distance_mm: float = 1.0
    max_spread_mm: float = 50.0
    settle_s: float = 0.25
    reverse_time_s: float = 0.8
    reverse_scale: float = 0.25
    reverse_timeout_s: float = 1.2
    left_arc_deg: float = 90.0
    left_arc_duty: float = 6.0
    left_arc_speed_scale: float = 0.4
    left_stop_lead_deg: float = 8.0
    left_arc_settle_s: float = 0.2
    left_arc_timeout_s: float = 3.5
    adopt_line_after_deg: float = 70.0
    left_seek_distance_m: float = 0.06
    left_seek_timeout_s: float = 0.8
    right_arc_deg: float = 90.0
    right_arc_duty: float = 6.0
    turn_start_duty: float = 0.0
    turn_start_window_s: float = 0.0
    right_arc_speed_scale: float = 0.35
    right_stop_lead_deg: float = 5.0
    right_arc_settle_s: float = 0.2
    right_arc_timeout_s: float = 3.5
    right_seek_distance_m: float = 0.15
    right_seek_timeout_s: float = 2.5
    seek_blind_timeout_s: float = 0.4
    seek_recover_timeout_s: float = 0.8
    seek_recover_duty: float = 5.0
    seek_recover_deg: float = 30.0
    seek_speed_scale: float = 0.3
    seek_speed_mps: float = 0.08
    line_confirm_frames: int = 3
    visible_confirm_frames: int = 2
    line_center_tolerance_norm: float = 0.35
    line_heading_limit_deg: float = 45.0
    early_adopt_deg: float = 20.0
    early_adopt_center_norm: float = 0.05
    capture_hold_min_deg: float = 45.0
    capture_hold_s: float = 0.8
    pivot_max_rate_dps: float = 80.0
    max_control_gap_s: float = 0.6
    total_timeout_s: float = 12.0
    cooldown_s: float = 2.0
    handover_speed_scale: float = 0.5
    handover_slow_s: float = 0.5

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("obstacle.bypass.enabled must be boolean")
        stop = _number(self.stop_distance_mm, "obstacle.stop_distance_mm", 20.0, 2000.0)
        hard = _number(self.hard_stop_distance_mm, "obstacle.hard_stop_distance_mm", 10.0, 1999.0)
        pre_slow = _number(self.pre_slow_distance_mm, "obstacle.bypass.pre_slow_distance_mm", 20.0, 4000.0)
        if hard >= stop:
            raise ValueError("obstacle.hard_stop_distance_mm must be below stop_distance_mm")
        if pre_slow < stop:
            raise ValueError("obstacle.bypass.pre_slow_distance_mm must not be below stop_distance_mm")
        _number(self.pre_slow_scale, "obstacle.bypass.pre_slow_scale", 0.05, 1.0)
        _integer(self.confirm_frames, "obstacle.bypass.confirm_frames", 1, 20)
        _number(self.min_valid_distance_mm, "obstacle.bypass.min_valid_distance_mm", 0.0, 500.0)
        _number(self.max_spread_mm, "obstacle.bypass.max_spread_mm", 1.0, 1000.0)
        _number(self.settle_s, "obstacle.bypass.settle_s", 0.0, 3.0)
        reverse_time = _number(self.reverse_time_s, "obstacle.bypass.reverse_time_s", 0.0, 3.0)
        reverse_timeout = _number(self.reverse_timeout_s, "obstacle.bypass.reverse_timeout_s", 0.1, 5.0)
        if reverse_time > reverse_timeout:
            raise ValueError("obstacle.bypass.reverse_time_s must not exceed reverse_timeout_s")
        _number(self.reverse_scale, "obstacle.bypass.reverse_scale", 0.05, 1.0)
        left_arc = _number(self.left_arc_deg, "obstacle.bypass.left_arc_deg", 30.0, 135.0)
        right_arc = _number(self.right_arc_deg, "obstacle.bypass.right_arc_deg", 30.0, 135.0)
        _number(self.left_arc_duty, "obstacle.bypass.left_arc_duty", 0.0, 40.0)
        _number(self.right_arc_duty, "obstacle.bypass.right_arc_duty", 0.0, 40.0)
        _number(self.turn_start_duty, "obstacle.bypass.turn_start_duty", 0.0, 12.0)
        _number(self.turn_start_window_s, "obstacle.bypass.turn_start_window_s", 0.0, 0.3)
        _number(self.left_arc_speed_scale, "obstacle.bypass.left_arc_speed_scale", 0.0, 1.0)
        _number(self.right_arc_speed_scale, "obstacle.bypass.right_arc_speed_scale", 0.0, 1.0)
        left_lead = _number(self.left_stop_lead_deg, "obstacle.bypass.left_stop_lead_deg", 0.0, 30.0)
        right_lead = _number(self.right_stop_lead_deg, "obstacle.bypass.right_stop_lead_deg", 0.0, 30.0)
        if left_lead >= left_arc or right_lead >= right_arc:
            raise ValueError("obstacle.bypass stop lead must be below the arc angle")
        _number(self.left_arc_settle_s, "obstacle.bypass.left_arc_settle_s", 0.0, 1.5)
        _number(self.right_arc_settle_s, "obstacle.bypass.right_arc_settle_s", 0.0, 1.5)
        _number(self.left_arc_timeout_s, "obstacle.bypass.left_arc_timeout_s", 0.3, 10.0)
        _number(self.right_arc_timeout_s, "obstacle.bypass.right_arc_timeout_s", 0.3, 10.0)
        adopt_after = _number(self.adopt_line_after_deg, "obstacle.bypass.adopt_line_after_deg", 0.0, 135.0)
        if adopt_after > left_arc:
            raise ValueError("obstacle.bypass.adopt_line_after_deg must not exceed left_arc_deg")
        _number(self.left_seek_distance_m, "obstacle.bypass.left_seek_distance_m", 0.02, 0.8)
        _number(self.right_seek_distance_m, "obstacle.bypass.right_seek_distance_m", 0.02, 0.8)
        _number(self.left_seek_timeout_s, "obstacle.bypass.left_seek_timeout_s", 0.2, 10.0)
        _number(self.right_seek_timeout_s, "obstacle.bypass.right_seek_timeout_s", 0.2, 10.0)
        _number(self.seek_blind_timeout_s, "obstacle.bypass.seek_blind_timeout_s", 0.15, 2.0)
        _number(self.seek_recover_timeout_s, "obstacle.bypass.seek_recover_timeout_s", 0.2, 3.0)
        _number(self.seek_recover_duty, "obstacle.bypass.seek_recover_duty", 0.0, 20.0)
        _number(self.seek_recover_deg, "obstacle.bypass.seek_recover_deg", 10.0, 90.0)
        _number(self.seek_speed_scale, "obstacle.bypass.seek_speed_scale", 0.05, 1.0)
        _number(self.seek_speed_mps, "obstacle.bypass.seek_speed_mps", 0.02, 0.4)
        _integer(self.line_confirm_frames, "obstacle.bypass.line_confirm_frames", 1, 20)
        _integer(self.visible_confirm_frames, "obstacle.bypass.visible_confirm_frames", 1, 10)
        _number(self.line_center_tolerance_norm, "obstacle.bypass.line_center_tolerance_norm", 0.05, 0.49)
        _number(self.line_heading_limit_deg, "obstacle.bypass.line_heading_limit_deg", 5.0, 90.0)
        _number(self.early_adopt_deg, "obstacle.bypass.early_adopt_deg", 5.0, 60.0)
        _number(self.early_adopt_center_norm, "obstacle.bypass.early_adopt_center_norm", 0.01, 0.3)
        _number(self.capture_hold_min_deg, "obstacle.bypass.capture_hold_min_deg", 20.0, 80.0)
        _number(self.capture_hold_s, "obstacle.bypass.capture_hold_s", 0.2, 2.0)
        _number(self.pivot_max_rate_dps, "obstacle.bypass.pivot_max_rate_dps", 10.0, 150.0)
        _number(self.max_control_gap_s, "obstacle.bypass.max_control_gap_s", 0.2, 2.0)
        _number(self.total_timeout_s, "obstacle.bypass.total_timeout_s", 2.0, 60.0)
        _number(self.cooldown_s, "obstacle.bypass.cooldown_s", 0.0, 10.0)
        _number(self.handover_speed_scale, "obstacle.bypass.handover_speed_scale", 0.1, 1.0)
        _number(self.handover_slow_s, "obstacle.bypass.handover_slow_s", 0.0, 3.0)

    @classmethod
    def from_obstacle_config(cls, config: dict[str, Any]) -> "ObstacleBypassConfig":
        if not isinstance(config, dict):
            raise ValueError("obstacle configuration must be an object")
        bypass = config.get("bypass", {})
        if bypass is None:
            bypass = {}
        if not isinstance(bypass, dict):
            raise ValueError("obstacle.bypass must be an object")
        values = {
            "enabled": bypass.get("enabled", False),
            "stop_distance_mm": config.get("stop_distance_mm", 250.0),
            "hard_stop_distance_mm": config.get("hard_stop_distance_mm", 60.0),
            "pre_slow_distance_mm": bypass.get("pre_slow_distance_mm", 250.0),
            "pre_slow_scale": bypass.get("pre_slow_scale", 0.4),
            "confirm_frames": bypass.get("confirm_frames", 3),
            "min_valid_distance_mm": bypass.get("min_valid_distance_mm", 1.0),
            "max_spread_mm": bypass.get("max_spread_mm", 50.0),
            "settle_s": bypass.get("settle_s", 0.25),
            "reverse_time_s": bypass.get("reverse_time_s", 0.8),
            "reverse_scale": bypass.get("reverse_scale", 0.25),
            "reverse_timeout_s": bypass.get("reverse_timeout_s", 1.2),
            "left_arc_deg": bypass.get("left_arc_deg", 90.0),
            "left_arc_duty": bypass.get("left_arc_duty", 6.0),
            "left_arc_speed_scale": bypass.get("left_arc_speed_scale", 0.4),
            "left_stop_lead_deg": bypass.get("left_stop_lead_deg", 8.0),
            "left_arc_settle_s": bypass.get("left_arc_settle_s", 0.2),
            "left_arc_timeout_s": bypass.get("left_timeout_s", bypass.get("left_arc_timeout_s", 3.5)),
            "adopt_line_after_deg": bypass.get("adopt_line_after_deg", 70.0),
            "left_seek_distance_m": bypass.get("left_seek_distance_m", 0.20),
            "left_seek_timeout_s": bypass.get("left_seek_timeout_s", 2.5),
            "right_arc_deg": bypass.get("right_arc_deg", 90.0),
            "right_arc_duty": bypass.get("right_arc_duty", 6.0),
            "turn_start_duty": bypass.get("turn_start_duty", 0.0),
            "turn_start_window_s": bypass.get("turn_start_window_s", 0.0),
            "right_arc_speed_scale": bypass.get("right_arc_speed_scale", 0.35),
            "right_stop_lead_deg": bypass.get("right_stop_lead_deg", 5.0),
            "right_arc_settle_s": bypass.get("right_arc_settle_s", 0.2),
            "right_arc_timeout_s": bypass.get("right_timeout_s", bypass.get("right_arc_timeout_s", 3.5)),
            "right_seek_distance_m": bypass.get("right_seek_distance_m", 0.15),
            "right_seek_timeout_s": bypass.get("right_seek_timeout_s", 2.5),
            "seek_blind_timeout_s": bypass.get("seek_blind_timeout_s", 0.4),
            "seek_recover_timeout_s": bypass.get("seek_recover_timeout_s", 0.8),
            "seek_recover_duty": bypass.get("seek_recover_duty", 5.0),
            "seek_recover_deg": bypass.get("seek_recover_deg", 30.0),
            "seek_speed_scale": bypass.get("seek_speed_scale", 0.3),
            "seek_speed_mps": bypass.get("seek_speed_mps", 0.08),
            "line_confirm_frames": bypass.get("line_confirm_frames", 3),
            "visible_confirm_frames": bypass.get("visible_confirm_frames", 2),
            "line_center_tolerance_norm": bypass.get("line_center_tolerance_norm", 0.35),
            "line_heading_limit_deg": bypass.get("line_heading_limit_deg", 45.0),
            "early_adopt_deg": bypass.get("early_adopt_deg", 20.0),
            "early_adopt_center_norm": bypass.get("early_adopt_center_norm", 0.05),
            "capture_hold_min_deg": bypass.get("capture_hold_min_deg", 45.0),
            "capture_hold_s": bypass.get("capture_hold_s", 0.8),
            "pivot_max_rate_dps": bypass.get("pivot_max_rate_dps", 80.0),
            "max_control_gap_s": bypass.get("max_control_gap_s", 0.6),
            "total_timeout_s": bypass.get("total_timeout_s", 12.0),
            "cooldown_s": bypass.get("cooldown_s", 2.0),
            "handover_speed_scale": bypass.get("handover_speed_scale", 0.5),
            "handover_slow_s": bypass.get("handover_slow_s", 0.5),
        }
        try:
            return cls(**values)
        except TypeError as exc:
            raise ValueError(f"invalid obstacle.bypass configuration: {exc}") from exc


@dataclass(frozen=True)
class ObstacleBypassCommand:
    mode: str = INACTIVE
    phase: str = PHASE_IDLE
    reason: str = ""
    message: str = ""
    forward_scale: float = 1.0
    turn_duty: float = 0.0
    reverse_scale: float = 0.0
    speed_limit_scale: float = 1.0
    progress_deg: float | None = None
    attempts: int = 0
    line_candidate: bool = False
    handover: bool = False

    @property
    def active(self) -> bool:
        return self.mode == ACTIVE

    @property
    def stop(self) -> bool:
        return self.mode == STOP


class ObstacleBypassController:
    def __init__(self, config: ObstacleBypassConfig):
        self.config = config
        self.reset()

    def reset(self) -> None:
        self._phase = PHASE_IDLE
        self._reason = ""
        self._message = ""
        self._attempts = 0
        self._confirm_count = 0
        self._confirmed = False
        self._stop_started = 0.0
        self._phase_started = 0.0
        self._total_started = 0.0
        self._cooldown_until = 0.0
        self._handover_until = 0.0
        self._yaw_start = 0.0
        self._previous_turn_rate: float | None = None
        self._turn_start_complete = False
        self._arc_settle_until: float | None = None
        self._line_count = 0
        self._visible_count = 0
        self._line_candidate = False
        self._seek_last_evidence = 0.0
        self._recover_started = 0.0
        self._recover_direction = 0
        self._recover_yaw_start = 0.0
        self._capture_started = 0.0
        self._candidate_count = 0
        self._imu_recovery_count: int | None = None
        self._last_distance_mm: int | None = None

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def attempts(self) -> int:
        return self._attempts

    @property
    def active(self) -> bool:
        return self._phase in _ACTIVE_PHASES

    @property
    def failed(self) -> bool:
        return self._phase == PHASE_FAILED

    def _command(
        self,
        mode: str,
        *,
        phase: str | None = None,
        reason: str = "",
        message: str = "",
        forward_scale: float = 1.0,
        turn_duty: float = 0.0,
        reverse_scale: float = 0.0,
        speed_limit_scale: float = 1.0,
        progress_deg: float | None = None,
        handover: bool = False,
    ) -> ObstacleBypassCommand:
        return ObstacleBypassCommand(
            mode=mode,
            phase=self._phase if phase is None else phase,
            reason=reason,
            message=message,
            forward_scale=forward_scale,
            turn_duty=turn_duty,
            reverse_scale=reverse_scale,
            speed_limit_scale=speed_limit_scale,
            progress_deg=progress_deg,
            attempts=self._attempts,
            line_candidate=self._line_candidate,
            handover=handover,
        )

    def _inactive(self, now: float) -> ObstacleBypassCommand:
        scale = 1.0
        if now < self._handover_until:
            scale = self.config.handover_speed_scale
        return self._command(INACTIVE, speed_limit_scale=scale)

    def _slow(self) -> ObstacleBypassCommand:
        return self._command(SLOW, reason="obstacle_pre_slow",
                             speed_limit_scale=self.config.pre_slow_scale)

    def _stop(self, reason: str, message: str = "") -> ObstacleBypassCommand:
        self._reason = reason
        self._message = message or reason
        return self._command(STOP, reason=reason, message=self._message)

    def _fail(self, reason: str, message: str = "",
              progress_deg: float | None = None) -> ObstacleBypassCommand:
        self._phase = PHASE_FAILED
        self._reason = reason
        self._message = message or reason
        return self._command(STOP, reason=reason, message=self._message,
                             progress_deg=progress_deg)

    def _line_is_candidate(self, line_valid: bool, center_norm: float | None,
                           heading_deg: float | None) -> bool:
        if not line_valid or center_norm is None or heading_deg is None:
            return False
        if not math.isfinite(float(center_norm)) or not math.isfinite(float(heading_deg)):
            return False
        return (
            abs(float(center_norm)) <= self.config.line_center_tolerance_norm
            and abs(float(heading_deg)) <= self.config.line_heading_limit_deg
        )

    def _count_line(self, *, allowed: bool, line_valid: bool, center_norm: float | None,
                    heading_deg: float | None) -> bool:
        candidate = self._line_is_candidate(line_valid, center_norm, heading_deg)
        self._line_candidate = candidate
        if not allowed:
            return False
        if candidate:
            self._line_count += 1
        else:
            self._line_count = 0
        return candidate and self._line_count >= self.config.line_confirm_frames

    def _count_visible_line(self, line_valid: bool, line_near: bool, line_far: bool) -> tuple[bool, bool]:
        visible = bool(line_valid and (line_near or line_far))
        if visible:
            self._visible_count += 1
        else:
            self._visible_count = 0
        return visible, self._visible_count >= self.config.visible_confirm_frames

    def _handover(self, now: float) -> ObstacleBypassCommand:
        self._phase = PHASE_IDLE
        self._phase_started = 0.0
        self._total_started = 0.0
        self._stop_started = 0.0
        self._arc_settle_until = None
        self._confirmed = False
        self._reason = "line_acquired"
        self._message = "Stable line acquired; handing over to tracking"
        self._line_count = 0
        self._visible_count = 0
        self._line_candidate = True
        self._cooldown_until = now + self.config.cooldown_s
        self._handover_until = now + self.config.handover_slow_s
        return self._command(
            INACTIVE,
            phase=PHASE_IDLE,
            reason=self._reason,
            message=self._message,
            speed_limit_scale=self.config.handover_speed_scale,
            handover=True,
        )

    def _imu_ready(self, imu: dict[str, Any]) -> bool:
        if not isinstance(imu, dict) or not imu.get("imu_healthy"):
            return False
        for key in ("imu_delta_yaw_deg", "imu_yaw_rate_dps"):
            value = imu.get(key)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
                return False
        return True

    def _control_epoch_reason(self, dt: float, imu: dict[str, Any]) -> str:
        try:
            interval = float(dt)
        except (TypeError, ValueError):
            return "control_gap"
        if not math.isfinite(interval) or interval <= 0.0:
            return "control_gap"
        if interval > self.config.max_control_gap_s:
            return "control_gap"
        recovery = imu.get("imu_recovery_count", 0)
        if self._imu_recovery_count is None:
            self._imu_recovery_count = recovery
        elif recovery != self._imu_recovery_count:
            return "imu_epoch_changed"
        return ""

    def _remaining_total(self, now: float) -> bool:
        return self._total_started <= 0.0 or now - self._total_started <= self.config.total_timeout_s

    def _turn_duty(self, duty: float, angle: float, progress: float, rate: float, elapsed_s: float) -> float:
        duty *= max(0.45, min(1.0, (angle - progress) / 35.0))
        rate = max(0.0, rate)
        rise = 0.0 if self._previous_turn_rate is None else max(0.0, rate - self._previous_turn_rate)
        self._previous_turn_rate = rate
        projected_rate = rate + rise
        target_rate = self.config.pivot_max_rate_dps * 0.6
        if projected_rate > target_rate:
            duty *= target_rate / projected_rate
        self._turn_start_complete = (self._turn_start_complete
                                     or elapsed_s >= self.config.turn_start_window_s
                                     or rate >= 5.0 or progress >= 3.0)
        speed_scale = (self.config.left_arc_speed_scale if self._phase == PHASE_LEFT_ARC
                       else self.config.right_arc_speed_scale)
        if not self._turn_start_complete and speed_scale == 0:
            duty = max(duty, self.config.turn_start_duty)
        return duty

    def update(
        self,
        *,
        now: float,
        dt: float,
        distance_mm: int | None,
        obstacle_enabled: bool,
        armed: bool,
        corner_active: bool,
        tracking_ready: bool = True,
        imu: dict[str, Any],
        line_valid: bool,
        line_center_norm: float | None,
        line_heading_deg: float | None,
        line_sensor_channels: tuple[bool, ...] | None = None,
        line_near: bool = False,
        line_far: bool = False,
        camera_ok: bool = True,
        user_stop: bool = False,
    ) -> ObstacleBypassCommand:
        cfg = self.config
        if user_stop or not camera_ok or not armed:
            self.reset()
            if user_stop:
                return self._stop("user_stop", "User stop requested")
            if not camera_ok:
                return self._stop("camera_error", "Camera unavailable")
            return self._command(INACTIVE)

        if not obstacle_enabled:
            self.reset()
            return self._command(INACTIVE)

        if distance_mm is not None:
            self._last_distance_mm = int(distance_mm)

        if self._phase == PHASE_FAILED:
            return self._command(STOP, reason=self._reason, message=self._message)

        if distance_mm is not None and distance_mm <= cfg.hard_stop_distance_mm:
            if self.active:
                return self._fail("hard_stop", f"Obstacle at {distance_mm} mm; hard stop")
            return self._stop("hard_stop", f"Obstacle at {distance_mm} mm; hard stop")

        if (self._phase in {PHASE_LEFT_SEEK, PHASE_RIGHT_SEEK}
                and distance_mm is not None and distance_mm <= cfg.stop_distance_mm):
            return self._fail("search_blocked", f"Obstacle at {distance_mm} mm; forward search blocked")

        if self._phase == PHASE_IDLE:
            if now < self._cooldown_until:
                return self._inactive(now)
            if distance_mm is None:
                # No valid echo is not an obstacle. Vision remains the primary
                # control source; out-of-range or unstable auxiliary readings
                # must not stop line following.
                return self._inactive(now)
            if distance_mm <= cfg.stop_distance_mm:
                if not cfg.enabled:
                    return self._stop("obstacle_stop", f"Obstacle at {distance_mm} mm")
                if not tracking_ready:
                    return self._stop("not_tracking", f"Obstacle at {distance_mm} mm; tracking not active")
                self._phase = PHASE_STOP
                self._reason = "obstacle_confirm"
                self._message = f"Obstacle at {distance_mm} mm; confirming"
                self._stop_started = now
                self._confirm_count = 1
                self._confirmed = False
                self._line_count = 0
                self._line_candidate = False
                self._arc_settle_until = None
                return self._command(ACTIVE, reason=self._reason, message=self._message)
            if distance_mm <= cfg.pre_slow_distance_mm:
                return self._slow()
            return self._inactive(now)

        if not self._remaining_total(now):
            return self._fail("total_timeout", "Obstacle bypass total timeout")

        if self._confirmed:
            if not self._imu_ready(imu):
                return self._fail("imu_unavailable", "IMU unavailable during bypass")
            epoch_reason = self._control_epoch_reason(dt, imu)
            if epoch_reason:
                if epoch_reason == "control_gap":
                    return self._fail(
                        "control_gap",
                        f"Control interval {float(dt):.3f}s exceeds {cfg.max_control_gap_s:.3f}s",
                    )
                return self._fail("imu_epoch_changed", "IMU recovery epoch changed")
            if abs(float(imu["imu_yaw_rate_dps"])) > cfg.pivot_max_rate_dps:
                return self._fail("yaw_rate_limit", "Yaw rate exceeded during bypass")

        if self._phase == PHASE_STOP:
            if not self._confirmed:
                if distance_mm is None:
                    self._phase = PHASE_IDLE
                    self._reason = "obstacle_reading_lost"
                    self._message = "Obstacle reading lost before confirmation"
                    self._confirm_count = 0
                    return self._inactive(now)
                if distance_mm > cfg.stop_distance_mm:
                    self._phase = PHASE_IDLE
                    self._reason = "obstacle_cleared_before_confirm"
                    self._message = "Obstacle reading cleared before confirmation"
                    self._confirm_count = 0
                    return self._inactive(now)
                self._confirm_count += 1
                if self._confirm_count >= cfg.confirm_frames:
                    self._confirmed = True
                    self._stop_started = now
                    self._total_started = now
                    self._attempts += 1
                    self._imu_recovery_count = imu.get("imu_recovery_count", 0)
                    self._reason = "obstacle_confirmed"
                    self._message = "Obstacle confirmed; preparing bypass"
                return self._command(ACTIVE, reason=self._reason, message=self._message)
            if now - self._stop_started < cfg.settle_s:
                return self._command(ACTIVE, reason=self._reason, message=self._message)
            if cfg.reverse_time_s == 0:
                return self._enter_left_arc(now, imu)
            self._phase = PHASE_REVERSE
            self._phase_started = now
            self._reason = "reverse_clearance"
            self._message = "Reversing to clear the turning envelope"
            return self._command(ACTIVE, forward_scale=0.0,
                                 reason=self._reason, message=self._message)

        if self._phase == PHASE_REVERSE:
            duration = min(cfg.reverse_time_s, cfg.reverse_timeout_s)
            if now - self._phase_started < duration:
                return self._command(ACTIVE, forward_scale=0.0,
                                     reverse_scale=cfg.reverse_scale,
                                     reason=self._reason, message=self._message)
            return self._enter_left_arc(now, imu)

        if self._phase == PHASE_LEFT_ARC:
            progress = float(imu["imu_delta_yaw_deg"]) - self._yaw_start
            if progress < -8.0:
                return self._fail("turn_direction", "Left arc turned the wrong way", progress)
            if progress > cfg.left_arc_deg + 15.0:
                return self._fail("left_arc_overshoot", "Left arc overshoot", progress)
            if now - self._phase_started > cfg.left_arc_timeout_s:
                return self._fail("left_arc_timeout", "Left arc timed out", progress)
            channels = line_sensor_channels or ()
            left_evidence = (
                (len(channels) >= 4 and bool(channels[0] or channels[1]))
                or (line_center_norm is not None and line_center_norm <= -cfg.early_adopt_center_norm)
            )
            early_allowed = progress >= cfg.early_adopt_deg and left_evidence
            if self._count_line(
                allowed=early_allowed or (
                    cfg.left_arc_speed_scale > 0 and progress >= cfg.adopt_line_after_deg),
                line_valid=line_valid, center_norm=line_center_norm,
                heading_deg=line_heading_deg,
            ):
                return self._handover(now)
            if early_allowed:
                _, visible_confirmed = self._count_visible_line(line_valid, line_near, line_far)
                if visible_confirmed:
                    return self._handover(now)
            else:
                self._visible_count = 0
            if line_near or line_far:
                self._candidate_count += 1
            else:
                self._candidate_count = 0
            if progress >= cfg.capture_hold_min_deg and self._candidate_count >= cfg.visible_confirm_frames:
                return self._enter_capture_hold(now, progress)
            target = cfg.left_arc_deg - cfg.left_stop_lead_deg
            if progress >= target:
                if self._arc_settle_until is None:
                    self._arc_settle_until = now + cfg.left_arc_settle_s
                    return self._command(ACTIVE, forward_scale=0.0,
                                         reason="left_arc_settle", progress_deg=progress)
                if now < self._arc_settle_until:
                    return self._command(ACTIVE, forward_scale=0.0,
                                         reason="left_arc_settle", progress_deg=progress)
                return self._enter_seek(PHASE_LEFT_SEEK, now, distance_mm, progress)
            return self._command(ACTIVE, forward_scale=cfg.left_arc_speed_scale,
                                 turn_duty=self._turn_duty(cfg.left_arc_duty, cfg.left_arc_deg,
                                     progress, float(imu["imu_yaw_rate_dps"]), now-self._phase_started), progress_deg=progress,
                                 reason=self._reason, message=self._message)

        if self._phase == PHASE_LEFT_SEEK:
            if self._count_line(allowed=True, line_valid=line_valid,
                                center_norm=line_center_norm, heading_deg=line_heading_deg):
                return self._handover(now)
            channels = line_sensor_channels or ()
            middle_black = len(channels) >= 4 and bool(channels[1] or channels[2])
            if middle_black and line_valid and (line_near or line_far):
                return self._handover(now)
            visible, visible_confirmed = self._count_visible_line(line_valid, line_near, line_far)
            if visible_confirmed:
                return self._handover(now)
            if visible:
                self._seek_last_evidence = now
                return self._command(ACTIVE, forward_scale=0.0,
                                     reason="left_seek_visible_wait",
                                     message="Line visible; stopping to confirm handover")
            sensor_black = any(channels)
            if line_valid or sensor_black:
                self._seek_last_evidence = now
            if sensor_black:
                if line_valid:
                    return self._command(ACTIVE, forward_scale=0.0,
                                         reason="left_seek_waiting_line",
                                         message="Line sensor on tape; waiting for visual confirmation")
                return self._enter_seek_recover(now, imu, channels)
            if now - self._seek_last_evidence >= cfg.seek_blind_timeout_s:
                return self._enter_right_arc(now, imu)
            duration = min(cfg.left_seek_timeout_s,
                           cfg.left_seek_distance_m / max(0.02, cfg.seek_speed_mps))
            if now - self._phase_started >= duration:
                return self._enter_right_arc(now, imu)
            return self._command(ACTIVE, forward_scale=cfg.seek_speed_scale,
                                 reason=self._reason, message=self._message)

        if self._phase == PHASE_RIGHT_ARC:
            progress = -(float(imu["imu_delta_yaw_deg"]) - self._yaw_start)
            if progress < -8.0:
                return self._fail("turn_direction", "Right arc turned the wrong way", progress)
            if progress > cfg.right_arc_deg + 15.0:
                return self._fail("right_arc_overshoot", "Right arc overshoot", progress)
            if now - self._phase_started > cfg.right_arc_timeout_s:
                return self._fail("right_arc_timeout", "Right arc timed out", progress)
            channels = line_sensor_channels or ()
            right_evidence = (
                (len(channels) >= 4 and bool(channels[2] or channels[3]))
                or (line_center_norm is not None and line_center_norm >= cfg.early_adopt_center_norm)
            )
            early_allowed = progress >= cfg.early_adopt_deg and right_evidence
            if self._count_line(allowed=early_allowed or cfg.right_arc_speed_scale > 0,
                                line_valid=line_valid,
                                center_norm=line_center_norm, heading_deg=line_heading_deg):
                return self._handover(now)
            if early_allowed:
                _, visible_confirmed = self._count_visible_line(line_valid, line_near, line_far)
                if visible_confirmed:
                    return self._handover(now)
            else:
                self._visible_count = 0
            if line_near or line_far:
                self._candidate_count += 1
            else:
                self._candidate_count = 0
            if progress >= cfg.capture_hold_min_deg and self._candidate_count >= cfg.visible_confirm_frames:
                return self._enter_capture_hold(now, progress)
            target = cfg.right_arc_deg - cfg.right_stop_lead_deg
            if progress >= target:
                if self._arc_settle_until is None:
                    self._arc_settle_until = now + cfg.right_arc_settle_s
                    return self._command(ACTIVE, forward_scale=0.0,
                                         reason="right_arc_settle", progress_deg=progress)
                if now < self._arc_settle_until:
                    return self._command(ACTIVE, forward_scale=0.0,
                                         reason="right_arc_settle", progress_deg=progress)
                return self._enter_seek(PHASE_RIGHT_SEEK, now, distance_mm, progress)
            return self._command(ACTIVE, forward_scale=cfg.right_arc_speed_scale,
                                 turn_duty=-self._turn_duty(cfg.right_arc_duty, cfg.right_arc_deg,
                                     progress, -float(imu["imu_yaw_rate_dps"]), now-self._phase_started), progress_deg=progress,
                                 reason=self._reason, message=self._message)

        if self._phase == PHASE_RIGHT_SEEK:
            if self._count_line(allowed=True, line_valid=line_valid,
                                center_norm=line_center_norm, heading_deg=line_heading_deg):
                return self._handover(now)
            channels = line_sensor_channels or ()
            middle_black = len(channels) >= 4 and bool(channels[1] or channels[2])
            if middle_black and line_valid and (line_near or line_far):
                return self._handover(now)
            visible, visible_confirmed = self._count_visible_line(line_valid, line_near, line_far)
            if visible_confirmed:
                return self._handover(now)
            if visible:
                self._seek_last_evidence = now
                return self._command(ACTIVE, forward_scale=0.0,
                                     reason="right_seek_visible_wait",
                                     message="Line visible; stopping to confirm handover")
            sensor_black = any(channels)
            if line_valid or sensor_black:
                self._seek_last_evidence = now
            if sensor_black:
                if line_valid:
                    return self._command(ACTIVE, forward_scale=0.0,
                                         reason="right_seek_waiting_line",
                                         message="Line sensor on tape; waiting for visual confirmation")
                return self._enter_seek_recover(now, imu, channels)
            if now - self._seek_last_evidence >= cfg.seek_blind_timeout_s:
                return self._fail("seek_no_line", "No near line or sensor evidence during seek")
            duration = min(cfg.right_seek_timeout_s,
                           cfg.right_seek_distance_m / max(0.02, cfg.seek_speed_mps))
            if now - self._phase_started >= duration:
                return self._fail("no_line_after_search", "No stable line found after bypass search")
            return self._command(ACTIVE, forward_scale=cfg.seek_speed_scale,
                                 reason=self._reason, message=self._message)

        if self._phase == PHASE_CAPTURE_HOLD:
            if self._count_line(allowed=True, line_valid=line_valid,
                                center_norm=line_center_norm, heading_deg=line_heading_deg):
                return self._handover(now)
            _, visible_confirmed = self._count_visible_line(line_valid, line_near, line_far)
            if visible_confirmed:
                return self._handover(now)
            if now - self._capture_started >= cfg.capture_hold_s:
                return self._fail("capture_unconfirmed", "Spatial candidate did not confirm after braking")
            return self._command(ACTIVE, forward_scale=0.0,
                                 reason="capture_hold",
                                 message="Spatial line candidate; braking to confirm")

        if self._phase == PHASE_SEEK_RECOVER:
            if self._count_line(allowed=True, line_valid=line_valid,
                                center_norm=line_center_norm, heading_deg=line_heading_deg):
                return self._handover(now)
            channels = line_sensor_channels or ()
            visible, visible_confirmed = self._count_visible_line(line_valid, line_near, line_far)
            if visible_confirmed:
                return self._handover(now)
            if visible:
                middle_black = len(channels) >= 4 and bool(channels[1] or channels[2])
                if middle_black:
                    return self._handover(now)
                return self._command(ACTIVE, forward_scale=0.0,
                                     reason="seek_recover_waiting_line",
                                     message="Line visible; stopping the recovery turn")
            elapsed = now - self._recover_started
            if self._recover_direction == 0:
                if elapsed >= cfg.seek_recover_timeout_s:
                    return self._fail("seek_line_not_found", "No line found after sensor stop")
                return self._command(ACTIVE, forward_scale=0.0,
                                     reason="seek_recover_wait",
                                     message="Line sensor on tape; waiting for visual confirmation")
            progress = self._recover_direction * (
                float(imu["imu_delta_yaw_deg"]) - self._recover_yaw_start)
            if progress < -8.0:
                return self._fail("turn_direction", "Seek recovery turned the wrong way", progress)
            if progress >= cfg.seek_recover_deg or elapsed >= cfg.seek_recover_timeout_s:
                return self._fail("seek_line_not_found", "No line found after seek recovery", progress)
            return self._command(ACTIVE, forward_scale=0.0,
                                 turn_duty=self._recover_direction * cfg.seek_recover_duty,
                                 progress_deg=progress, reason="seek_recover",
                                 message="Line sensor detected tape; turning toward it")

        self.reset()
        return self._inactive(now)

    def _enter_capture_hold(self, now: float, progress: float) -> ObstacleBypassCommand:
        self._phase = PHASE_CAPTURE_HOLD
        self._capture_started = now
        self._line_count = 0
        self._visible_count = 0
        self._reason = "capture_hold"
        self._message = "Spatial line candidate; braking to confirm"
        return self._command(ACTIVE, forward_scale=0.0, progress_deg=progress,
                             reason=self._reason, message=self._message)

    def _enter_seek(self, phase: str, now: float, distance_mm: int | None,
                    progress: float) -> ObstacleBypassCommand:
        if distance_mm is not None and distance_mm <= self.config.stop_distance_mm:
            return self._fail("search_blocked", f"Obstacle at {distance_mm} mm; forward search blocked", progress)
        self._phase = phase
        self._phase_started = now
        self._seek_last_evidence = now
        self._arc_settle_until = None
        self._line_count = 0
        self._visible_count = 0
        self._reason = phase.lower()
        self._message = "Turn complete; seeking a line ahead"
        return self._command(ACTIVE, forward_scale=self.config.seek_speed_scale,
                             progress_deg=progress, reason=self._reason, message=self._message)

    @staticmethod
    def _sensor_direction(channels: tuple[bool, ...]) -> int:
        if len(channels) < 4:
            return 0
        left = bool(channels[0] or channels[1])
        right = bool(channels[2] or channels[3])
        if left == right:
            return 0
        return 1 if left else -1

    def _enter_seek_recover(self, now: float, imu: dict[str, Any],
                            channels: tuple[bool, ...]) -> ObstacleBypassCommand:
        if not self._imu_ready(imu):
            return self._fail("imu_unavailable", "IMU unavailable before seek recovery")
        direction = self._sensor_direction(channels)
        self._phase = PHASE_SEEK_RECOVER
        self._recover_started = now
        self._recover_direction = direction
        self._recover_yaw_start = float(imu["imu_delta_yaw_deg"])
        self._line_count = 0
        self._visible_count = 0
        self._reason = "seek_recover"
        self._message = ("Line sensor detected tape; turning toward it"
                         if direction else "Line sensor detected tape; waiting for visual confirmation")
        return self._command(ACTIVE, forward_scale=0.0,
                             turn_duty=direction * self.config.seek_recover_duty,
                             reason=self._reason, message=self._message)

    def _enter_left_arc(self, now: float, imu: dict[str, Any]) -> ObstacleBypassCommand:
        if not self._imu_ready(imu):
            return self._fail("imu_unavailable", "IMU angle unavailable before left arc")
        self._phase = PHASE_LEFT_ARC
        self._phase_started = now
        self._yaw_start = float(imu["imu_delta_yaw_deg"])
        self._previous_turn_rate = None
        self._turn_start_complete = False
        self._imu_recovery_count = imu.get("imu_recovery_count", 0)
        self._arc_settle_until = None
        self._line_count = 0
        self._visible_count = 0
        self._candidate_count = 0
        self._reason = "left_arc"
        self._message = ("Turning left in place before seeking a line"
                         if self.config.left_arc_speed_scale == 0 else "Turning left while seeking a line")
        return self._command(ACTIVE, forward_scale=self.config.left_arc_speed_scale,
                             turn_duty=(max(self.config.left_arc_duty, self.config.turn_start_duty)
                                        if self.config.left_arc_speed_scale == 0 and self.config.turn_start_window_s > 0
                                        else self.config.left_arc_duty), progress_deg=0.0,
                             reason=self._reason, message=self._message)

    def _enter_right_arc(self, now: float, imu: dict[str, Any]) -> ObstacleBypassCommand:
        if not self._imu_ready(imu):
            return self._fail("imu_unavailable", "IMU angle unavailable before right arc")
        self._phase = PHASE_RIGHT_ARC
        self._phase_started = now
        self._yaw_start = float(imu["imu_delta_yaw_deg"])
        self._previous_turn_rate = None
        self._turn_start_complete = False
        self._imu_recovery_count = imu.get("imu_recovery_count", 0)
        self._arc_settle_until = None
        self._line_count = 0
        self._visible_count = 0
        self._candidate_count = 0
        self._reason = "right_arc"
        self._message = ("Turning right in place before seeking a line"
                         if self.config.right_arc_speed_scale == 0 else "Turning right while seeking a line")
        return self._command(ACTIVE, forward_scale=self.config.right_arc_speed_scale,
                             turn_duty=-(max(self.config.right_arc_duty, self.config.turn_start_duty)
                                         if self.config.right_arc_speed_scale == 0 and self.config.turn_start_window_s > 0
                                         else self.config.right_arc_duty), progress_deg=0.0,
                             reason=self._reason, message=self._message)
