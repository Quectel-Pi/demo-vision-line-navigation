"""Yaw feedback calculations for the explicit feedback runtime mode.

Turn input/output are normalized, left positive. Gyro rates use degrees/second.
The response table must describe forward driving at the recorded duty range.
This module opens no hardware. A future runtime caller MUST stop/disarm when
YawFeedbackError is raised or stop_required is true; never reuse an old output.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any


class YawFeedbackError(ValueError):
    """Input/configuration is unsafe for feedback; caller must stop/disarm."""


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise YawFeedbackError(f"{name} must be a finite number")
    return float(value)


def _clip(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def forward_motor_profile_key(motor: dict[str, Any], turn_duty_max: float) -> str:
    """Fingerprint actual driver/mapping/deadzone settings and turn conversion."""
    encoded = json.dumps({"motor": motor, "turn_duty_max": _number(turn_duty_max, "turn_duty_max")},
                         sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class YawFeedbackConfig:
    enabled: bool = False
    axis_sign_verified: bool = False
    response_verified: bool = False
    response_scope: str = "unverified"
    calibration_record: str = ""
    motor_profile_key: str = ""
    forward_duty_min: float | None = None
    forward_duty_max: float | None = None
    turn_duty_max: float | None = None
    response_points: tuple[tuple[float, float], ...] = ()
    yaw_rate_kp: float = 0.25
    yaw_rate_ki: float = 0.0
    yaw_rate_kd: float = 0.0
    integral_limit: float = 0.5
    derivative_filter_alpha: float = 0.3
    imu_weight: float = 0.1
    max_correction: float = 0.15
    max_visual_turn: float = 1.0
    max_turn: float = 1.0
    max_target_yaw_rate_dps: float = 80.0
    max_actual_yaw_rate_dps: float = 80.0
    yaw_error_deadband_dps: float = 0.5
    max_dt_s: float = 0.2
    stale_after_s: float = 0.2
    turn_smoothing_alpha: float = 0.35
    turn_slew_per_s: float = 1.5
    uncertain_max_turn: float = 0.2
    uncertain_speed_limit: float = 0.5

    def __post_init__(self) -> None:
        for name in ("enabled", "axis_sign_verified", "response_verified"):
            if not isinstance(getattr(self, name), bool):
                raise YawFeedbackError(f"{name} must be boolean")
        limits = {
            "yaw_rate_kp": (0, 10), "yaw_rate_ki": (0, 10), "yaw_rate_kd": (0, 10),
            "integral_limit": (0, 10), "derivative_filter_alpha": (0.000001, 1),
            "imu_weight": (0, 0.2), "max_correction": (0.000001, 1),
            "max_visual_turn": (0.000001, 1), "max_turn": (0.000001, 1),
            "max_target_yaw_rate_dps": (0.000001, 220),
            "max_actual_yaw_rate_dps": (0.000001, 220),
            "yaw_error_deadband_dps": (0, 10), "max_dt_s": (0.000001, 0.2),
            "stale_after_s": (0.000001, 0.2), "turn_smoothing_alpha": (0.000001, 1),
            "turn_slew_per_s": (0.000001, 20), "uncertain_max_turn": (0, 1),
            "uncertain_speed_limit": (0, 1),
        }
        for name, (lower, upper) in limits.items():
            if not lower <= _number(getattr(self, name), name) <= upper:
                raise YawFeedbackError(f"{name} must be within [{lower}, {upper}]")
        if self.max_target_yaw_rate_dps > self.max_actual_yaw_rate_dps:
            raise YawFeedbackError("target rate limit exceeds actual-rate safety limit")
        if self.uncertain_max_turn > self.max_turn:
            raise YawFeedbackError("uncertain turn limit exceeds max_turn")
        if not self.enabled:
            return
        if not self.axis_sign_verified or not self.response_verified:
            raise YawFeedbackError("verify yaw axis/sign and forward response before enabling feedback")
        if self.response_scope != "forward_at_recorded_speed":
            raise YawFeedbackError("forward feedback cannot use unverified or stationary response")
        if not isinstance(self.calibration_record, str) or not self.calibration_record.strip():
            raise YawFeedbackError("calibration_record is required")
        if not isinstance(self.motor_profile_key, str) or not self.motor_profile_key.strip():
            raise YawFeedbackError("motor_profile_key is required")
        low = _number(self.forward_duty_min, "forward_duty_min")
        high = _number(self.forward_duty_max, "forward_duty_max")
        turn = _number(self.turn_duty_max, "turn_duty_max")
        if not 0 < low <= high <= 100 or not 0 < turn <= 100:
            raise YawFeedbackError("invalid calibrated duty range")
        points = self.response_points
        if len(points) < 3:
            raise YawFeedbackError("response_points need negative, zero and positive turn measurements")
        previous_input, previous_rate = -math.inf, -math.inf
        zero_seen = False
        for point in points:
            if not isinstance(point, (tuple, list)) or len(point) != 2:
                raise YawFeedbackError("each response point must contain turn input and rate")
            command, rate = _number(point[0], "response input"), _number(point[1], "response rate")
            if not -1 <= command <= 1 or command <= previous_input or rate < previous_rate:
                raise YawFeedbackError("response table must have increasing inputs and nondecreasing rates")
            if abs(rate) > self.max_actual_yaw_rate_dps or command * rate < 0:
                raise YawFeedbackError("response table has inconsistent sign or excessive rate")
            if command == 0:
                if rate != 0:
                    raise YawFeedbackError("zero turn reference must have zero expected yaw rate")
                zero_seen = True
            previous_input, previous_rate = command, rate
        if not zero_seen or points[0][0] >= 0 or points[-1][0] <= 0:
            raise YawFeedbackError("response table must cover both directions and zero")
        if points[0][1] >= 0 or points[-1][1] <= 0:
            raise YawFeedbackError("response table has no demonstrated rotation in both directions")


def yaw_feedback_config_from_dict(values: dict[str, Any]) -> YawFeedbackConfig:
    """Read the settings sidecar; runtime mode must also explicitly be feedback."""
    if not isinstance(values, dict):
        raise YawFeedbackError("yaw feedback settings must be an object")
    unknown = set(values) - set(YawFeedbackConfig.__dataclass_fields__)
    if unknown:
        raise YawFeedbackError(f"unknown yaw feedback settings: {', '.join(sorted(unknown))}")
    parsed = dict(values)
    points = parsed.get("response_points", ())
    if not isinstance(points, (list, tuple)):
        raise YawFeedbackError("response_points must be an array")
    if not all(isinstance(point, (list, tuple)) and len(point) == 2 for point in points):
        raise YawFeedbackError("response_points must contain input/rate pairs")
    parsed["response_points"] = tuple(tuple(point) for point in points)
    return YawFeedbackConfig(**parsed)


@dataclass(frozen=True)
class YawFeedbackResult:
    visual_turn: float
    final_turn: float
    feedback_active: bool = False
    stop_required: bool = False
    speed_limit: float = 1.0
    target_yaw_rate_dps: float | None = None
    yaw_rate_error_dps: float | None = None
    correction: float | None = None
    output_limited: bool = False


class YawRateFeedback:
    """Bounded normalized correction; errors latch until explicit reset."""

    def __init__(self, config: YawFeedbackConfig) -> None:
        self.config = config
        self.reset()

    def reset(self) -> None:
        """Caller may reset only after stopping and checking conditions again."""
        self._integral = 0.0
        self._previous_error: float | None = None
        self._derivative = 0.0
        self._previous_turn = 0.0
        self._fault: str | None = None

    def _expected_rate(self, turn: float) -> float:
        points = self.config.response_points
        if not points[0][0] <= turn <= points[-1][0]:
            raise YawFeedbackError("turn is outside the calibrated response table; extrapolation refused")
        for (x0, y0), (x1, y1) in zip(points, points[1:]):
            if x0 <= turn <= x1:
                return y0 + (y1 - y0) * (turn - x0) / (x1 - x0)
        raise YawFeedbackError("no calibrated response interval")

    def update(self, visual_turn: float, actual_yaw_rate_dps: float | None, *,
               dt_s: float, imu_age_s: float | None, imu_healthy: bool, path_state: str,
               forward_duty: float, turn_duty_max: float, motor_profile_key: str) -> YawFeedbackResult:
        try:
            return self._update(visual_turn, actual_yaw_rate_dps, dt_s=dt_s, imu_age_s=imu_age_s,
                                imu_healthy=imu_healthy, path_state=path_state, forward_duty=forward_duty,
                                turn_duty_max=turn_duty_max, motor_profile_key=motor_profile_key)
        except YawFeedbackError as exc:
            self.reset()
            self._fault = str(exc)
            raise

    def _update(self, visual_turn: float, actual_yaw_rate_dps: float | None, *,
                dt_s: float, imu_age_s: float | None, imu_healthy: bool, path_state: str,
                forward_duty: float, turn_duty_max: float, motor_profile_key: str) -> YawFeedbackResult:
        cfg = self.config
        if self._fault is not None:
            raise YawFeedbackError(f"feedback fault latched; stop and reset: {self._fault}")
        if cfg.enabled and path_state == "LOST":
            self.reset()
            return YawFeedbackResult(0.0, 0.0, stop_required=True, speed_limit=0.0)
        visual = _number(visual_turn, "visual_turn")
        if not cfg.enabled:
            # Baseline passthrough: no IMU dependency, smoothing or new limits.
            return YawFeedbackResult(visual, visual)
        if path_state not in {"VALID", "UNCERTAIN"}:
            raise YawFeedbackError("unknown path state")
        if imu_healthy is not True:
            raise YawFeedbackError("IMU is not healthy")
        actual = _number(actual_yaw_rate_dps, "actual_yaw_rate_dps")
        elapsed, age = _number(dt_s, "dt_s"), _number(imu_age_s, "imu_age_s")
        if not 0 < elapsed <= cfg.max_dt_s or not 0 <= age <= cfg.stale_after_s:
            raise YawFeedbackError("abnormal control interval or stale IMU")
        if abs(actual) > cfg.max_actual_yaw_rate_dps:
            raise YawFeedbackError("actual yaw rate exceeds safety limit")
        forward, turn_limit = _number(forward_duty, "forward_duty"), _number(turn_duty_max, "turn_duty_max")
        if not cfg.forward_duty_min <= forward <= cfg.forward_duty_max:
            raise YawFeedbackError("forward duty is outside calibrated conditions")
        if not math.isclose(turn_limit, cfg.turn_duty_max, rel_tol=0, abs_tol=1e-9):
            raise YawFeedbackError("turn duty limit changed since calibration")
        if motor_profile_key != cfg.motor_profile_key:
            raise YawFeedbackError("motor mapping/deadzone/profile changed since calibration")
        max_turn = min(cfg.max_turn, forward / turn_limit)
        visual = _clip(visual, min(cfg.max_visual_turn, max_turn))
        if path_state == "UNCERTAIN":
            self.reset()
            final = _clip(visual, min(cfg.uncertain_max_turn, max_turn))
            return YawFeedbackResult(visual, final, speed_limit=cfg.uncertain_speed_limit,
                                     output_limited=final != visual)
        # A small-turn endpoint is not a target for a sharper bend. Outside
        # the measured turn range, preserve visual steering without a rate
        # correction; all IMU health/rate checks above and slew limits below
        # still apply. Clear rate PID memory before reentering its valid range.
        feedback_active = cfg.response_points[0][0] <= visual <= cfg.response_points[-1][0]
        target = None
        rate_error = None
        error = None
        integral = 0.0
        derivative = 0.0
        raw_correction = 0.0
        if feedback_active:
            target = _clip(self._expected_rate(visual), cfg.max_target_yaw_rate_dps)
            rate_error = target - actual
            error = 0.0 if abs(rate_error) <= cfg.yaw_error_deadband_dps else rate_error / cfg.max_target_yaw_rate_dps
            integral = _clip(self._integral + error * elapsed, cfg.integral_limit)
            derivative = 0.0 if self._previous_error is None else (error - self._previous_error) / elapsed
            derivative = cfg.derivative_filter_alpha * derivative + (1 - cfg.derivative_filter_alpha) * self._derivative
            raw_correction = cfg.yaw_rate_kp * error + cfg.yaw_rate_ki * integral + cfg.yaw_rate_kd * derivative
        else:
            self._integral = 0.0
        correction = _clip(raw_correction, cfg.max_correction)
        requested = visual + cfg.imu_weight * correction
        bounded = _clip(requested, max_turn)
        smoothed = self._previous_turn + cfg.turn_smoothing_alpha * (bounded - self._previous_turn)
        final = self._previous_turn + _clip(smoothed - self._previous_turn, cfg.turn_slew_per_s * elapsed)
        final = _clip(final, max_turn)
        limited = (correction != raw_correction or bounded != requested or not math.isclose(final, smoothed, abs_tol=1e-12))
        if not limited:
            self._integral = integral
        self._previous_error, self._derivative, self._previous_turn = error, derivative, final
        return YawFeedbackResult(visual, final, feedback_active=feedback_active,
                                 target_yaw_rate_dps=target, yaw_rate_error_dps=rate_error,
                                 correction=cfg.imu_weight * correction, output_limited=limited)
