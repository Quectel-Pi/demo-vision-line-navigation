"""Calibrated logical wheel proposals; no device is opened by this module."""

from dataclasses import dataclass, field
import math

from .motor import model_a_config_from_dict
from .motor.hiwonder_model_a import MODEL_A_COMMAND_SOURCE_SHA256, model_a_wheel_commands
from .turning import mix_forward_turn_duty
from .yaw_rate_feedback import forward_motor_profile_key


def _validate_points(points):
    if len(points) < 3 or (0., 0.) not in [tuple(p) for p in points]:
        raise ValueError("response requires both directions and an exact zero reference")
    previous = (-math.inf, -math.inf)
    for duty, rate in points:
        if (any(isinstance(v, bool) or not math.isfinite(v) for v in (duty, rate))
                or duty <= previous[0] or rate <= previous[1] or duty*rate < 0
                or (duty == 0) != (rate == 0)):
            raise ValueError("response duty and rate must be finite, signed and strictly increasing")
        previous = duty, rate
    if not points[0][0] < 0 < points[-1][0]:
        raise ValueError("response must cover both turn directions")


@dataclass(frozen=True)
class ForwardDriveResponse:
    speed_mps: float
    forward_duty: float
    yaw_points: tuple

    def __post_init__(self):
        if any(isinstance(v, bool) or not math.isfinite(v) or v <= 0
               for v in (self.speed_mps, self.forward_duty)):
            raise ValueError("forward speed and duty must be positive")
        _validate_points(self.yaw_points)
        if max(abs(p[0]) for p in self.yaw_points) > self.forward_duty:
            raise ValueError("forward response must not reverse the inner wheels")


@dataclass(frozen=True)
class CornerDriveCalibration:
    pivot_verified: bool = False
    forward_verified: bool = False
    motor_profile_key: str = ""
    driver_source_sha256: str = ""
    calibration_record: str = ""
    pivot_points: tuple = ()
    forward_rows: tuple = ()

    def __post_init__(self):
        if type(self.pivot_verified) is not bool or type(self.forward_verified) is not bool:
            raise ValueError("calibration verification flags must be boolean")
        if self.pivot_verified:
            _validate_points(self.pivot_points)
        if self.forward_verified and not self.forward_rows:
            raise ValueError("verified forward calibration requires measured speed rows")
        speeds = [r.speed_mps for r in self.forward_rows]
        if speeds != sorted(set(speeds)):
            raise ValueError("forward speed rows must be unique and increasing")
        if (self.pivot_verified or self.forward_verified) and not all(
                isinstance(v, str) and v.strip() for v in
                (self.motor_profile_key, self.driver_source_sha256, self.calibration_record)):
            raise ValueError("verified response requires motor/source fingerprints and evidence record")


@dataclass(frozen=True)
class CornerDriveConfig:
    yaw_rate_kp_duty_per_dps: float
    max_correction_duty: float
    max_logical_wheel_duty: float
    max_actual_yaw_rate_dps: float
    max_imu_age_s: float
    max_command_age_s: float
    max_update_interval_s: float
    calibration: CornerDriveCalibration = field(default_factory=CornerDriveCalibration)
    execution_enabled: bool = False

    def __post_init__(self):
        if type(self.execution_enabled) is not bool:
            raise ValueError("execution_enabled must be boolean")
        for name in ("yaw_rate_kp_duty_per_dps", "max_correction_duty", "max_logical_wheel_duty",
                     "max_actual_yaw_rate_dps", "max_imu_age_s", "max_command_age_s", "max_update_interval_s"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not math.isfinite(value) or value < 0
                    or (name not in {"yaw_rate_kp_duty_per_dps", "max_correction_duty"} and value == 0)):
                raise ValueError(f"invalid corner_drive.{name}")
        if self.max_logical_wheel_duty > 100:
            raise ValueError("logical wheel limit must not exceed100")

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        calibration = dict(values.pop("calibration", {}))
        calibration["forward_rows"] = tuple(ForwardDriveResponse(**r)
            for r in calibration.get("forward_rows", ()))
        return cls(calibration=CornerDriveCalibration(**calibration), **values)


@dataclass(frozen=True)
class CornerDriveCommand:
    state: str = "HOLD"
    reason: str = "drive_unconfigured"
    timestamp_s: float | None = None
    requested_at_s: float | None = None
    mode: str = "none"
    wheels: tuple = (0., 0., 0., 0.)
    sdk_commands: tuple = ()
    forward_duty: float = 0.
    feedforward_turn_duty: float = 0.
    feedback_turn_duty: float = 0.
    turn_duty: float = 0.
    target_yaw_rate_dps: float = 0.
    requested_forward_speed_mps: float = 0.
    limited: bool = False
    motor_profile_key: str = ""


class CornerDriveMapper:
    def __init__(self, config):
        self.config = config
        self.reset()

    def reset(self):
        self._fault = self._last_time = None

    def profile_matches(self, motor_config, turn_duty_max):
        calibration = self.config.calibration
        return (motor_config.get("driver", "hiwonder_model_a") == "hiwonder_model_a"
            and motor_config.get("mixer", "duty") == "duty"
            and calibration.driver_source_sha256 == MODEL_A_COMMAND_SOURCE_SHA256
            and calibration.motor_profile_key == forward_motor_profile_key(motor_config, turn_duty_max))

    def _fault_command(self, reason, now_s, requested_at_s):
        self._fault = reason
        return CornerDriveCommand(state="FAULT", reason=reason, timestamp_s=now_s,
                                  requested_at_s=requested_at_s)

    def validate_dispatch(self, command, *, now_s, motor_config, turn_duty_max):
        calibration = self.config.calibration
        verified = (calibration.pivot_verified if command.mode == "pivot"
                    else calibration.forward_verified if command.mode == "forward" else False)
        if (not verified or not self.profile_matches(motor_config, turn_duty_max)
                or command.motor_profile_key != calibration.motor_profile_key):
            raise ValueError("corner drive profile is no longer valid")
        if (command.state != "READY" or command.requested_at_s is None or command.timestamp_s is None
                or not 0 <= now_s-command.requested_at_s <= self.config.max_command_age_s
                or not 0 <= now_s-command.timestamp_s <= self.config.max_command_age_s):
            raise ValueError("corner drive command is unavailable or stale")
        sdk, clipped = model_a_wheel_commands(model_a_config_from_dict(motor_config), command.wheels)
        if (clipped or max(abs(v) for v in command.wheels) > self.config.max_logical_wheel_duty
                or tuple(tuple(p) for p in sdk) != command.sdk_commands):
            raise ValueError("corner drive command no longer matches bounded preview")

    def map_setpoint(self, forward_speed_mps, yaw_rate_dps, *, now_s, requested_at_s,
                     imu, motor_config, turn_duty_max):
        cfg, calibration = self.config, self.config.calibration
        if self._fault is not None:
            return self._fault_command(self._fault, now_s, requested_at_s)
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
               for v in (forward_speed_mps, yaw_rate_dps, now_s)):
            return self._fault_command("setpoint_invalid", now_s, requested_at_s)
        if forward_speed_mps == 0 and yaw_rate_dps == 0:
            self._last_time = None
            return CornerDriveCommand(state="STOP", reason="zero_target", timestamp_s=now_s,
                                      requested_at_s=requested_at_s)
        if forward_speed_mps < 0:
            return self._fault_command("reverse_not_in_corner_profile", now_s, requested_at_s)
        if requested_at_s is None or not 0 <= now_s-requested_at_s <= cfg.max_command_age_s:
            return self._fault_command("setpoint_stale_or_future", now_s, requested_at_s)
        mode = "pivot" if forward_speed_mps == 0 else "forward"
        if not (calibration.pivot_verified if mode == "pivot" else calibration.forward_verified):
            return CornerDriveCommand(reason=mode+"_response_unverified", timestamp_s=now_s,
                                      requested_at_s=requested_at_s, mode=mode)
        if not self.profile_matches(motor_config, turn_duty_max):
            return self._fault_command("motor_or_driver_profile_changed", now_s, requested_at_s)
        if self._last_time is not None and not 0 < now_s-self._last_time <= cfg.max_update_interval_s:
            return self._fault_command("drive_update_gap", now_s, requested_at_s)
        keys = ("imu_yaw_rate_dps", "imu_timestamp_monotonic_s", "imu_age_s")
        if (not imu.get("imu_healthy") or not all(isinstance(imu.get(k), (int, float))
                and math.isfinite(imu[k]) for k in keys)
                or not 0 <= imu["imu_age_s"] <= cfg.max_imu_age_s
                or abs(now_s-imu["imu_timestamp_monotonic_s"]) > cfg.max_imu_age_s):
            return self._fault_command("imu_unavailable_or_stale", now_s, requested_at_s)
        actual_rate = imu["imu_yaw_rate_dps"]
        if abs(actual_rate) > cfg.max_actual_yaw_rate_dps:
            return self._fault_command("actual_yaw_rate_exceeded", now_s, requested_at_s)
        self._last_time = now_s
        forward = 0.
        if mode == "pivot":
            points = calibration.pivot_points
        else:
            rows = [r for r in calibration.forward_rows
                    if math.isclose(r.speed_mps, forward_speed_mps, rel_tol=0, abs_tol=1e-9)]
            if not rows:
                return CornerDriveCommand(reason="forward_speed_not_measured", timestamp_s=now_s,
                                          requested_at_s=requested_at_s, mode=mode)
            forward, points = rows[0].forward_duty, rows[0].yaw_points
        feedforward = None
        if yaw_rate_dps == 0:
            feedforward = 0.
        else:
            branch = [p for p in points if p[1]*yaw_rate_dps > 0]
            for duty, rate in branch:
                if math.isclose(rate, yaw_rate_dps, rel_tol=0, abs_tol=1e-9):
                    feedforward = duty
            for a, b in zip(branch, branch[1:]):
                if a[1] <= yaw_rate_dps <= b[1]:
                    feedforward = a[0]+(b[0]-a[0])*(yaw_rate_dps-a[1])/(b[1]-a[1])
        if feedforward is None:
            return CornerDriveCommand(reason="yaw_rate_not_in_measured_branch", timestamp_s=now_s,
                                      requested_at_s=requested_at_s, mode=mode)
        correction = cfg.yaw_rate_kp_duty_per_dps*(yaw_rate_dps-actual_rate)
        correction = max(-cfg.max_correction_duty, min(cfg.max_correction_duty, correction))
        requested = feedforward+correction
        branch = [p[0] for p in points if p[0]*requested > 0]
        turn = requested
        if requested != 0:
            if (feedforward != 0 and feedforward*requested < 0) or abs(requested) < min(abs(v) for v in branch):
                turn = 0.
            else:
                turn = math.copysign(min(abs(requested), max(abs(v) for v in branch)), requested)
        if abs(turn) > turn_duty_max:
            return self._fault_command("turn_limit_below_response_table", now_s, requested_at_s)
        wheels = mix_forward_turn_duty(forward, turn)
        if max(abs(v) for v in wheels) > cfg.max_logical_wheel_duty:
            return self._fault_command("logical_wheel_limit_exceeded", now_s, requested_at_s)
        sdk, clipped = model_a_wheel_commands(model_a_config_from_dict(motor_config), wheels)
        if clipped:
            return self._fault_command("driver_would_clip_calibrated_command", now_s, requested_at_s)
        return CornerDriveCommand(state="READY", reason="measured_response_mapping",
            timestamp_s=now_s, requested_at_s=requested_at_s, mode=mode, wheels=wheels,
            sdk_commands=tuple(tuple(p) for p in sdk), forward_duty=forward,
            feedforward_turn_duty=feedforward, feedback_turn_duty=turn-feedforward,
            turn_duty=turn, target_yaw_rate_dps=yaw_rate_dps,
            requested_forward_speed_mps=forward_speed_mps,
            limited=turn != requested or correction != cfg.yaw_rate_kp_duty_per_dps*(yaw_rate_dps-actual_rate),
            motor_profile_key=calibration.motor_profile_key)
