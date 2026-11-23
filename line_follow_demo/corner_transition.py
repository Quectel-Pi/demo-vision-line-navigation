"""Offline turn/exit setpoints. IMU yaw and requested yaw rate are positive left."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ExitPathEvidence:
    state: str = "UNAVAILABLE"
    reason: str = "exit_path_unavailable"
    timestamp_s: float | None = None
    heading_deg: float | None = None
    heading_error_deg: float | None = None
    lateral_interval_m: tuple[float, float] | None = None
    steering_curvature_m_inv: float | None = None
    unobserved_distance_m: float | None = None


def exit_path_evidence(plan, *, timestamp_s, observation_unique, position_error_m):
    if not observation_unique or plan.state != "VALID":
        return ExitPathEvidence(reason="path_not_validated", timestamp_s=timestamp_s)
    if (plan.near_heading_deg is None or plan.near_heading_support_m is None
            or plan.near_heading_variation_deg is None or plan.lateral_error_m is None
            or plan.unobserved_distance_m is None or plan.steering_curvature_m_inv is None
            or plan.near_heading_support_m <= 2*position_error_m):
        return ExitPathEvidence(reason="near_tangent_unresolved", timestamp_s=timestamp_s)
    error = math.degrees(math.asin(2*position_error_m/plan.near_heading_support_m))
    error += plan.near_heading_variation_deg
    low, high = plan.near_heading_deg-error, plan.near_heading_deg+error
    if max(abs(low), abs(high)) >= 90:
        return ExitPathEvidence(reason="exit_not_forward", timestamp_s=timestamp_s)
    normal_error = position_error_m+plan.unobserved_distance_m*math.sin(math.radians(error))
    normal = plan.lateral_error_m-normal_error, plan.lateral_error_m+normal_error
    cos_min = math.cos(math.radians(max(abs(low), abs(high))))
    cos_max = math.cos(math.radians(0. if low <= 0 <= high else min(abs(low), abs(high))))
    lateral = [n/c for n in normal for c in (cos_min, cos_max)]
    return ExitPathEvidence(state="VALID", reason="ordered_near_path",
        timestamp_s=timestamp_s, heading_deg=plan.near_heading_deg, heading_error_deg=error,
        lateral_interval_m=(min(lateral), max(lateral)),
        steering_curvature_m_inv=plan.steering_curvature_m_inv,
        unobserved_distance_m=plan.unobserved_distance_m)


@dataclass(frozen=True)
class CornerTransitionConfig:
    max_target_yaw_rate_dps: float
    max_actual_yaw_rate_dps: float
    angle_tolerance_deg: float
    settled_yaw_rate_dps: float
    max_imu_age_s: float
    max_update_interval_s: float
    settle_duration_s: float
    turn_timeout_s: float
    exit_timeout_s: float
    exit_max_age_s: float
    exit_confirm_duration_s: float
    exit_confirm_samples: int
    exit_heading_tolerance_deg: float
    exit_capture_lateral_m: float
    exit_handover_lateral_m: float
    exit_max_unobserved_m: float
    exit_speed_mps: float
    position_error_m: float
    yaw_stop_latency_s: float | None = None
    yaw_brake_deceleration_dps2: float | None = None

    def __post_init__(self):
        optional = {"yaw_stop_latency_s", "yaw_brake_deceleration_dps2"}
        for name in self.__dataclass_fields__:
            if name in optional:
                continue
            value = getattr(self, name)
            if (isinstance(value, bool) or not math.isfinite(value)
                    or value < 0 or (name != "exit_max_unobserved_m" and value == 0)):
                raise ValueError(f"invalid corner_transition.{name}")
        if (type(self.exit_confirm_samples) is not int or self.exit_confirm_samples < 2
                or self.max_target_yaw_rate_dps > self.max_actual_yaw_rate_dps
                or not self.angle_tolerance_deg < self.exit_heading_tolerance_deg < 90
                or self.exit_handover_lateral_m > self.exit_capture_lateral_m):
            raise ValueError("invalid transition confirmation or limits")
        latency, decel = self.yaw_stop_latency_s, self.yaw_brake_deceleration_dps2
        if (latency is None) != (decel is None):
            raise ValueError("angular stopping bounds must be provided together")
        if latency is not None and (isinstance(latency, bool) or not math.isfinite(latency)
                or latency < 0 or isinstance(decel, bool) or not math.isfinite(decel) or decel <= 0):
            raise ValueError("invalid angular stopping bounds")


@dataclass(frozen=True)
class CornerTransitionDecision:
    phase: str = "IDLE"
    reason: str = "transition_unconfigured"
    track_id: int | None = None
    target_imu_yaw_deg: float | None = None
    angle_error_deg: float | None = None
    target_yaw_rate_dps: float = 0.
    yaw_rate_error_dps: float | None = None
    forward_speed_mps: float = 0.
    handover: bool = False
    exit_confirm_samples: int = 0
    timestamp_s: float | None = None


class CornerTransitionController:
    def __init__(self, config):
        self.config = config
        self.reset()

    @property
    def phase(self):
        return self._phase

    @property
    def timestamp_s(self):
        return self._last_time

    @property
    def latest_decision(self):
        return self._latest_decision

    def reset(self, reason="transition_reset"):
        self._phase, self._reason = "IDLE", reason
        self._context = self._epoch = self._last_time = None
        self._track_id = self._target = self._completed_track = None
        self._started = self._settled_since = self._exit_started = None
        self._line_since = self._last_line_time = None
        self._line_count = 0
        self._latest_decision = CornerTransitionDecision(reason=reason)

    def _decision(self, reason, *, error=None, target_rate=0., actual_rate=None,
                  forward=0., handover=False):
        decision = CornerTransitionDecision(phase=self._phase, reason=reason,
            track_id=self._track_id, target_imu_yaw_deg=self._target, angle_error_deg=error,
            target_yaw_rate_dps=target_rate,
            yaw_rate_error_dps=None if actual_rate is None else target_rate-actual_rate,
            forward_speed_mps=forward, handover=handover, exit_confirm_samples=self._line_count,
            timestamp_s=self._last_time)
        self._latest_decision = decision
        return decision

    def _fail(self, reason):
        self._phase, self._reason = "FAILED", reason
        return self._decision(reason)

    def update(self, approach, *, now_s, imu, exit_path=None, context=()):
        cfg = self.config
        if self._phase == "FAILED":
            return self._decision(self._reason)
        if self._last_time is not None and not 0 < now_s-self._last_time <= cfg.max_update_interval_s:
            return self._fail("control_timestamp_gap")
        self._last_time = now_s
        if self._context is not None and context != self._context:
            self.reset("control_context_changed")
            self._context = context
            return self._decision("control_context_changed")
        self._context = context
        keys = ("imu_delta_yaw_deg", "imu_yaw_rate_dps", "imu_timestamp_monotonic_s", "imu_age_s")
        if (not imu.get("imu_healthy") or not all(isinstance(imu.get(k), (int, float))
                and math.isfinite(imu[k]) for k in keys)
                or not 0 <= imu["imu_age_s"] <= cfg.max_imu_age_s
                or abs(now_s-imu["imu_timestamp_monotonic_s"]) > cfg.max_imu_age_s):
            return self._fail("imu_unavailable_or_stale")
        epoch = imu.get("imu_recovery_count", 0)
        if self._epoch is not None and epoch != self._epoch:
            return self._fail("imu_epoch_changed")
        self._epoch = epoch
        yaw, rate = imu["imu_delta_yaw_deg"], imu["imu_yaw_rate_dps"]
        if abs(rate) > cfg.max_actual_yaw_rate_dps:
            return self._fail("actual_yaw_rate_exceeded")
        if self._phase in {"IDLE", "COMPLETE"}:
            if approach.phase != "TURN_READY":
                return self._decision("awaiting_turn_ready")
            if (approach.timestamp_s is None or not 0 <= now_s-approach.timestamp_s <= cfg.max_update_interval_s
                    or approach.track_id is None or approach.target_exit_heading_deg is None):
                return self._decision("turn_reference_unavailable_or_stale")
            if approach.track_id == self._completed_track:
                return self._decision("completed_corner_not_rearmed")
            if cfg.yaw_stop_latency_s is None:
                return self._decision("angular_stop_model_unconfigured")
            self._track_id = approach.track_id
            self._target = yaw-approach.target_exit_heading_deg
            self._started, self._settled_since = now_s, None
            self._exit_started = self._line_since = self._last_line_time = None
            self._line_count = 0
            self._phase = "TURN"
        difference = self._target-yaw
        error = (difference+180) % 360-180
        if error == -180 and difference > 0:
            error = 180.
        if self._phase in {"TURN", "SETTLE"}:
            if now_s-self._started > cfg.turn_timeout_s:
                return self._fail("turn_timeout")
            if abs(error) <= cfg.angle_tolerance_deg and abs(rate) <= cfg.settled_yaw_rate_dps:
                self._phase = "SETTLE"
                self._settled_since = now_s if self._settled_since is None else self._settled_since
                if now_s-self._settled_since >= cfg.settle_duration_s:
                    self._phase, self._exit_started = "EXIT", now_s
                else:
                    return self._decision("confirming_angular_settle", error=error, actual_rate=rate)
            else:
                self._settled_since = None
                self._phase = "TURN"
                remaining = abs(error)
                decel = cfg.yaw_brake_deceleration_dps2
                latency = cfg.yaw_stop_latency_s+cfg.max_update_interval_s
                safe_rate = decel*(math.sqrt(latency*latency+2*remaining/decel)-latency)
                target_rate = math.copysign(min(cfg.max_target_yaw_rate_dps, safe_rate), error)
                stopping_angle = abs(rate)*latency+rate*rate/(2*decel)
                if (abs(error) <= cfg.angle_tolerance_deg
                        or (rate*error > 0 and abs(error) <= stopping_angle)
                        or (rate*error < 0 and abs(rate) > cfg.settled_yaw_rate_dps)):
                    target_rate = 0.
                return self._decision("angular_feedback", error=error,
                                      target_rate=target_rate, actual_rate=rate)
        if now_s-self._exit_started > cfg.exit_timeout_s:
            return self._fail("exit_path_timeout")
        evidence = exit_path or ExitPathEvidence()
        valid = (evidence.state == "VALID" and evidence.timestamp_s is not None
                 and 0 <= now_s-evidence.timestamp_s <= cfg.exit_max_age_s
                 and evidence.heading_deg is not None and evidence.heading_error_deg is not None
                 and evidence.lateral_interval_m is not None
                 and evidence.steering_curvature_m_inv is not None
                 and evidence.unobserved_distance_m is not None
                 and evidence.unobserved_distance_m <= cfg.exit_max_unobserved_m)
        if valid:
            remaining_right = -error
            heading_difference = abs((evidence.heading_deg-remaining_right+180) % 360-180)
            lateral = max(abs(v) for v in evidence.lateral_interval_m)
            valid = (heading_difference+evidence.heading_error_deg <= cfg.exit_heading_tolerance_deg
                     and abs(evidence.heading_deg)+evidence.heading_error_deg <= cfg.exit_heading_tolerance_deg
                     and lateral <= cfg.exit_capture_lateral_m)
        if not valid:
            self._line_since, self._line_count = None, 0
            return self._decision("awaiting_associated_exit", error=error, actual_rate=rate)
        if self._last_line_time is not None and evidence.timestamp_s <= self._last_line_time:
            return self._decision("awaiting_new_exit_frame", error=error, actual_rate=rate)
        if (self._last_line_time is not None
                and evidence.timestamp_s-self._last_line_time > cfg.exit_max_age_s):
            self._line_since, self._line_count = None, 0
        self._last_line_time = evidence.timestamp_s
        if lateral <= cfg.exit_handover_lateral_m:
            self._line_since = evidence.timestamp_s if self._line_since is None else self._line_since
            self._line_count += 1
        else:
            self._line_since, self._line_count = None, 0
        target_rate = -math.degrees(cfg.exit_speed_mps*evidence.steering_curvature_m_inv)
        target_rate = max(-cfg.max_target_yaw_rate_dps, min(cfg.max_target_yaw_rate_dps, target_rate))
        if (self._line_count >= cfg.exit_confirm_samples
                and evidence.timestamp_s-self._line_since >= cfg.exit_confirm_duration_s):
            self._phase, self._completed_track = "COMPLETE", self._track_id
            return self._decision("exit_handover_ready", error=error, target_rate=target_rate,
                                  actual_rate=rate, forward=cfg.exit_speed_mps, handover=True)
        return self._decision("limited_exit_tracking", error=error, target_rate=target_rate,
                              actual_rate=rate, forward=cfg.exit_speed_mps)
