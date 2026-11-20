"""Metric approach decisions for offline review; does not send wheel commands."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class CornerApproachConfig:
    position_tolerance_m: float
    lateral_tolerance_m: float
    max_position_error_m: float
    stopped_speed_tolerance_mps: float
    stopped_yaw_rate_tolerance_dps: float
    max_input_age_s: float
    max_prediction_age_s: float

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"invalid corner_approach.{name}")


@dataclass(frozen=True)
class CornerApproachDecision:
    phase: str = "HOLD"
    reason: str = "approach_unconfigured"
    track_id: int | None = None
    current_forward_interval_m: tuple[float, float] | None = None
    current_lateral_interval_m: tuple[float, float] | None = None
    stopped_forward_interval_m: tuple[float, float] | None = None
    advance_margin_m: float | None = None
    target_exit_heading_deg: float | None = None
    timestamp_s: float | None = None


def decide_corner_approach(motion, config, *, now_s, yaw_rate_dps=None, position=None):
    def result(phase, reason, **fields):
        return CornerApproachDecision(phase=phase, reason=reason,
            track_id=motion.track_id, target_exit_heading_deg=motion.outgoing_heading_deg,
            timestamp_s=now_s,
            **fields)

    if motion.state not in {"OBSERVED", "TRACKING", "PREDICTED"}:
        return result("HOLD", "motion_unavailable")
    if (motion.timestamp_s is None or not 0 <= now_s-motion.timestamp_s <= config.max_input_age_s
            or motion.age_s is None or not 0 <= motion.age_s <= config.max_prediction_age_s):
        return result("HOLD", "motion_not_fresh")
    if motion.corner_position_m is None or motion.position_error_m is None:
        return result("HOLD", "position_error_exceeded")
    x, y = motion.corner_position_m
    error = motion.position_error_m
    forward, lateral = (y-error, y+error), (x-error, x+error)
    stopping = motion.stopped_corner_forward_interval_m
    if position is not None:
        if position.state not in {"VISUAL_ONLY", "FUSED"}:
            return result("HOLD", position.reason)
        if position.timestamp_s != motion.timestamp_s or position.track_id != motion.track_id:
            return result("HOLD", "position_frame_or_track_mismatch")
        forward, lateral = position.forward_interval_m, position.lateral_interval_m
        error = max(forward[1]-forward[0], lateral[1]-lateral[0])/2
        distances = motion.stop_distance_interval_m
        stopping = (None if distances is None else
                    (forward[0]-distances[1], forward[1]-distances[0]))
    if error > config.max_position_error_m:
        return result("HOLD", "position_error_exceeded")
    fields = dict(current_forward_interval_m=forward,
                  current_lateral_interval_m=lateral,
                  stopped_forward_interval_m=stopping)
    tolerance = config.position_tolerance_m
    if forward[1] < -tolerance:
        return result("HOLD", "corner_already_passed", **fields)
    if max(abs(v) for v in lateral) > config.lateral_tolerance_m:
        return result("HOLD", "lateral_position_not_ready", **fields)
    at_corner = max(abs(v) for v in forward) <= tolerance
    if at_corner:
        if motion.velocity_mps is None or motion.velocity_error_mps is None:
            return result("BRAKE", "stopped_speed_unresolved", **fields)
        speed_upper = math.hypot(*motion.velocity_mps)+motion.velocity_error_mps
        if speed_upper > config.stopped_speed_tolerance_mps:
            return result("BRAKE", "vehicle_not_settled", **fields)
        if yaw_rate_dps is None:
            return result("BRAKE", "angular_rate_unavailable", **fields)
        if abs(yaw_rate_dps) > config.stopped_yaw_rate_tolerance_dps:
            return result("BRAKE", "vehicle_still_rotating", **fields)
        if motion.outgoing_heading_deg is None:
            return result("HOLD", "exit_heading_unavailable", **fields)
        return result("TURN_READY", "position_and_speed_in_bounds", **fields)
    if forward[0] <= tolerance:
        return result("BRAKE", "position_interval_overlaps_turn_window", **fields)
    if stopping is None:
        return result("HOLD", motion.stop_reason, **fields)
    # The nearest possible stopping position must still be ahead of the
    # turn window before additional forward displacement can be considered.
    margin = stopping[0]-tolerance
    if margin <= 0:
        return result("BRAKE", "braking_window_reached", advance_margin_m=margin, **fields)
    return result("ADVANCE", "stopping_interval_before_corner",
                  advance_margin_m=margin, **fields)
