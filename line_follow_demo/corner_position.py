"""Conditional camera/contact interval intersection in the current body frame."""

from dataclasses import dataclass, replace
import math

from .corner_motion import _rotate


@dataclass(frozen=True)
class CornerPositionConfig:
    max_sensor_age_s: float
    max_yaw_rate_dps: float

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"invalid position_evidence.fusion.{name}")


@dataclass(frozen=True)
class CornerPositionEstimate:
    state: str = "UNAVAILABLE"
    reason: str = "position_fusion_unconfigured"
    timestamp_s: float | None = None
    track_id: int | None = None
    forward_interval_m: tuple[float, float] | None = None
    lateral_interval_m: tuple[float, float] | None = None
    sensor_timestamp_s: float | None = None
    sensor_age_s: float | None = None
    propagated_forward_interval_m: tuple[float, float] | None = None
    propagated_lateral_interval_m: tuple[float, float] | None = None
    propagation_translation_error_m: float = 0.
    propagation_rotation_error_m: float = 0.


def fuse_corner_position(motion, contact, *, now_s, imu, config, motion_config, contact_config):
    unavailable = CornerPositionEstimate(timestamp_s=now_s, track_id=motion.track_id)
    if (motion.state not in {"OBSERVED", "TRACKING", "PREDICTED"}
            or motion.timestamp_s != now_s or motion.corner_position_m is None
            or motion.position_error_m is None):
        return replace(unavailable, reason="current_visual_position_unavailable")
    x, y = motion.corner_position_m
    error = motion.position_error_m
    visual = replace(unavailable, state="VISUAL_ONLY", reason="contact_unconfirmed",
        forward_interval_m=(y-error, y+error), lateral_interval_m=(x-error, x+error))
    if contact.state not in {"CONTACT", "INCONSISTENT"}:
        return visual
    if contact.reference_track_id != motion.track_id or motion.track_id is None:
        return replace(visual, reason="contact_track_mismatch")
    if (contact.timestamp_s is None or not
            0 <= now_s-contact.timestamp_s <= config.max_sensor_age_s):
        return replace(visual, reason="contact_stale_or_future")
    keys = ("imu_delta_yaw_deg", "imu_timestamp_monotonic_s", "imu_age_s", "imu_yaw_rate_dps")
    if (not imu.get("imu_healthy") or not all(isinstance(imu.get(k), (int, float))
            and math.isfinite(imu[k]) for k in keys)
            or not 0 <= imu["imu_age_s"] <= motion_config.max_imu_skew_s
            or abs(now_s-imu["imu_timestamp_monotonic_s"]) > motion_config.max_imu_skew_s
            or abs(imu["imu_yaw_rate_dps"]) > config.max_yaw_rate_dps):
        return replace(unavailable, reason="current_imu_unavailable_or_unsynchronized")
    if contact.imu_recovery_count != imu.get("imu_recovery_count", 0):
        return replace(visual, reason="contact_imu_epoch_changed")
    if (contact.imu_timestamp_s is None or contact.imu_yaw_deg is None
            or contact.imu_yaw_rate_dps is None or abs(contact.imu_yaw_rate_dps) > config.max_yaw_rate_dps
            or abs(contact.timestamp_s-contact.imu_timestamp_s) > contact_config.max_imu_skew_s):
        return replace(visual, reason="contact_imu_unavailable_or_unsynchronized")
    age = now_s-contact.timestamp_s
    visual = replace(visual, sensor_timestamp_s=contact.timestamp_s, sensor_age_s=age)
    delta = imu["imu_delta_yaw_deg"]-contact.imu_yaw_deg
    yaw_time = abs(imu["imu_timestamp_monotonic_s"]-contact.imu_timestamp_s)
    if abs(delta) > config.max_yaw_rate_dps*yaw_time+motion_config.relative_yaw_error_deg+1e-9:
        return replace(visual, reason="contact_yaw_change_exceeds_motion_bound")
    if contact.state == "INCONSISTENT":
        return replace(visual, state="INCONSISTENT", reason="contact_disagrees_with_corner_pose",
            forward_interval_m=None, lateral_interval_m=None)
    if (contact.corner_x_interval_m is None or contact.corner_y_interval_m is None
            or not contact.outgoing_contact_required):
        return replace(visual, reason="contact_metric_band_unavailable")
    # Both contact intervals already include the earlier vision prior. Their
    # intersection adds constraints, never independent statistical confidence.
    corners = [(sx, sy) for sx in contact.corner_x_interval_m for sy in contact.corner_y_interval_m]
    rotated = [_rotate(point, delta) for point in corners]
    skew = abs(now_s-imu["imu_timestamp_monotonic_s"])+abs(contact.timestamp_s-contact.imu_timestamp_s)
    angle_error = min(180., motion_config.relative_yaw_error_deg+config.max_yaw_rate_dps*skew)
    rotation_error = 2*max(math.hypot(*p) for p in corners)*math.sin(math.radians(angle_error)/2)
    translation_error = max(motion_config.max_speed_mps, contact_config.max_speed_mps)*age
    expansion = rotation_error+translation_error
    lateral = min(p[0] for p in rotated)-expansion, max(p[0] for p in rotated)+expansion
    forward = min(p[1] for p in rotated)-expansion, max(p[1] for p in rotated)+expansion
    lateral_overlap = max(lateral[0], x-error), min(lateral[1], x+error)
    forward_overlap = max(forward[0], y-error), min(forward[1], y+error)
    fields = dict(propagated_forward_interval_m=forward, propagated_lateral_interval_m=lateral,
        propagation_translation_error_m=translation_error, propagation_rotation_error_m=rotation_error)
    if lateral_overlap[0] > lateral_overlap[1] or forward_overlap[0] > forward_overlap[1]:
        return replace(visual, state="INCONSISTENT", reason="propagated_contact_disagrees_with_vision",
            forward_interval_m=None, lateral_interval_m=None, **fields)
    return replace(visual, state="FUSED", reason="same_time_interval_intersection",
        forward_interval_m=forward_overlap, lateral_interval_m=lateral_overlap, **fields)
