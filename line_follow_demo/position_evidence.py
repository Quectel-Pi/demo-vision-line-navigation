"""Partial position observations; these results never authorize wheel motion."""

from __future__ import annotations

from dataclasses import dataclass, replace
from collections import deque
import math

import numpy as np


def _product(a, b):
    values = [x*y for x in a for y in b]
    return min(values), max(values)


def _trig_interval(angle_deg, error_deg, function):
    low, high = math.radians(angle_deg-error_deg), math.radians(angle_deg+error_deg)
    samples = [low, high]
    samples.extend(k*math.pi/2 for k in range(math.ceil(low/(math.pi/2)), math.floor(high/(math.pi/2))+1))
    values = [function(x) for x in samples]
    if any(abs(v) < 1e-15 for v in values):
        values.append(0.)
    return min(values), max(values)


@dataclass(frozen=True)
class SensorContactConfig:
    channel_x_m: tuple[float, float, float, float]
    row_y_m: float
    row_center_x_m: float
    tape_half_width_max_m: float
    detection_radius_m: float
    probe_position_error_m: float
    row_position_error_m: float
    heading_error_deg: float
    max_heading_age_s: float
    max_speed_mps: float
    max_imu_skew_s: float
    contact_duration_s: float
    max_sample_gap_s: float

    def __post_init__(self):
        if (len(self.channel_x_m) != 4 or not all(math.isfinite(v) for v in self.channel_x_m)
                or list(self.channel_x_m) != sorted(set(self.channel_x_m))):
            raise ValueError("probe positions must follow confirmed left-to-right channel order")
        for name in ("row_y_m", "row_center_x_m", "tape_half_width_max_m", "detection_radius_m",
                     "probe_position_error_m", "row_position_error_m", "heading_error_deg", "max_heading_age_s",
                     "max_speed_mps", "max_imu_skew_s", "contact_duration_s", "max_sample_gap_s"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value):
                raise ValueError(f"invalid sensor contact {name}")
            if name != "row_center_x_m" and value < 0:
                raise ValueError(f"invalid sensor contact {name}")
        if (self.tape_half_width_max_m <= 0 or self.max_heading_age_s <= 0
                or self.max_imu_skew_s <= 0 or self.max_sample_gap_s <= 0
                or not 0 < self.contact_duration_s <= self.max_sample_gap_s
                or self.heading_error_deg >= 90):
            raise ValueError("invalid sensor contact limits")


@dataclass(frozen=True)
class SensorContactEvidence:
    state: str = "UNAVAILABLE"
    reason: str = "sensor_geometry_unconfigured"
    timestamp_s: float | None = None
    normal_span_min_m: float | None = None
    incoming_strip_max_m: float | None = None
    outgoing_contact_required: bool = False
    corner_y_interval_m: tuple[float, float] | None = None
    contact_band_y_interval_m: tuple[float, float] | None = None
    prior_corner_y_interval_m: tuple[float, float] | None = None
    last_contact_at_s: float | None = None
    heading_observed_at_s: float | None = None
    sample_duration_s: float = 0.
    timing_position_error_m: float = 0.
    timing_heading_error_deg: float = 0.
    reference_track_id: int | None = None
    corner_x_interval_m: tuple[float, float] | None = None
    imu_yaw_deg: float | None = None
    imu_timestamp_s: float | None = None
    imu_recovery_count: int | None = None
    imu_yaw_rate_dps: float | None = None


class SensorContactObserver:
    def __init__(self, config: SensorContactConfig):
        self.config = config
        self.reset()

    def reset(self):
        self._references = deque()
        self._sample_track_id = None
        self._context = self._epoch = self._last_time = None
        self._contact_since = self._clear_since = None
        self._contact_samples = self._clear_samples = 0
        self._confirmed = False
        self._last_contact = None

    def invalidate_samples(self):
        self._contact_since = self._clear_since = None
        self._contact_samples = self._clear_samples = 0
        self._confirmed, self._last_contact = False, None

    def observe_corner(self, motion, *, imu, context=()):
        """Register a vision-time reference without inventing a sensor sample."""
        if motion is None or motion.state not in {"OBSERVED", "TRACKING"} or motion.age_s != 0:
            return False
        cfg = self.config
        if (not imu.get("imu_healthy") or imu.get("imu_timestamp_monotonic_s") is None
                or abs(motion.observed_at_s-imu["imu_timestamp_monotonic_s"]) > cfg.max_imu_skew_s):
            self.reset()
            return False
        epoch = imu.get("imu_recovery_count", 0)
        if context != self._context or epoch != self._epoch:
            self.reset()
        self._context, self._epoch = context, epoch
        if self._references and motion.observed_at_s <= self._references[-1][0].observed_at_s:
            return motion.observed_at_s == self._references[-1][0].observed_at_s
        self._references.append((motion, imu["imu_delta_yaw_deg"]))
        while motion.observed_at_s-self._references[0][0].observed_at_s > cfg.max_heading_age_s:
            self._references.popleft()
        return True

    def update(self, channels, motion, *, timestamp_s, imu, active, context=(),
               sample_started_s=None, yaw_rate_bound_dps=0.):
        cfg = self.config
        sample_started_s = timestamp_s if sample_started_s is None else sample_started_s
        if (not math.isfinite(sample_started_s) or sample_started_s > timestamp_s
                or not math.isfinite(yaw_rate_bound_dps) or yaw_rate_bound_dps < 0):
            self.reset()
            return SensorContactEvidence(reason="sample_interval_invalid")
        if (not isinstance(timestamp_s, (int, float)) or not math.isfinite(timestamp_s)
                or (self._last_time is not None and timestamp_s <= self._last_time)):
            self.reset()
            return SensorContactEvidence(reason="timestamp_invalid_or_not_increasing")
        if not active:
            self.reset()
            return SensorContactEvidence(reason="not_in_corner_approach", timestamp_s=timestamp_s)
        keys = ("imu_delta_yaw_deg", "imu_yaw_rate_dps", "imu_timestamp_monotonic_s", "imu_age_s")
        if (not imu.get("imu_healthy") or not all(isinstance(imu.get(k), (int, float))
                and math.isfinite(imu[k]) for k in keys)
                or not 0 <= imu["imu_age_s"] <= cfg.max_imu_skew_s
                or abs(timestamp_s-imu["imu_timestamp_monotonic_s"]) > cfg.max_imu_skew_s):
            self.reset()
            return SensorContactEvidence(reason="imu_unavailable_or_unsynchronized", timestamp_s=timestamp_s)
        if timestamp_s > sample_started_s and (yaw_rate_bound_dps <= 0 or abs(imu["imu_yaw_rate_dps"]) > yaw_rate_bound_dps):
            self.invalidate_samples()
            return SensorContactEvidence(reason="sample_yaw_rate_exceeds_timing_bound", timestamp_s=timestamp_s)
        epoch = imu.get("imu_recovery_count", 0)
        if context != self._context or epoch != self._epoch:
            self.reset()
        elif self._last_time is not None and timestamp_s-self._last_time > cfg.max_sample_gap_s:
            self.invalidate_samples()
        self._context, self._epoch, self._last_time = context, epoch, timestamp_s
        if motion is not None and motion.state in {"OBSERVED", "TRACKING"}:
            if motion.observed_at_s > sample_started_s:
                self.invalidate_samples()
                return SensorContactEvidence(reason="corner_reference_is_from_future", timestamp_s=timestamp_s)
            self.observe_corner(motion, imu=imu, context=context)
        if channels is None or len(channels) != 4 or any(type(v) is not bool for v in channels):
            self.invalidate_samples()
            return SensorContactEvidence(reason="sensor_sample_unavailable", timestamp_s=timestamp_s)
        if not self._references:
            return SensorContactEvidence(reason="corner_heading_unavailable", timestamp_s=timestamp_s)
        # A completed I2C read may be published after the next image arrived.
        selected = next((item for item in reversed(self._references) if item[0].observed_at_s <= sample_started_s), None)
        if selected is None:
            self.invalidate_samples()
            return SensorContactEvidence(reason="sample_precedes_corner_reference", timestamp_s=timestamp_s)
        reference, yaw = selected
        if reference.track_id != self._sample_track_id:
            self.invalidate_samples()
            self._sample_track_id = reference.track_id
        age = timestamp_s-reference.observed_at_s
        if age > cfg.max_heading_age_s:
            self.invalidate_samples()
            return SensorContactEvidence(reason="corner_heading_expired", timestamp_s=timestamp_s)
        delta = imu["imu_delta_yaw_deg"]-yaw
        incoming = reference.incoming_heading_deg+delta
        outgoing = reference.outgoing_heading_deg+delta
        duration = timestamp_s-sample_started_s
        timing_position_error = cfg.max_speed_mps*duration
        timing_heading_error = yaw_rate_bound_dps*duration
        heading_error = min(180., cfg.heading_error_deg+timing_heading_error)
        x_values = [x+cfg.row_center_x_m for x, on in zip(cfg.channel_x_m, channels) if on]
        half_width = cfg.tape_half_width_max_m+cfg.detection_radius_m+cfg.probe_position_error_m
        # A common mounting offset cancels from pairwise probe separation.
        # It still enlarges the metric contact band below.
        cos_in = _trig_interval(incoming, heading_error, math.cos)
        min_cos = 0. if cos_in[0] <= 0 <= cos_in[1] else min(abs(v) for v in cos_in)
        span = (max(x_values)-min(x_values))*min_cos if x_values else 0.
        required = span > 2*half_width
        result = SensorContactEvidence(timestamp_s=timestamp_s, normal_span_min_m=span,
            incoming_strip_max_m=2*half_width, outgoing_contact_required=required,
            last_contact_at_s=self._last_contact, heading_observed_at_s=reference.observed_at_s,
            sample_duration_s=duration, timing_position_error_m=timing_position_error,
            timing_heading_error_deg=timing_heading_error, reference_track_id=reference.track_id,
            imu_yaw_deg=imu["imu_delta_yaw_deg"], imu_timestamp_s=imu["imu_timestamp_monotonic_s"],
            imu_recovery_count=epoch, imu_yaw_rate_dps=imu["imu_yaw_rate_dps"])
        if not required:
            self._contact_since, self._contact_samples = None, 0
            if any(channels):
                self._clear_since, self._clear_samples = None, 0
                return replace(result, state="AMBIGUOUS", reason="compatible_with_incoming_strip")
            if not self._confirmed:
                return replace(result, state="IDLE", reason="no_black_contact")
            self._clear_since = timestamp_s if self._clear_since is None else self._clear_since
            self._clear_samples += 1
            released = self._clear_samples >= 2 and timestamp_s-self._clear_since >= cfg.contact_duration_s
            if released:
                self._confirmed = False
                self._clear_since, self._clear_samples = None, 0
            return replace(result, state="RELEASED" if released else "CLEARING",
                           reason="contact_cleared_without_new_position" if released else "confirming_clear")
        self._clear_since, self._clear_samples = None, 0
        self._contact_since = timestamp_s if self._contact_since is None else self._contact_since
        self._contact_samples += 1
        confirmed = self._contact_samples >= 2 and timestamp_s-self._contact_since >= cfg.contact_duration_s
        # At least one active probe must touch the exit under the two-leg
        # tape model. The union avoids pretending we know which one it is.
        angle = math.radians(delta)
        x, y = reference.corner_position_m
        centre_x = math.cos(angle)*x+math.sin(angle)*y
        centre_y = -math.sin(angle)*x+math.cos(angle)*y
        lateral_error = (reference.position_error_m+cfg.max_speed_mps*age
                         +2*math.hypot(x, y)*math.sin(math.radians(heading_error)/2))
        prior_y = centre_y-lateral_error, centre_y+lateral_error
        result = replace(result, corner_x_interval_m=(centre_x-lateral_error, centre_x+lateral_error))
        sin_out = _trig_interval(outgoing, heading_error, math.sin)
        cos_out = _trig_interval(outgoing, heading_error, math.cos)
        if sin_out[0] <= 0 <= sin_out[1]:
            band, interval, reason = None, None, "exit_direction_cannot_constrain_forward_position"
        else:
            ratio = _product(cos_out, (-1/sin_out[0], -1/sin_out[1]))
            width = (half_width+cfg.row_position_error_m+timing_position_error)/min(abs(v) for v in sin_out)
            intervals = []
            for probe_x in x_values:
                projected = _product(ratio, (probe_x-centre_x-lateral_error, probe_x-centre_x+lateral_error))
                intervals.append((cfg.row_y_m+projected[0]-width, cfg.row_y_m+projected[1]+width))
            band = min(i[0] for i in intervals), max(i[1] for i in intervals)
            interval = max(band[0], prior_y[0]), min(band[1], prior_y[1])
            if interval[0] > interval[1]:
                self._contact_since, self._contact_samples = None, 0
                self._confirmed, self._last_contact = False, None
                return replace(result, state="INCONSISTENT", reason="contact_disagrees_with_corner_pose",
                    contact_band_y_interval_m=band, prior_corner_y_interval_m=prior_y,
                    last_contact_at_s=None)
            reason = "conditional_exit_contact_band"
        if confirmed:
            self._confirmed, self._last_contact = True, timestamp_s
        return replace(result, state="CONTACT" if confirmed else "CANDIDATE", reason=reason,
                       corner_y_interval_m=interval, contact_band_y_interval_m=band,
                       prior_corner_y_interval_m=prior_y, last_contact_at_s=self._last_contact)


@dataclass(frozen=True)
class LinePoseConfig:
    front_to_control_center_m: float
    observation_error_m: float
    min_span_m: float
    max_residual_m: float
    max_normal_error_m: float
    max_segment_m: float

    def __post_init__(self):
        for name, value in self.__dict__.items():
            if (isinstance(value, bool) or not math.isfinite(value)
                    or value < 0 or (name != "front_to_control_center_m" and value == 0)):
                raise ValueError(f"invalid line pose {name}")


@dataclass(frozen=True)
class LinePoseEvidence:
    state: str = "UNAVAILABLE"
    reason: str = "line_pose_unconfigured"
    heading_deg: float | None = None
    heading_error_deg: float | None = None
    normal_offset_m: float | None = None
    normal_error_m: float | None = None
    normal_offset_interval_m: tuple[float, float] | None = None
    lateral_intersection_interval_m: tuple[float, float] | None = None
    normal_direction: int | None = None
    lateral_direction: int | None = None
    along_path_position_observable: bool = False
    mapped_points: int = 0
    mapped_path_complete: bool = False


def estimate_image_line_pose(pixels, *, mapper, config, observation_unique,
                             frame_width, frame_height, pan_pulse_us, tilt_pulse_us):
    points, missing_after_start = [], False
    complete = True
    for pixel in pixels:
        point = mapper.map_image_point(*pixel, frame_width=frame_width, frame_height=frame_height,
            pan_pulse_us=pan_pulse_us, tilt_pulse_us=tilt_pulse_us)
        if point is None:
            complete = False
            missing_after_start |= bool(points)
            continue
        if missing_after_start:
            return LinePoseEvidence(reason="calibrated_path_split")
        point = point[0], point[1]+config.front_to_control_center_m
        if not points or math.dist(points[-1], point) > 1e-6:
            points.append(point)
    if len(points) < 3:
        return LinePoseEvidence(reason="insufficient_calibrated_points", mapped_points=len(points))
    if any(math.dist(a, b) > config.max_segment_m for a, b in zip(points, points[1:])):
        return LinePoseEvidence(reason="path_disconnected", mapped_points=len(points))
    span = math.dist(points[0], points[-1])
    if span < config.min_span_m or span <= 2*config.observation_error_m:
        return LinePoseEvidence(reason="line_support_too_short", mapped_points=len(points))
    array = np.asarray(points)
    centre = array.mean(axis=0)
    _, _, axes = np.linalg.svd(array-centre, full_matrices=False)
    direction = axes[0]
    if direction @ (array[-1]-array[0]) < 0:
        direction = -direction
    normal = np.asarray([direction[1], -direction[0]])
    if float(np.max(np.abs((array-centre) @ normal))) > min(config.max_residual_m, config.observation_error_m):
        return LinePoseEvidence(reason="path_not_straight_within_limit", mapped_points=len(points))
    heading = math.degrees(math.atan2(direction[0], direction[1]))
    endpoint_heading = math.degrees(math.atan2(points[-1][0]-points[0][0], points[-1][1]-points[0][1]))
    angle_error = math.degrees(math.asin(2*config.observation_error_m/span))
    angle_error += abs((heading-endpoint_heading+180) % 360-180)
    angle_error = min(180., angle_error)
    offset = float(normal @ centre)
    error = config.observation_error_m+2*float(np.linalg.norm(centre))*math.sin(math.radians(angle_error)/2)
    interval = offset-error, offset+error
    cos_heading = _trig_interval(heading, angle_error, math.cos)
    lateral = None if cos_heading[0] <= 0 <= cos_heading[1] else _product(
        interval, (1/cos_heading[0], 1/cos_heading[1]))
    reason = "normal_component_observed"
    if not observation_unique:
        reason = "path_identity_unconfirmed"
    elif error > config.max_normal_error_m:
        reason = "normal_error_exceeded"
    normal_direction = (1 if interval[0] > 0 else -1 if interval[1] < 0 else 0) if observation_unique else None
    lateral_direction = ((1 if lateral[0] > 0 else -1 if lateral[1] < 0 else 0)
                         if lateral is not None and observation_unique else None)
    return LinePoseEvidence(state="OBSERVED" if reason == "normal_component_observed" else "UNCERTAIN",
        reason=reason, heading_deg=heading, heading_error_deg=angle_error,
        normal_offset_m=offset, normal_error_m=error, normal_offset_interval_m=interval,
        lateral_intersection_interval_m=lateral, normal_direction=normal_direction,
        lateral_direction=lateral_direction, mapped_points=len(points), mapped_path_complete=complete)
