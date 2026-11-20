"""Read-only corner-relative motion estimates with conditional error bounds.

+X is right, +Y is forward, and IMU yaw is positive left. Error bounds are
provided model assumptions, not confidence inferred from fitting residuals.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import math


@dataclass(frozen=True)
class CornerMotionConfig:
    front_to_control_center_m: float
    observation_error_m: float
    max_position_error_m: float
    max_speed_mps: float
    max_acceleration_mps2: float
    relative_yaw_error_deg: float
    heading_tolerance_deg: float
    velocity_window_s: float
    min_velocity_span_s: float
    prediction_horizon_s: float
    max_imu_skew_s: float
    stop_latency_bounds_s: tuple[float, float] | None = None
    braking_deceleration_bounds_mps2: tuple[float, float] | None = None
    stop_max_yaw_rate_dps: float | None = None

    def __post_init__(self):
        for name in ("front_to_control_center_m", "observation_error_m", "max_position_error_m",
                     "max_speed_mps", "max_acceleration_mps2", "relative_yaw_error_deg",
                     "heading_tolerance_deg", "velocity_window_s", "min_velocity_span_s",
                     "prediction_horizon_s", "max_imu_skew_s"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid corner_motion.{name}")
        if (self.observation_error_m <= 0 or self.max_position_error_m < self.observation_error_m
                or self.max_speed_mps <= 0 or self.max_imu_skew_s <= 0
                or not 0 < self.min_velocity_span_s <= self.velocity_window_s
                or not 0 < self.heading_tolerance_deg < 90
                or not 0 <= self.relative_yaw_error_deg < 90):
            raise ValueError("invalid corner motion limits")
        values = (self.stop_latency_bounds_s, self.braking_deceleration_bounds_mps2,
                  self.stop_max_yaw_rate_dps)
        if any(v is not None for v in values):
            if any(v is None for v in values):
                raise ValueError("all stopping model bounds must be provided together")
            for bounds, positive in ((values[0], False), (values[1], True)):
                if (len(bounds) != 2 or any(isinstance(v, bool) or not math.isfinite(v) for v in bounds)
                        or bounds[0] < 0 or (positive and bounds[0] <= 0) or bounds[1] < bounds[0]):
                    raise ValueError("invalid stopping model interval")
            if (isinstance(values[2], bool) or not math.isfinite(values[2]) or values[2] < 0):
                raise ValueError("invalid stopping model yaw-rate limit")


@dataclass(frozen=True)
class CornerMotionEstimate:
    state: str = "UNAVAILABLE"
    reason: str = "estimator_unconfigured"
    track_id: int | None = None
    timestamp_s: float | None = None
    observed_at_s: float | None = None
    age_s: float | None = None
    corner_position_m: tuple[float, float] | None = None
    incoming_heading_deg: float | None = None
    outgoing_heading_deg: float | None = None
    corner_turn_deg: float | None = None
    position_error_m: float | None = None
    velocity_mps: tuple[float, float] | None = None
    velocity_error_mps: float | None = None
    velocity_fit_residual_m: float | None = None
    velocity_reason: str = "velocity_unavailable"
    observations: int = 0
    stop_reason: str = "stop_model_unconfigured"
    stop_distance_interval_m: tuple[float, float] | None = None
    stopped_corner_forward_interval_m: tuple[float, float] | None = None


@dataclass(frozen=True)
class _Observation:
    timestamp: float
    point: tuple[float, float]
    yaw: float
    incoming_heading: float
    outgoing_heading: float


def _rotate(point, left_yaw_deg):
    angle = math.radians(left_yaw_deg)
    c, s = math.cos(angle), math.sin(angle)
    return c*point[0]+s*point[1], -s*point[0]+c*point[1]


def _angle_difference(a, b):
    return (a-b+180) % 360-180


class CornerMotionTracker:
    def __init__(self, config: CornerMotionConfig):
        self.config = config
        self._track_id = 0
        self.reset()

    def reset(self):
        self._history = deque()
        self._velocity = self._velocity_error = self._residual = None
        self._velocity_reason = "velocity_unavailable"
        self._context = self._epoch = self._last_timestamp = None

    def update(self, geometry, *, timestamp_s, imu, context=()):
        cfg = self.config
        if (not isinstance(timestamp_s, (int, float)) or not math.isfinite(timestamp_s)
                or (self._last_timestamp is not None and timestamp_s <= self._last_timestamp)):
            self.reset()
            return self._with_stopping_projection(CornerMotionEstimate(reason="timestamp_not_increasing"), 0.)
        keys = ("imu_delta_yaw_deg", "imu_yaw_rate_dps", "imu_timestamp_monotonic_s", "imu_age_s")
        if (not imu.get("imu_healthy") or not all(
                isinstance(imu.get(k), (int, float)) and math.isfinite(imu[k]) for k in keys)
                or not 0 <= imu["imu_age_s"] <= cfg.max_imu_skew_s
                or abs(timestamp_s-imu["imu_timestamp_monotonic_s"]) > cfg.max_imu_skew_s):
            self.reset()
            return self._with_stopping_projection(CornerMotionEstimate(reason="imu_unavailable_or_unsynchronized"), 0.)
        epoch = imu.get("imu_recovery_count", 0)
        reason = "corner_observed"
        if self._epoch is not None and epoch != self._epoch:
            self.reset()
            reason = "imu_epoch_changed"
        elif self._context is not None and context != self._context:
            self.reset()
            reason = "observation_context_changed"
        self._context, self._epoch, self._last_timestamp = context, epoch, timestamp_s
        yaw = imu["imu_delta_yaw_deg"]
        valid = (geometry.state == "VALID" and geometry.corner_point_m is not None
                 and len(geometry.corner_point_m) == 2
                 and all(isinstance(v, (int, float)) and math.isfinite(v) for v in
                         (*geometry.corner_point_m, geometry.incoming_heading_deg,
                          geometry.outgoing_heading_deg)))
        if valid:
            point = geometry.corner_point_m[0], geometry.corner_point_m[1]+cfg.front_to_control_center_m
            observation = _Observation(timestamp_s, point, yaw,
                geometry.incoming_heading_deg, geometry.outgoing_heading_deg)
            if self._history:
                last = self._history[-1]
                age = timestamp_s-last.timestamp
                rotated = _rotate(last.point, yaw-last.yaw)
                rotation_error = 2*math.hypot(*last.point)*math.sin(math.radians(cfg.relative_yaw_error_deg)/2)
                same_heading = all(abs(_angle_difference(new, old+yaw-last.yaw)) <= cfg.heading_tolerance_deg
                                   for new, old in ((observation.incoming_heading, last.incoming_heading),
                                                    (observation.outgoing_heading, last.outgoing_heading)))
                if (age > cfg.prediction_horizon_s or not same_heading
                        or math.dist(rotated, point) > cfg.max_speed_mps*age+2*cfg.observation_error_m+rotation_error):
                    self._history.clear()
                    reason = "corner_association_changed"
            if not self._history:
                self._track_id += 1
            self._history.append(observation)
            while timestamp_s-self._history[0].timestamp > cfg.velocity_window_s:
                self._history.popleft()
            self._velocity_reason = self._fit_velocity()
            result = CornerMotionEstimate(
                state="TRACKING" if self._velocity is not None else "OBSERVED", reason=reason,
                track_id=self._track_id, timestamp_s=timestamp_s, observed_at_s=timestamp_s, age_s=0.,
                corner_position_m=point, position_error_m=cfg.observation_error_m,
                incoming_heading_deg=observation.incoming_heading,
                outgoing_heading_deg=observation.outgoing_heading,
                corner_turn_deg=_angle_difference(observation.outgoing_heading, observation.incoming_heading),
                velocity_mps=self._velocity, velocity_error_mps=self._velocity_error,
                velocity_fit_residual_m=self._residual, velocity_reason=self._velocity_reason,
                observations=len(self._history))
        else:
            if not self._history:
                return self._with_stopping_projection(CornerMotionEstimate(reason=reason if reason != "corner_observed" else
                                            "corner_observation_unavailable", timestamp_s=timestamp_s), 0.)
            last = self._history[-1]
            age = timestamp_s-last.timestamp
            if age > cfg.prediction_horizon_s:
                self._history.clear()
                self._velocity = self._velocity_error = self._residual = None
                return self._with_stopping_projection(CornerMotionEstimate(reason="prediction_expired", timestamp_s=timestamp_s), 0.)
            delta = yaw-last.yaw
            point = _rotate(last.point, delta)
            error = cfg.observation_error_m+2*math.hypot(*last.point)*math.sin(math.radians(cfg.relative_yaw_error_deg)/2)
            velocity = None if self._velocity is None else _rotate(self._velocity, delta)
            velocity_error = None
            if velocity is None:
                error += cfg.max_speed_mps*age
            else:
                rotation_speed_error = 2*math.hypot(*velocity)*math.sin(math.radians(cfg.relative_yaw_error_deg)/2)
                velocity_error = self._velocity_error+rotation_speed_error+cfg.max_acceleration_mps2*age
                point = point[0]-velocity[0]*age, point[1]-velocity[1]*age
                error += (self._velocity_error+rotation_speed_error)*age+.5*cfg.max_acceleration_mps2*age**2
            usable = error <= cfg.max_position_error_m
            result = CornerMotionEstimate(
                state="PREDICTED" if usable else "UNCERTAIN",
                reason="bounded_prediction" if usable else "position_error_exceeded",
                track_id=self._track_id, timestamp_s=timestamp_s, observed_at_s=last.timestamp, age_s=age,
                corner_position_m=point, position_error_m=error, velocity_mps=velocity,
                incoming_heading_deg=_angle_difference(last.incoming_heading+delta, 0.),
                outgoing_heading_deg=_angle_difference(last.outgoing_heading+delta, 0.),
                corner_turn_deg=_angle_difference(last.outgoing_heading, last.incoming_heading),
                velocity_error_mps=velocity_error, velocity_fit_residual_m=self._residual,
                velocity_reason=self._velocity_reason,
                observations=len(self._history))
        return self._with_stopping_projection(result, imu["imu_yaw_rate_dps"])

    def _fit_velocity(self):
        cfg = self.config
        self._velocity = self._velocity_error = self._residual = None
        last = self._history[-1]
        if len(self._history) < 3:
            return "not_enough_observations"
        if last.timestamp-self._history[0].timestamp < cfg.min_velocity_span_s:
            return "observation_span_too_short"
        times = [s.timestamp-last.timestamp for s in self._history]
        points = [_rotate(s.point, last.yaw-s.yaw) for s in self._history]
        errors = [cfg.observation_error_m+(0.0 if s is last else
                  2*math.hypot(*s.point)*math.sin(math.radians(cfg.relative_yaw_error_deg)/2))
                  for s in self._history]
        mean_t = sum(times)/len(times)
        centred = [t-mean_t for t in times]
        denominator = sum(t*t for t in centred)
        slope = tuple(sum(t*p[axis] for t, p in zip(centred, points))/denominator for axis in (0, 1))
        means = tuple(sum(p[axis] for p in points)/len(points) for axis in (0, 1))
        residual = max(math.dist(p, (means[0]+slope[0]*t, means[1]+slope[1]*t))
                       for t, p in zip(centred, points))
        # L1 weights preserve a worst-case observation bound. More samples
        # or a zero fitting residual do not establish independent accuracy.
        error = sum(abs(t)*e for t, e in zip(centred, errors))/denominator
        # Regression is a positive weighted average of past velocities;
        # bounded acceleration contributes its mean age, not sample noise.
        error += .5*cfg.max_acceleration_mps2*abs(sum(t*raw_t**2
                    for t, raw_t in zip(centred, times)))/denominator
        if residual > 2*max(errors)+.5*cfg.max_acceleration_mps2*times[0]**2:
            return "motion_fit_inconsistent"
        if math.hypot(*slope)-error > cfg.max_speed_mps:
            return "speed_bound_exceeded"
        self._velocity = -slope[0], -slope[1]
        self._velocity_error, self._residual = error, residual
        return "bounded_velocity_fit"

    def _with_stopping_projection(self, result, yaw_rate):
        cfg = self.config
        if cfg.stop_latency_bounds_s is None:
            return result
        if result.state == "UNCERTAIN" or result.velocity_mps is None:
            return replace(result, stop_reason="motion_estimate_unavailable")
        if abs(yaw_rate) > cfg.stop_max_yaw_rate_dps:
            return replace(result, stop_reason="straight_stop_model_not_applicable")
        velocity = result.velocity_mps[1]
        lower, upper = velocity-result.velocity_error_mps, velocity+result.velocity_error_mps
        if lower < 0:
            return replace(result, stop_reason="forward_velocity_not_resolved")
        latency_min, latency_max = cfg.stop_latency_bounds_s
        decel_min, decel_max = cfg.braking_deceleration_bounds_mps2
        acceleration = cfg.max_acceleration_mps2
        lower_distances = []
        for latency in (latency_min, latency_max):
            moving_time = min(latency, lower/acceleration) if acceleration else latency
            distance = lower*moving_time-.5*acceleration*moving_time**2
            distance += max(0., lower-acceleration*latency)**2/(2*decel_max)
            lower_distances.append(distance)
        lower_distance = min(lower_distances)
        upper_distance = upper*latency_max+.5*acceleration*latency_max**2
        upper_distance += (upper+acceleration*latency_max)**2/(2*decel_min)
        y = result.corner_position_m[1]
        error = result.position_error_m
        return replace(result, stop_reason="conditional_straight_stop_projection",
            stop_distance_interval_m=(lower_distance, upper_distance),
            stopped_corner_forward_interval_m=(y-error-upper_distance, y+error-lower_distance))
