"""Connected L-tape preview and a bounded approach/turn/reacquire controller.

Image angles are not metric ground angles. IMU measures actual relative yaw;
the 90 degree nominal turn is always subject to fresh visual reacquisition.
No device is opened and no motor is commanded by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class RightAngleConfig:
    enabled: bool = False
    confirm_frames: int = 3
    min_confirm_y_norm: float = 0.48
    trigger_y_norm: float = 0.78
    approach_scale: float = 0.60
    advance_duration_s: float = 0.0
    sensor_tail_duration_s: float = 0.0
    sensor_clear_frames: int = 2
    left_sensor_anchor_min_channels: int = 2
    left_two_channel_tail_duration_s: float = 0.0
    left_three_channel_tail_duration_s: float | None = None
    exit_max_heading_deg: float = 30.0
    exit_max_center_offset_norm: float = 0.20
    approach_lookahead_s: float = 0.06
    max_approach_lead_norm: float = 0.08
    pivot_duty: float = 12.0
    nominal_angle_deg: float = 90.0
    stop_lead_deg: float = 0.0
    stop_response_s: float = 0.0
    min_reacquire_angle_deg: float = 50.0
    max_angle_deg: float = 105.0
    max_rate_dps: float = 80.0
    approach_timeout_s: float = 5.0
    turn_timeout_s: float = 4.0
    evidence_timeout_s: float = 0.25
    reacquire_frames: int = 3
    cooldown_s: float = 1.5

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise ValueError("right_angle.enabled must be boolean")
        for key in ("confirm_frames", "reacquire_frames", "sensor_clear_frames"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or not 2 <= value <= 10:
                raise ValueError(f"invalid right_angle.{key}")
        if self.left_sensor_anchor_min_channels not in (2, 3):
            raise ValueError("invalid right_angle.left_sensor_anchor_min_channels")
        bounds = {"trigger_y_norm": (.60, .85), "approach_scale": (.2, .75), "pivot_duty": (1, 20),
            "nominal_angle_deg": (75, 95), "min_reacquire_angle_deg": (35, 75),
            "stop_lead_deg": (0, 15),
            "stop_response_s": (0, 1),
            "max_angle_deg": (95, 120), "max_rate_dps": (10, 100), "approach_timeout_s": (.5, 10),
            "turn_timeout_s": (.5, 6), "evidence_timeout_s": (.05, .3), "cooldown_s": (.25, 5),
            "advance_duration_s": (0, 1.5), "min_confirm_y_norm": (.3, .65),
            "sensor_tail_duration_s": (0, .8),
            "left_two_channel_tail_duration_s": (0, .8),
            "exit_max_heading_deg": (5, 45), "exit_max_center_offset_norm": (.05, .3),
            "approach_lookahead_s": (0, .1), "max_approach_lead_norm": (0, .1)}
        for key, (low, high) in bounds.items():
            value = getattr(self, key)
            if isinstance(value, bool) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"invalid right_angle.{key}")
        if self.left_three_channel_tail_duration_s is not None and (
            isinstance(self.left_three_channel_tail_duration_s, bool)
            or not math.isfinite(self.left_three_channel_tail_duration_s)
            or not 0 <= self.left_three_channel_tail_duration_s <= .8
        ):
            raise ValueError("invalid right_angle.left_three_channel_tail_duration_s")
        if not self.min_reacquire_angle_deg < self.nominal_angle_deg < self.max_angle_deg:
            raise ValueError("right angle limits must be ordered")


@dataclass(frozen=True)
class CornerObservation:
    direction: int = 0  # +1 left, -1 right, 0 no single L.
    x_norm: float = 0.0
    y_norm: float = 0.0
    incoming_offset: float = 0.0
    incoming_heading_deg: float = 0.0
    confidence: float = 0.0
    reason: str = "No connected L-shaped tape"
    path_points_px: tuple[tuple[float, float], ...] = ()


def _runs(values):
    indices = np.flatnonzero(np.diff(np.r_[False, values != 0, False]))
    return list(zip(indices[::2], indices[1::2]))


def _arm_gap_limit(tape_width, image_width):
    # A glare break may span part of the projected tape width, not an arbitrary
    # distance across the floor. Keep the image-wide cap small as well.
    return max(3, round(min(tape_width * .75, image_width * .05)))


def detect_right_angle(frame, vision, *, tracking_corner=False) -> CornerObservation:
    """Require a bottom-connected narrow stem, one thin arm, no forward exit.

    T/cross junctions and detached transverse marks do not select a direction.
    This uses the existing calibrated ground ROI and black threshold, while
    handling the transverse arm which cannot be represented as x(y).
    """
    if not vision.calibration_ready:
        return CornerObservation(reason="Ground calibration required")
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = vision.blur_ksize | 1
    if blur >= 3:
        gray = cv2.GaussianBlur(gray, (blur, blur), 0)
    mask = cv2.threshold(gray, vision.threshold, 255, cv2.THRESH_BINARY_INV)[1]
    polygon = np.asarray([[round(x*(width-1)), round(y*(height-1))]
                          for x, y in vision.ground_roi_norm], dtype=np.int32)
    ground = np.zeros_like(mask)
    cv2.fillPoly(ground, [polygon], 255)
    mask = cv2.bitwise_and(mask, ground)
    if vision.morph_ksize >= 3:
        kernel = np.ones((vision.morph_ksize, vision.morph_ksize), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    bottom_limit = int(polygon[:, 1].max()) - max(2, round(height*.01))
    seed_rows = []
    for y in range(bottom_limit, bottom_limit-max(3, round(height*.04)), -max(2, round(height*.01))):
        expected = vision.expected_tape_width_px(y/(height-1), width)
        if expected is None:
            continue
        seeds = [(a, b) for a, b in _runs(mask[y])
                 if expected*.5 <= b-a <= expected*1.6 and abs((a+b)/2-width/2) < width*.35]
        if len(seeds) == 1:
            a, b = seeds[0]
            seed_rows.append((abs((b-a)/expected-1), -y, y, a, b))
    if not seed_rows:
        return CornerObservation(reason="No unique calibrated near stem")
    _, _, bottom, a, b = min(seed_rows)
    _, labels = cv2.connectedComponents(mask, connectivity=8)
    initial_x = (a+b)/2
    label = labels[bottom, int(initial_x)]
    component = np.where(labels == label, 255, 0).astype(np.uint8)
    stem = []
    center = initial_x
    for y in range(bottom, int(polygon[:, 1].min()), -max(2, round(height*.0125))):
        expected = vision.expected_tape_width_px(y/(height-1), width)
        if expected is None:
            break
        runs = [(a, b) for a, b in _runs(component[y])
                if expected*.5 <= b-a <= expected*1.6 and abs((a+b)/2-center) < width*.04]
        if len(runs) != 1:
            break
        center = sum(runs[0])/2
        stem.append((center, y))
    minimum_samples = 4 if tracking_corner else 6
    minimum_span = .035 if tracking_corner else .065
    if len(stem) < minimum_samples or stem[0][1]-stem[-1][1] < height*minimum_span:
        return CornerObservation(reason="Insufficient incoming leg")
    stem = np.asarray(stem)
    slope, intercept = np.polyfit(stem[:, 1], stem[:, 0], 1)
    if abs(slope) > .55 or np.sqrt(np.mean((np.polyval((slope, intercept), stem[:, 1])-stem[:, 0])**2)) > width*.012:
        return CornerObservation(reason="Incoming leg not established")
    # Look across bounded reflection breaks in the full mask. Candidates still
    # need a shoulder anchored to the near stem's original connected component.
    arm_width = vision.expected_tape_width_px(stem[-1, 1]/(height-1), width)
    segments = cv2.HoughLinesP(cv2.Canny(mask, 60, 160), 1, np.pi/180, 25,
                              minLineLength=max(25, int(width*.14)),
                              maxLineGap=_arm_gap_limit(arm_width, width))
    found = []
    if segments is None:
        return CornerObservation()
    for x0, y0, x1, y1 in segments.reshape(-1, 4):
        if abs(x1-x0) < width*.14:
            continue
        arm_slope = (y1-y0)/(x1-x0)
        if abs(arm_slope) > .65:
            continue
        arm_intercept = y0-arm_slope*x0
        cy = (arm_slope*intercept+arm_intercept)/(1-arm_slope*slope)
        cx = slope*cy+intercept
        if not polygon[:, 1].min()+height*.035 < cy < bottom-height*.055:
            continue
        if abs(cy-stem[-1, 1]) > max(height*.10, arm_width*.75):
            continue
        tape_width = vision.expected_tape_width_px(cy/(height-1), width)
        if tape_width is None:
            continue
        side = 1 if max(x0, x1)-cx > cx-min(x0, x1) else -1  # image right/left
        endpoint = max(x0, x1) if side > 0 else min(x0, x1)
        length = side*(endpoint-cx)
        if length < max(width*.18, tape_width*2):
            continue
        samples = []
        for sample_index, x in enumerate(np.linspace(cx+side*tape_width*1.1, endpoint-side*width*.015, 7)):
            xi = int(round(x))
            if not 0 <= xi < width:
                break
            guess = arm_slope*x+arm_intercept
            runs = [(a, b) for a, b in _runs(mask[:, xi])
                    if a-3 <= guess <= b+3 and 3 <= b-a <= max(height*.13, tape_width*1.35)]
            if sample_index < 2 and (len(runs) != 1 or not component[round(sum(runs[0])/2), xi]):
                break
            if len(runs) == 1:
                samples.append((x, sum(runs[0])/2, runs[0][1]-runs[0][0]))
        if len(samples) < 6:
            continue
        samples = np.asarray(samples)
        arm_slope, arm_intercept = np.polyfit(samples[:, 0], samples[:, 1], 1)
        thickness = float(np.median(samples[:, 2]))
        if np.sqrt(np.mean((np.polyval((arm_slope, arm_intercept), samples[:, 0])-samples[:, 1])**2)) > height*.012:
            continue
        cy = (arm_slope*intercept+arm_intercept)/(1-arm_slope*slope)
        cx = slope*cy+intercept
        arm_x = np.arange(round(cx+side*tape_width*1.1), round(endpoint)+side, side)
        arm_y = np.rint(arm_slope*arm_x+arm_intercept).astype(int)
        if not len(arm_x) or np.any((arm_x < 0) | (arm_x >= width) | (arm_y < 0) | (arm_y >= height)):
            continue
        # Reflections can erase the centre while tape edges remain visible.
        band_rows = np.clip(arm_y[:, None] + np.arange(
            -round(thickness/2), round(thickness/2)+1), 0, height-1)
        support = np.any(mask[band_rows, arm_x[:, None]] != 0, axis=1)
        gaps = _runs(~support)
        if np.mean(support) < .8 or any(b-a > _arm_gap_limit(tape_width, width) for a, b in gaps):
            continue
        # A forward continuation or a second side exit is a junction, not L.
        forward = []
        for y in np.linspace(cy-max(2*thickness, height*.10), cy-max(1.3*thickness, height*.06), 5):
            yi, xi = round(y), round(slope*y+intercept)
            if 0 <= yi < height and 1 <= xi < width-1:
                forward.append(np.any(mask[yi, xi-1:xi+2]))
        opposite = []
        for distance in np.linspace(tape_width*1.4, max(tape_width*2.2, width*.18), 5):
            x = cx-side*distance
            y = arm_slope*x+arm_intercept
            if 0 <= round(x) < width and 0 <= round(y) < height:
                opposite.append(bool(mask[round(y), round(x)]))
        if not forward or any(forward) or sum(opposite) >= 2:
            continue
        stem_points = tuple((float(x), float(y)) for x, y in stem[::max(1, len(stem)//3)])
        exit_points = tuple((float(x), float(arm_slope*x+arm_intercept))
                            for x in np.linspace(cx+side*tape_width, endpoint, 4))
        found.append(CornerObservation(-side, cx/width, cy/height,
            (slope*(height*.80)+intercept-width/2)/(width/2),
            math.degrees(math.atan(-slope)), .85, "Connected incoming leg and single transverse exit",
            (*stem_points, (float(cx), float(cy)), *exit_points)))
    directions = {ob.direction for ob in found}
    if len(directions) != 1:
        return CornerObservation(reason="No unambiguous single L exit")
    return found[0]


@dataclass(frozen=True)
class CornerCommand:
    state: str = "IDLE"
    active: bool = False
    failed: bool = False
    direction: int = 0
    forward_scale: float = 0.0
    turn_input: float = 0.0
    pivot_duty: float = 0.0
    progress_deg: float | None = None
    reason: str = ""


class RightAngleController:
    def __init__(self, config: RightAngleConfig):
        self.config = config
        self.reset()

    def reset(self):
        self.state = "IDLE"
        self._direction = self._count = self._reacquire = 0
        self._since = self._last_seen = self._cooldown_until = 0.0
        self._yaw_start = 0.0
        self._stop_since = 0.0
        self._reason = ""
        self._recovery_count = 0
        self._last_corner_y = None
        self._approach_rate = 0.0
        self._advance_until = 0.0
        self._sensor_seen = self._sensor_anchor_used = False
        self._sensor_clear_count = 0
        self._sensor_broad_count = 0
        self._sensor_two_channel_anchor = False
        self._last_pivot_rate_dps = None
        self._confirm_hold = False

    def update(self, observation, *, now, dt, armed, imu, line_aligned,
               line_sensor_channels=None, ground_corner=None, exit_path_available=False):
        cfg = self.config
        if not cfg.enabled:
            self.reset()
            return CornerCommand()
        if not armed:
            if self.state == "FAILED":
                return CornerCommand("FAILED", direction=self._direction, reason=self._reason)
            self.reset()
            return CornerCommand()
        if self.state == "FAILED":
            return CornerCommand("FAILED", failed=True, direction=self._direction, reason=self._reason)
        if not imu.get("imu_healthy") or not all(isinstance(imu.get(key), (int, float)) and math.isfinite(imu[key])
                for key in ("imu_delta_yaw_deg", "imu_yaw_rate_dps")):
            if self.state in {"TURN", "SETTLE_TURN"}:
                return self._fail("IMU angle unavailable during corner pivot")
            self.reset()
            return CornerCommand()
        if self.state == "COOLDOWN":
            if now < self._cooldown_until and not (observation.direction and observation.path_points_px):
                return CornerCommand("COOLDOWN")
            self.reset()
        if self.state in {"IDLE", "CONFIRM"}:
            metric_confirmed = (
                ground_corner is not None and ground_corner.state == "VALID"
                and ground_corner.turn_deg is not None
                and math.isfinite(ground_corner.turn_deg)
                and ground_corner.turn_deg*observation.direction < 0)
            if observation.direction == 0 or (
                observation.y_norm < cfg.min_confirm_y_norm and not metric_confirmed
            ):
                if self.state == "CONFIRM" and now-self._last_seen <= cfg.evidence_timeout_s:
                    self._count = 0
                    return CornerCommand("CONFIRM", active=self._confirm_hold, direction=self._direction,
                                         reason="Holding for fresh corner confirmation")
                self._count = 0
                self.state = "IDLE"
                return CornerCommand()
            self._count = self._count+1 if self._direction == observation.direction else 1
            self._direction = observation.direction
            self.state = "CONFIRM"
            self._last_seen = now
            self._confirm_hold = bool(observation.path_points_px or observation.confidence >= .8)
            if self._count < cfg.confirm_frames:
                return CornerCommand("CONFIRM", active=self._confirm_hold, direction=self._direction,
                                     reason="Holding for connected corner confirmation")
            self.state = "APPROACH"
            self._since = self._last_seen = now
            self._recovery_count = imu.get("imu_recovery_count", 0)
        if not 0 < dt <= .2 or imu.get("imu_recovery_count", 0) != self._recovery_count:
            return self._fail("IMU epoch or control interval changed")
        if abs(imu["imu_yaw_rate_dps"]) > cfg.max_rate_dps:
            if self.state in {"ADVANCE", "SETTLE"}:
                self.reset()
                return CornerCommand()
            return self._fail("Corner yaw rate exceeded limit")
        if self.state == "APPROACH":
            if now-self._since > cfg.approach_timeout_s:
                return self._fail("Corner approach timed out")
            if observation.direction != self._direction:
                self._last_corner_y = None
                self._approach_rate = 0.0
                # Never advance using a stale corner location.
                if now-self._last_seen > cfg.evidence_timeout_s:
                    return self._fail("Corner evidence lost before turn")
                return CornerCommand("WAIT_EVIDENCE", active=True, direction=self._direction)
            elapsed = now-self._last_seen
            rate = 0.0 if self._last_corner_y is None or elapsed <= 0 else (observation.y_norm-self._last_corner_y)/elapsed
            # Use two consecutive positive approach rates and at most one frame
            # of lead. A static corner, one jump or missing evidence cannot turn.
            lead = min(max(0.0, rate), max(0.0, self._approach_rate)) * min(dt, cfg.approach_lookahead_s)
            lead = min(lead, cfg.max_approach_lead_norm)
            self._last_seen, self._last_corner_y, self._approach_rate = now, observation.y_norm, rate
            if observation.y_norm+lead < cfg.trigger_y_norm:
                turn = max(-.5, min(.5, -(observation.incoming_offset*1.5 + math.radians(observation.incoming_heading_deg))))
                return CornerCommand("APPROACH", True, direction=self._direction,
                                     forward_scale=cfg.approach_scale, turn_input=turn)
            self._reason = "Fresh corner predicts turn point within one frame" if observation.y_norm < cfg.trigger_y_norm else "Fresh corner reached turn point"
            self._since, self._yaw_start = now, imu["imu_delta_yaw_deg"]
            if cfg.advance_duration_s:
                self.state = "ADVANCE"
                self._advance_until = now + cfg.advance_duration_s
                self._sensor_seen = bool(line_sensor_channels and sum(line_sensor_channels) >= 3)
                self._sensor_anchor_used = False
                self._sensor_clear_count = 0
                self._sensor_broad_count = 0
                self._sensor_two_channel_anchor = False
                return CornerCommand("ADVANCE", True, direction=self._direction,
                                     forward_scale=cfg.approach_scale, reason=self._reason)
            self.state = "TURN"
        if self.state == "ADVANCE":
            if cfg.sensor_tail_duration_s and not self._sensor_anchor_used and now < self._advance_until:
                if line_sensor_channels is None:
                    self._sensor_clear_count = 0
                elif any(line_sensor_channels):
                    active_channels = sum(line_sensor_channels)
                    if active_channels >= 3:
                        self._sensor_seen = True
                        self._sensor_two_channel_anchor = False
                    elif self._direction == 1 and active_channels >= cfg.left_sensor_anchor_min_channels:
                        self._sensor_broad_count += 1
                        if self._sensor_broad_count >= 2:
                            self._sensor_seen = True
                            self._sensor_two_channel_anchor = True
                    else:
                        self._sensor_broad_count = 0
                    self._sensor_clear_count = 0
                elif self._sensor_seen:
                    self._sensor_clear_count += 1
                    if self._sensor_clear_count >= cfg.sensor_clear_frames:
                        if self._direction == 1:
                            tail = (cfg.left_two_channel_tail_duration_s if self._sensor_two_channel_anchor
                                    else cfg.left_three_channel_tail_duration_s if cfg.left_three_channel_tail_duration_s is not None
                                    else cfg.sensor_tail_duration_s)
                        else:
                            tail = cfg.sensor_tail_duration_s
                        self._advance_until = min(now + tail, self._since + 1.5)
                        self._sensor_anchor_used = True
                        self._reason = "Line sensor cleared; bounded tail advance"
            if now < self._advance_until:
                return CornerCommand("ADVANCE", True, direction=self._direction,
                                     forward_scale=cfg.approach_scale, reason=self._reason)
            self.state, self._since = "SETTLE", now
        if self.state == "SETTLE":
            if now-self._since < .2:
                return CornerCommand("SETTLE", True, direction=self._direction,
                                     reason="Stopping after corner alignment advance")
            self.state, self._since, self._yaw_start = "TURN", now, imu["imu_delta_yaw_deg"]
        progress = self._direction*(imu["imu_delta_yaw_deg"]-self._yaw_start)
        if now-self._since > cfg.turn_timeout_s or progress > cfg.max_angle_deg or progress < -8:
            return self._fail("Corner turn exceeded time/angle/direction limit", progress)
        settled_exit = (self.state == "SETTLE_TURN" and exit_path_available
                        and abs(imu["imu_yaw_rate_dps"]) <= cfg.max_rate_dps*.1)
        if progress >= cfg.min_reacquire_angle_deg and (line_aligned or settled_exit):
            self._reacquire += 1
        else:
            self._reacquire = 0
        if self._reacquire >= cfg.reacquire_frames:
            self.state = "COOLDOWN"
            self._cooldown_until = now+cfg.cooldown_s
            reason = ("Continuous exit path acquired after pivot settled; tracking will correct offset"
                      if settled_exit and not line_aligned else "New tape path reacquired")
            return CornerCommand("COMPLETE", progress_deg=progress, reason=reason)
        if self.state == "SETTLE_TURN":
            if now-self._stop_since >= .8:
                return self._fail("Corner stopped without aligned exit path", progress)
            return CornerCommand("SETTLE_TURN", True, direction=self._direction, progress_deg=progress)
        stopping_lead = max(cfg.stop_lead_deg,
                            max(0.0, self._direction*imu["imu_yaw_rate_dps"])*cfg.stop_response_s)
        if progress >= cfg.nominal_angle_deg-stopping_lead:
            self.state, self._stop_since = "SETTLE_TURN", now
            return CornerCommand("SETTLE_TURN", True, direction=self._direction, progress_deg=progress)
        duty = cfg.pivot_duty*max(.45, min(1.0, (cfg.nominal_angle_deg-progress)/35))
        rate = max(0.0, self._direction*imu["imu_yaw_rate_dps"])
        rate_rise = (0.0 if self._last_pivot_rate_dps is None
                     else max(0.0, rate-self._last_pivot_rate_dps))
        self._last_pivot_rate_dps = rate
        # Leave room for one control frame of continued acceleration before
        # the independent hard rate limit is reached.
        projected_rate = rate+rate_rise
        target_rate = cfg.max_rate_dps*.6
        if projected_rate > target_rate:
            duty *= target_rate/projected_rate
        return CornerCommand("TURN", True, direction=self._direction,
                             pivot_duty=self._direction*duty, progress_deg=progress, reason=self._reason)

    def _fail(self, reason, progress=None):
        self.state, self._reason = "FAILED", reason
        return CornerCommand("FAILED", failed=True, direction=self._direction, progress_deg=progress, reason=reason)
