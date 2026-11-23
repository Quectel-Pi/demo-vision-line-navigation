from dataclasses import dataclass
import math

import cv2
import numpy as np

from .right_angle import _runs


@dataclass(frozen=True)
class StopMarkerObservation:
    detected: bool = False
    forward_m: float | None = None
    lateral_m: float | None = None
    crossbar_m: float | None = None
    thickness_m: float | None = None
    length_source: str = ''


def detect_stop_marker(frame, vision, mapper, *, pan_pulse_us, tilt_pulse_us,
                       crossbar_m=.06, tolerance_m=.015):
    if not vision.calibration_ready or not mapper.available:
        return StopMarkerObservation()
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (vision.blur_ksize | 1, vision.blur_ksize | 1), 0)
    mask = cv2.threshold(gray, vision.threshold, 255, cv2.THRESH_BINARY_INV)[1]
    polygon = np.asarray([[round(x*(width-1)), round(y*(height-1))]
                          for x, y in vision.ground_roi_norm], dtype=np.int32)
    ground = np.zeros_like(mask)
    cv2.fillPoly(ground, [polygon], 255)
    mask = cv2.bitwise_and(mask, ground)
    bottom = int(polygon[:, 1].max())-max(2, round(height*.01))
    expected = vision.expected_tape_width_px(bottom/(height-1), width)
    seeds = [(a, b) for a, b in _runs(mask[bottom])
             if expected*.5 <= b-a <= expected*1.6 and abs((a+b)/2-width/2) < width*.35]
    if len(seeds) != 1:
        return StopMarkerObservation()
    _, labels = cv2.connectedComponents(mask, connectivity=8)
    label = labels[bottom, round(sum(seeds[0])/2)]
    component = np.where(labels == label, 255, 0).astype(np.uint8)
    context = dict(frame_width=width, frame_height=height,
                   pan_pulse_us=pan_pulse_us, tilt_pulse_us=tilt_pulse_us)
    candidates = []
    for y in range(int(polygon[:, 1].min()), bottom, max(2, round(height*.01))):
        expected = vision.expected_tape_width_px(y/(height-1), width)
        for a, b in _runs(component[y]):
            if b-a < 2.2*expected:
                continue
            arm_runs = []
            for x in (round(a+(b-a)*.2), round(a+(b-a)*.8)):
                runs = [(top, end) for top, end in _runs(component[:, x]) if top <= y < end]
                if len(runs) != 1:
                    break
                arm_runs.append(runs[0])
            if len(arm_runs) != 2:
                continue
            top = round(sum(r[0] for r in arm_runs)/2)
            end = round(sum(r[1] for r in arm_runs)/2)
            cy = (top+end)/2
            if abs(arm_runs[0][0]-arm_runs[1][0]) > max(3, (end-top)*.5):
                continue
            stem = []
            stem_widths = []
            stem_rows = []
            for sy in (end+round(height*.025*i) for i in (1, 2, 3)):
                if sy >= bottom:
                    break
                ew = vision.expected_tape_width_px(sy/(height-1), width)
                runs = [(l, r) for l, r in _runs(component[sy]) if ew*.5 <= r-l <= ew*1.6]
                if len(runs) != 1:
                    break
                stem.append(sum(runs[0])/2)
                stem_widths.append(runs[0][1]-runs[0][0])
                stem_rows.append(sy)
            if len(stem) != 3 or max(stem)-min(stem) > expected*.5:
                continue
            cx = float(np.polyval(np.polyfit(stem_rows, stem, 1), cy))
            junction_width = float(np.polyval(np.polyfit(stem_rows, stem_widths, 1), cy))
            if not expected*.5 <= junction_width <= expected*2:
                continue
            if min(cx-a, b-cx) < expected*1.1:
                continue
            fy = top-max(4, round(height*.025))
            if fy < polygon[:, 1].min() or np.any(mask[fy, max(0, round(cx-expected*.3)):min(width, round(cx+expected*.3))]):
                continue
            mapped = [mapper.map_image_point(x, yy, **context) for x, yy in (
                (a, cy), (b-1, cy), (cx, cy), (a+(b-a)*.2, top), (a+(b-a)*.2, end-1))]
            if any(point is None for point in mapped[2:]):
                continue
            if mapped[0] is not None and mapped[1] is not None:
                length = math.dist(mapped[0], mapped[1])
                length_source = 'GROUND_HOMOGRAPHY'
            else:
                # Extend the observed straight stem edges to the junction row;
                # compare widths there without extrapolating the ground map.
                length = (b-a-1)/junction_width*vision.tape_width_mm/1000
                length_source = 'LOCAL_TAPE_RATIO'
            thickness = math.dist(mapped[3], mapped[4])
            if abs(length-crossbar_m) <= tolerance_m and .010 <= thickness <= .030:
                candidates.append(StopMarkerObservation(True, mapped[2][1], mapped[2][0], length, thickness, length_source))
    return min(candidates, key=lambda ob: abs(ob.crossbar_m-crossbar_m)) if candidates else StopMarkerObservation()


@dataclass(frozen=True)
class StopMarkerCommand:
    state: str = 'IDLE'
    active: bool = False
    done: bool = False
    failed: bool = False
    duty: float = 0.
    reason: str = ''


class StopMarkerController:
    def __init__(self, config):
        if config.get('enabled', False):
            for key, low, high in (
                ('crossbar_m', .04, .10), ('crossbar_tolerance_m', .001, .02),
                ('sensor_forward_m', .01, .20), ('advance_speed_mps', .02, .30),
                ('approach_duty', 1., 10.),
            ):
                value = config[key]
                if isinstance(value, bool) or not math.isfinite(value) or not low <= value <= high:
                    raise ValueError(f'invalid stop_marker.{key}')
        self.config = config
        self.reset()

    def reset(self):
        self.state = 'IDLE'
        self._count = self._black_count = self._clear_count = 0
        self._last_seen = self._started = self._clear_since = self._deadline = 0.
        self._forward = None
        self._yaw_start = 0.
        self._recovery = 0
        self._remaining_after_sensor_m = 0.

    def update(self, observation, *, now, dt, armed, imu, channels):
        cfg = self.config
        if not cfg.get('enabled', False) or not armed:
            self.reset()
            return StopMarkerCommand()
        if self.state == 'IDLE' and not observation.detected:
            return StopMarkerCommand()
        if self.state in {'DONE', 'FAILED'}:
            return StopMarkerCommand(self.state, done=self.state == 'DONE', failed=self.state == 'FAILED')
        if not imu.get('imu_healthy') or not 0 < dt <= .2:
            return self._fail('Marker positioning requires fresh IMU and control frames')
        if self.state in {'IDLE', 'CONFIRM'}:
            if not observation.detected:
                self.reset()
                return StopMarkerCommand()
            self._count += 1
            self.state = 'CONFIRM'
            if self._count < 3:
                return StopMarkerCommand('CONFIRM', active=True)
            self.state = 'APPROACH'
            self._started = now
            self._yaw_start = imu['imu_delta_yaw_deg']
            self._recovery = imu.get('imu_recovery_count', 0)
            self._remaining_after_sensor_m = max(0., cfg['sensor_forward_m']-observation.thickness_m/2)
        if imu.get('imu_recovery_count', 0) != self._recovery or abs(imu['imu_delta_yaw_deg']-self._yaw_start) > 15:
            return self._fail('Marker approach changed heading or IMU epoch')
        if now-self._started > 3.:
            return self._fail('Marker positioning timed out')
        if observation.detected:
            self._last_seen, self._forward = now, observation.forward_m
        if self.state == 'APPROACH':
            if channels is None or len(channels) != 4:
                return self._fail('Marker positioning requires line sensors')
            middle_black = bool(channels[1] or channels[2])
            if middle_black:
                self._black_count += 1
                self._clear_count = 0
            elif self._black_count >= 2:
                if not self._clear_count:
                    self._clear_since = now
                self._clear_count += 1
                if self._clear_count >= 2 and self._forward is not None and self._forward <= .10:
                    self._deadline = self._clear_since + self._remaining_after_sensor_m/cfg['advance_speed_mps']
                    self.state = 'ADVANCE'
            if now-self._last_seen > .25 and (self._forward is None or self._forward > .10):
                return self._fail('Marker lost before near-field positioning')
        if self.state == 'ADVANCE' and now >= self._deadline:
            self.state = 'DONE'
            return StopMarkerCommand('DONE', done=True, reason='Short T marker reached; centre position is model-estimated')
        return StopMarkerCommand(self.state, active=True, duty=cfg['approach_duty'])

    def _fail(self, reason):
        self.state = 'FAILED'
        return StopMarkerCommand('FAILED', failed=True, reason=reason)
