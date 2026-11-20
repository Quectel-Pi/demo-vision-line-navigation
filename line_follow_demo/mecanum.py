from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MecanumGeometry:
    wheel_radius_m: float = 0.03
    wheelbase_x_m: float = 0.10  # 车体中心到前/后轮的 x 距离
    wheelbase_y_m: float = 0.10  # 车体中心到左/右轮的 y 距离


@dataclass(frozen=True)
class WheelCommands:
    fl: float
    fr: float
    rl: float
    rr: float

    def max_abs(self) -> float:
        return max(abs(self.fl), abs(self.fr), abs(self.rl), abs(self.rr))

    def scale(self, s: float) -> "WheelCommands":
        return WheelCommands(self.fl * s, self.fr * s, self.rl * s, self.rr * s)


def mecanum_inverse(vx_mps: float, vy_mps: float, wz_rps: float, geom: MecanumGeometry) -> WheelCommands:
    # 常见麦轮逆解（X 前进，Y 左移，Z 逆时针）
    # w = 1/r * [ vx - vy - (L+W)wz,
    #             vx + vy + (L+W)wz,
    #             vx + vy - (L+W)wz,
    #             vx - vy + (L+W)wz ]
    r = max(1e-6, float(geom.wheel_radius_m))
    lw = float(geom.wheelbase_x_m + geom.wheelbase_y_m)

    fl = (vx_mps - vy_mps - lw * wz_rps) / r
    fr = (vx_mps + vy_mps + lw * wz_rps) / r
    rl = (vx_mps + vy_mps - lw * wz_rps) / r
    rr = (vx_mps - vy_mps + lw * wz_rps) / r

    return WheelCommands(fl=fl, fr=fr, rl=rl, rr=rr)


def normalize_wheels(cmd: WheelCommands, max_abs: float = 1.0) -> WheelCommands:
    if max_abs <= 0:
        return cmd
    m = cmd.max_abs()
    if m <= max_abs or m <= 1e-9:
        return cmd
    return cmd.scale(max_abs / m)
