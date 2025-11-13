# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import math

# ====================== 공용 유틸 ======================

HEAD_KEYS = {
    "nose",
    "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
}


def head_center_from_coords(coords: Dict[str, Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    pts = [coords[k] for k in HEAD_KEYS if k in coords]
    if not pts:
        return coords.get("nose")
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def _sector_from_delta(dx: float, dy: float) -> str:
    ang = math.degrees(math.atan2(-dy, dx))  # 화면 기준: 위쪽이 +90°
    bins = [(-22.5, "E"), (22.5, "NE"), (67.5, "N"), (112.5, "NW"),
            (157.5, "W"), (-157.5, "W"), (-112.5, "SW"), (-67.5, "S"), (-22.5, "SE")]
    for th, name in bins:
        if ang <= th:
            return name
    return "E"


def _smoothstep_weight(val: float, vmin: float, vmax: float, gamma: float = 1.2) -> float:
    if val <= vmin:
        return 0.0
    if val >= vmax:
        w = 1.0
    else:
        t = (val - vmin) / max(1e-6, (vmax - vmin))
        t = max(0.0, min(1.0, t))
        w = t * t * (3 - 2 * t)
    return pow(w, gamma) if gamma and gamma > 0 else w


def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


# ====================== 3D 파라미터/함수 (신규) ======================

@dataclass
class HeadViewBias3DParams:
    # 머리↔홀드 사이 3D 거리 기준(미터) 가중치
    min_dist_m: float = 0.10  # 이하면 보정 0
    max_dist_m: float = 0.60  # 이상이면 보정 100%
    deadband_m: float = 0.05  # 미세 떨림 억제

    # 타깃 평면에서 허용할 최대 보정 오프셋(미터)
    max_offset_m: float = 0.07  # 7 cm

    # 각도 상한(절대, 도) — “1도 이하” 요구 반영(거리 짧아도 0.9° 캡)
    max_angle_cap_deg: float = 0.9

    gamma: float = 0.85
    reverse_direction: bool = True  # head→hold 반대 방향으로 보정


def compute_bias_angles_3d(
        head_xyz_m: Tuple[float, float, float],
        hold_xyz_m: Tuple[float, float, float],
        hold_range_m: Optional[float] = None,
        head_xy_px: Optional[Tuple[float, float]] = None,  # 섹터 로그용(선택)
        hold_xy_px: Optional[Tuple[float, float]] = None,  # 섹터 로그용(선택)
        params: HeadViewBias3DParams = HeadViewBias3DParams(),
):
    """
    입력 좌표계: (x:right, y:down, z:forward)  ← RealSense 일반 좌표 가정
    출력 각도: world yaw(+우측으로), pitch(+위로) [deg]
      - dyaw ≈ atan(Δx / R), dpitch ≈ -atan(Δy / R)  (소각 근사)
      - R: 레이저/카메라→타깃까지의 거리(미터). 미지정이면 hold_xyz의 거리 사용.
      - Δx, Δy: 타깃 평면에서 유도할 보정 오프셋(최대 max_offset_m)
    """
    if head_xyz_m is None or hold_xyz_m is None:
        return 0.0, 0.0, {"dist_m": 0.0, "weight": 0.0, "sector": "NA"}
    hx, hy, hz = map(float, head_xyz_m)
    tx, ty, tz = map(float, hold_xyz_m)

    # ① 3D 거리로 가중치
    dx3, dy3, dz3 = tx - hx, ty - hy, tz - hz
    d3 = math.sqrt(dx3 * dx3 + dy3 * dy3 + dz3 * dz3)
    if d3 <= params.deadband_m:
        sector = _sector_from_delta((hold_xy_px[0] - head_xy_px[0]) if (head_xy_px and hold_xy_px) else dx3,
                                    (hold_xy_px[1] - head_xy_px[1]) if (head_xy_px and hold_xy_px) else dy3)
        return 0.0, 0.0, {"dist_m": d3, "weight": 0.0, "sector": sector}

    w = _smoothstep_weight(d3, params.min_dist_m, params.max_dist_m, params.gamma)
    if w <= 0.0:
        sector = _sector_from_delta((hold_xy_px[0] - head_xy_px[0]) if (head_xy_px and hold_xy_px) else dx3,
                                    (hold_xy_px[1] - head_xy_px[1]) if (head_xy_px and hold_xy_px) else dy3)
        return 0.0, 0.0, {"dist_m": d3, "weight": 0.0, "sector": sector}

    # ② 보정 방향: head→hold의 반대 (x,y 만 사용해 화면상 8방위와 일치)
    bx, by = (-(tx - hx), -(ty - hy)) if params.reverse_direction else (tx - hx, ty - hy)
    n_xy = math.hypot(bx, by)
    if n_xy <= 1e-6:
        sector = "NA"
        return 0.0, 0.0, {"dist_m": d3, "weight": 0.0, "sector": sector}
    ux, uy = bx / n_xy, by / n_xy

    # ③ 타깃까지 거리(R) 산정
    if hold_range_m is None:
        # 카메라/레이저 원점과 홀드 사이 거리(벡터 노름). 레이저와 카메라 오프셋이 크면 별도로 넘겨줘.
        R = math.sqrt(tx * tx + ty * ty + tz * tz)
    else:
        R = max(1e-3, float(hold_range_m))

    # ④ 평면 오프셋 → 각도 변환
    #    최대 각도 = atan(max_offset / R), 단 절대 상한 cap 적용
    theta_max_deg = math.degrees(math.atan(params.max_offset_m / max(1e-3, R)))
    theta_max_deg = min(theta_max_deg, params.max_angle_cap_deg)  # “1도 미만” 유지

    #    실제 보정량 = theta_max * weight
    theta_deg = theta_max_deg * w

    #    yaw/pitch 분해: x, y 방향 유닛벡터(ux, uy)를 그대로 사용
    dyaw_world_deg = ux * theta_deg
    dpitch_world_deg = -uy * theta_deg  # 화면 y는 아래(+), 위로 틀려면 pitch +

    # ⑤ 로깅 정보
    if head_xy_px and hold_xy_px:
        dx_px = hold_xy_px[0] - head_xy_px[0]
        dy_px = hold_xy_px[1] - head_xy_px[1]
        sector = _sector_from_delta(dx_px, dy_px)
    else:
        sector = _sector_from_delta(tx - hx, ty - hy)

    info = {
        "dist_m": float(d3),
        "weight": float(w),
        "sector": sector,
        "R_m": float(R),
        "theta_max_deg": float(theta_max_deg),
    }
    return dyaw_world_deg, dpitch_world_deg, info
