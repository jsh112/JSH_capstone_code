# ==========================================================
# Laser-Body relationship logger (3-state version)
# ==========================================================

import time

def log_laser_relation(
    occlusion_logs,
    frame_id: int,
    hold_id: int,
    label: str,
    part: str,
    coords=None,
    laser_px=None,
    H=None,
):
    """
    레이저와 신체의 위치 관계를 3단계로 분류해 기록한다.
      1 = grip_contact      (손/발에 닿음)
      2 = body_blocked      (손/발 외의 신체 부위에 가림)
      3 = path_clear_above  (신체에 가려지지 않음 — 주로 위쪽 홀드 방향)
    """

    # 기본 상태
    state_code = 3  # 기본적으로 'path_clear_above'

    if label == "grip" and (part == "hand/foot" or part in ("left_hand", "right_hand", "left_foot", "right_foot")):
        state_code = 1
    elif label == "blocked":
        # 손발이 아닌 다른 부위면 blocked
        state_code = 2

    # (선택) 신체보다 위쪽에 있는 경우 추가 확인 (보정용)
    if coords is not None and laser_px is not None and H is not None:
        _, ly = laser_px
        min_body_y = min([p[1] for p in coords.values()]) if coords else H
        if ly < (min_body_y - 10):  # 여유 margin 10px
            state_code = 3

    occlusion_logs.append({
        "frame_id": frame_id,
        "timestamp": time.time(),
        "hold_id": hold_id,
        "label": label,
        "blocked_part": part,
        "state_code": state_code,
    })

    return state_code

import os, datetime, pandas as pd

def save_occlusion_log(occlusion_logs, log_dir="."):
    """
    occlusion_logs 리스트를 CSV로 저장한다.
    파일명: metric_occlusion_log_YYYYMMDD_HHMMSS.csv
    """
    if not occlusion_logs:
        print("[Metric] No occlusion logs to save.")
        return None

    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(log_dir, exist_ok=True)
    log_name = f"metric_occlusion_log_{timestamp_str}.csv"
    log_path = os.path.join(log_dir, log_name)

    pd.DataFrame(occlusion_logs).to_csv(log_path, index=False)
    print(f"[Metric] Occlusion log saved: {log_path} ({len(occlusion_logs)} entries)")
    return log_path