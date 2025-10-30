import time
import cv2
import numpy as np
from ultralytics import YOLO
import csv
import argparse

# === MediaPipe 모듈 ===
from Climb_Mediapipe import PoseTracker, draw_pose_points, classify_occluder

# 레이저 찾기
from find_laser import capture_once_and_return

# hold 관련 코드
from hold_utils import initial_5frames_all_classes

# servo 관련 코드
from servo_utils import send_servo_angles, yaw_pitch_from_X

from click_select import interactive_select_live_left_only

from web import choose_color_via_web

from realsense_adapter import RealSenseColorDepth

# ========= 사용자 환경 경로 =========
MODEL_PATH     = r"C:\Users\jshkr\OneDrive\문서\JSH_CAPSTONE_CODE\windows\param\best_6.pt"

SWAP_DISPLAY   = False   # 화면 표시 좌/우 스와프

WINDOW_NAME    = "Rectified L | R"
THRESH_MASK    = 0.7
ROW_TOL_Y      = 10

# 자동 진행(터치→다음 홀드) 관련
TOUCH_THRESHOLD = 1     # in-polygon 연속 프레임 임계(기본 10)
ADV_COOLDOWN    = 0.5    # 연속 넘김 방지 쿨다운(sec)

# ✅ 시간 기반 디버그 임계(초)
TOUCH_TIME_S    = 0.150 

# 저장 옵션
SAVE_VIDEO     = False
OUT_FPS        = 30
OUT_PATH       = "stereo_overlay.mp4"
CSV_GRIPS_PATH = "grip_records.csv"

# 런타임 보정 오프셋(레이저 실측)
CAL_YAW_OFFSET   = 0.0
CAL_PITCH_OFFSET = 0.0

# ---- 레이저 원점(LEFT 기준) 오프셋 (cm) ----
LASER_OFFSET_CM_LEFT = 2.5
LASER_OFFSET_CM_UP   = 7.3
LASER_OFFSET_CM_FWD  = -8.0
Y_UP_IS_NEGATIVE     = True  # 위 방향이 -y인 좌표계면 True

# === 서보 기준(중립 90/90) & 부호/스케일 ===
BASE_YAW_DEG   = 90.0   # 서보 중립
BASE_PITCH_DEG = 90.0   # 서보 중립
YAW_SIGN       = -1.0   # 반대로 가면 -1.0
PITCH_SIGN     = +1.0   # 반대로 가면 -1.0
YAW_SCALE      = 1.0    # 필요시 감도 미세조정
PITCH_SCALE    = 1.0

# === 전역 기준(초기 레이저 기준 각) ===
YAW_LASER0 = None
PITCH_LASER0 = None

K_EXTRA_PITCH_DEG = 1.0    # 최대 추가 피치 스케일(도)
K_EXTRA_YAW_DEG = -1.0      # yaw 스케일

# 높이 차이 기반 가산 파라미터
H0_MM       = 1400.0 # 높이 정규화(1 m)
Z0_MM       = 4000.0 # 깊이 정규화(4 m에서 감쇠=1배)
BETA_Z      = 1.2    # 깊이 감쇠 기울기(0이면 감쇠 없음)
H_SOFT_MM  = 160.0   # 높이차가 이 정도 이하일 때 가산을 살짝 눌러줌
GAMMA_H    = 0.1     # 1.0~1.5 권장 (커지면 초반 더 세게 눌림)

# --- 좌/우(레이저 기준 X) 가산 파라미터 ---
X0_MM        = 1200.0   # 좌우 정규화 스케일(1 m)
X_SOFT_MM    = 120.0    # 초기 구간 소프트 스타트 완화 폭
GAMMA_X      = 0.2      # 소프트 스타트 기울기(0.1~0.5 권장)
X_DEADBAND_MM= 25.0     # 레이저와 거의 같은 X일 때(±deadband) 가산 0

# === 이미지 차분 파라미터 ===
DIFF_EVERY_N     = 3      # N프레임마다 차분(=부하 감소)
ADAPT_ALPHA      = 0.02   # 배경(베이스라인) 적응 속도(0=고정)
DILATE_KERNEL_SZ = 3      # 마스크 팽창 커널(노이즈↓/완충)

# === 강한 차분 기반 판정 파라미터(occlusion robust) ===
TH_DIFF_HARD   = 35        # 8-bit 강한 차분 임계(권장 30~45)
FRAC_HARD_MIN  = 0.30      # 강한 차분 픽셀 비율 최소(30% 이상이면 '크게 달라짐')
FRAC_DYN_MIN   = 0.55      # 동적 임계(노이즈 적응) 기준 비율
ERODE_ITERS    = 1         # 마스크 코어만 사용(경계 흔들림 억제). 0~1 권장

ROTATE_MAP = {
    1: cv2.ROTATE_90_COUNTERCLOCKWISE,  # LEFT
    2: cv2.ROTATE_90_CLOCKWISE,         # RIGHT
}

CAP_SIZE = (1280, 720)
size = CAP_SIZE 
# ======== Servo controller import (stub fallback) ========
try:
    from servo_control import DualServoController
    HAS_SERVO = True
except Exception:
    HAS_SERVO = False
    class DualServoController:
        def __init__(self, *a, **k): print("[Servo] (stub) controller unavailable")
        def set_angles(self, pitch=None, yaw=None): print(f"[Servo] (stub) set_angles: P={pitch}, Y={yaw}")
        def center(self): print("[Servo] (stub) center")
        def query(self): print("[Servo] (stub) query"); return ""
        def laser_on(self): print("[Servo] (stub) laser_on")
        def laser_off(self): print("[Servo] (stub) laser_off")
        def close(self): pass
# ======================

def rotate_image(img, rot_code):
    return cv2.rotate(img, rot_code) if rot_code is not None else img

def rotate_point(pt, shape_hw, rot_code):
    """(x,y) 픽셀을 주어진 회전 코드로 변환. shape_hw는 '회전 전'의 (H,W)."""
    if pt is None or rot_code is None: 
        return pt
    h, w = shape_hw
    x, y = int(pt[0]), int(pt[1])
    if rot_code == cv2.ROTATE_90_CLOCKWISE:
        return (h - 1 - y, x)
    elif rot_code == cv2.ROTATE_90_COUNTERCLOCKWISE:
        return (y, w - 1 - x)
    elif rot_code == cv2.ROTATE_180:
        return (w - 1 - x, h - 1 - y)
    return (x, y)

def _open_camera_and_model():
    cap = RealSenseColorDepth(color=(1280,720,30), depth=(848,480,30), align_to_color=True, rotate90=False)
    model = YOLO(str(MODEL_PATH))
    return cap, model

def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM4")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--no_auto_advance", action="store_true")
    ap.add_argument("--no_web", action="store_true")
    ap.add_argument("--laser_on", action="store_true")
    return ap.parse_args()

def wrap_deg(d):
    return (d + 180.0) % 360.0 - 180.0

def extra_pitch_deg(X, O, X_laser, y_up_is_negative=True):
    """
    X: 홀드 3D(mm)
    O: 레이저 원점 3D(mm)  -> 깊이 기준점(Z만 사용)
    X_laser: 레이저 첫 점 3D(mm) -> 높이 기준점(Y만 사용)
    return: 홀드에 추가로 줄 피치(도, +면 더 위로)
    """
    if X is None or O is None or X_laser is None:
        return 0.0

    # --- 높이 차 h (위가 양수되도록) ---
    Yh = float(X[1]); Y0 = float(X_laser[1])
    h = (Y0 - Yh) if y_up_is_negative else (Yh - Y0)
    if h <= 0.0:
        return 0.0  # 레이저보다 낮으면 가산 없음

    # --- 깊이(Z) 감쇠 (옆(X) 영향 제거) ---
    Zh = float(X[2]); Z0 = float(O[2])
    z_depth = max(1.0, abs(Zh - Z0))          # mm

    # --- 높이 소프트 스타트 ---
    #  h가 작을수록 (h/(h+H_SOFT))가 0~0.5 근처 → 가산 살짝 눌림
    #  h가 커질수록 → 1에 수렴 → 기존 로직과 동일
    soft_h = (h / (h + H_SOFT_MM)) ** GAMMA_H

    height_term = (h / max(1.0, H0_MM)) * soft_h
    depth_term  = (Z0_MM / z_depth) ** BETA_Z

    return K_EXTRA_PITCH_DEG * height_term * depth_term

def extra_yaw_deg(X, O, X_laser, y_up_is_negative=True):
    """
    레이저 기준 X(좌/우)만으로 yaw에 가산을 준다.
    - 목표가 레이저보다 '오른쪽'(dx>0)이면 +방향 가산
    - '왼쪽'(dx<0)이면 -방향 가산
    - 레이저와 거의 같은 X(±X_DEADBAND_MM)이면 0
    - 멀수록(깊이 차↑) 감쇠(BETA_Z, Z0_MM)
    """
    if X is None or O is None or X_laser is None:
        return 0.0

    # --- 좌/우 오프셋(mm): +면 목표가 레이저보다 오른쪽 ---
    dx = float(X[0]) - float(X_laser[0])

    # --- 데드밴드: 거의 같은 X면 가산 0 ---
    if abs(dx) <= X_DEADBAND_MM:
        return 0.0

    # 데드밴드 이후의 유효 거리
    dx_eff = abs(dx) - X_DEADBAND_MM

    # --- 소프트 스타트(초반 과보정 방지) ---
    soft_x = (dx_eff / (dx_eff + X_SOFT_MM)) ** GAMMA_X

    # 좌/우 정규화 항(거리 커질수록 가중↑)
    lateral_term = (dx_eff / max(1.0, X0_MM)) * soft_x

    # --- 깊이(Z) 감쇠: 멀수록 덜 가산 ---
    Zh = float(X[2]); Z0 = float(O[2])
    z_depth = max(1.0, abs(Zh - Z0))
    depth_term = (Z0_MM / z_depth) ** BETA_Z

    # 부호: 오른쪽(+), 왼쪽(-)
    sgn = 1.0 if dx > 0 else -1.0

    # 최종 가산(deg). K_EXTRA_YAW_DEG 부호/크기는 현장에서 튜닝
    return sgn * K_EXTRA_YAW_DEG * lateral_term * depth_term



def depth_median_in_mask(depth_m, mask):
    ys, xs = np.where(mask > 0)
    if xs.size == 0: return None
    vals = depth_m[ys, xs]
    vals = vals[(vals > 0) & np.isfinite(vals)]
    if vals.size == 0: return None
    return float(np.median(vals))

def holds_to_3d(holds, depth_m, deproject_fn):
    """각 hold에 X(mm) 추가. center 기준 아니라 mask median Z 사용."""
    out = []
    for h in holds:
        cx, cy = h["center"]
        Zm = depth_median_in_mask(depth_m, h["mask"])  # m
        if Zm is None or Zm <= 0:
            continue
        X_m = deproject_fn(cx, cy, Zm)                 # (m)
        X_mm = X_m * 1000.0
        hh = dict(h)
        hh["X"] = X_mm
        out.append(hh)
    return out

def compute_laser_origin_cam():
    # 카메라 원점(0,0,0)에서 레이저 장착 위치(mm)로의 오프셋
    dx = -LASER_OFFSET_CM_LEFT * 10.0
    dy = (-1.0 if Y_UP_IS_NEGATIVE else 1.0) * LASER_OFFSET_CM_UP * 10.0
    dz = LASER_OFFSET_CM_FWD * 10.0
    O = np.array([dx, dy, dz], dtype=np.float64)
    L = np.array([0.0, 0.0, 0.0], dtype=np.float64)  # 사용 안 해도 호환 위해 유지
    print(f"[Laser] Origin O (mm, CAM-based) = {O}")
    return L, O

def _capture_laser_raw(args):
    try:
        laser = capture_once_and_return(
            port=args.port, baud=args.baud,
            wait_s=2.0, settle_n=8, show_preview=True,
            center_pitch=90.0, center_yaw=90.0, servo_settle_s=0.5, frame_size=(1280, 720),
        )
    except Exception as e:
        print(f"[A_Climbing] find_laser error: {e} → continue without laser")
        return None

    if not laser:
        print("[A_Climbing] 레이저 좌표 취득 실패(취소/에러). 계속 진행.")
        return None

    # 최신 포맷: left_raw / X_laser_mm
    print(f"[A_Climbing] 레이저(raw,3D): left_raw={laser.get('left_raw')}, X_laser_mm={laser.get('X_laser_mm')}")
    return laser

def compute_laser_origin_mid(T):
    L = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    M = (T.reshape(3) / 2.0)
    dx = -LASER_OFFSET_CM_LEFT * 10.0
    dy = (-1.0 if Y_UP_IS_NEGATIVE else 1.0) * LASER_OFFSET_CM_UP * 10.0
    dz = LASER_OFFSET_CM_FWD * 10.0
    O  = M + np.array([dx, dy, dz], dtype=np.float64)
    print(f"[Laser] Origin O (mm, MID-based) = {O}")
    return L, O

def _cluster_rows_by_y(holds, row_tol=30):
    """y기준으로 행 클러스터링: [(row_y, [idx들])] 반환"""
    rows = []
    for i, h in enumerate(holds):
        cy = h["center"][1]
        assigned = False
        for row in rows:
            if abs(cy - row["y"]) <= row_tol:
                row["idxs"].append(i)
                row["y"] = int(round(np.mean([holds[j]["center"][1] for j in row["idxs"]])))
                assigned = True
                break
        if not assigned:
            rows.append({"y": int(cy), "idxs": [i]})
    # y 오름차순
    rows.sort(key=lambda r: r["y"])
    return [(r["y"], r["idxs"]) for r in rows]

def _mask_from_contour(shape_hw, contour, dilate_k=DILATE_KERNEL_SZ):
    H, W = shape_hw
    m = np.zeros((H, W), np.uint8)
    cnt = np.asarray(contour, dtype=np.int32)
    if cnt.ndim == 2:  # (N,2) -> (N,1,2)
        cnt = cnt.reshape(-1, 1, 2)
    cv2.drawContours(m, [cnt], -1, 255, -1)
    if dilate_k and dilate_k > 0:
        ker = np.ones((dilate_k, dilate_k), np.uint8)
        m = cv2.dilate(m, ker, iterations=1)
    return m

def _foot_mask_from_coords(coords, shape_hw):
    H, W = shape_hw
    m = np.zeros((H, W), np.uint8)

    def _draw_capsule_local(mask, p1, p2, radius_px):
        p1 = tuple(map(int, map(round, p1)))
        p2 = tuple(map(int, map(round, p2)))
        r  = int(round(float(radius_px)))
        if r <= 0: return
        cv2.line(mask, p1, p2, 255, thickness=2*r, lineType=cv2.LINE_AA)
        cv2.circle(mask, p1, r, 255, -1, cv2.LINE_AA)
        cv2.circle(mask, p2, r, 255, -1, cv2.LINE_AA)

    def _draw_foot(heel_name, index_name):
        if heel_name in coords and index_name in coords:
            A = np.array(coords[heel_name], dtype=float)
            B = np.array(coords[index_name], dtype=float)
            L = np.linalg.norm(B - A)
            r = max(12.0, 0.22 * L)   # 필요시 튜닝: 0.18~0.28
            _draw_capsule_local(m, A, B, r)

    _draw_foot("left_heel",  "left_foot_index")
    _draw_foot("right_heel", "right_foot_index")
    return m

def _build_hold_db_with_baseline(cap, size, holds, n_frames=10, diff_gate=12.0):
    W, H = size
    db = {}
    for h in holds:
        x,y,w,hh = cv2.boundingRect(h["contour"])
        db[h["hold_index"]] = {
            "mask": _mask_from_contour((H,W), h["contour"]),
            "bbox": (x,y,w,hh),
            "center": tuple(h["center"]),
            "samples": [],
        }
    frames = []
    for _ in range(n_frames):
        ok, f = cap.read()
        if not ok: continue
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    if not frames:
        raise SystemExit("[Diff] 초기 베이스라인 프레임을 하나도 못 얻었습니다.")
    for hid, info in db.items():
        x,y,w,hh = info["bbox"]
        x = max(0, min(x, W-1)); y = max(0, min(y, H-1))
        w = max(1, min(w, W - x)); hh = max(1, min(hh, H - y))
        crops = [g[y:y+hh, x:x+w] for g in frames]
        if not crops: continue
        keep, ref = [], crops[0]
        for c in crops:
            if cv2.mean(cv2.absdiff(ref, c))[0] < diff_gate:
                keep.append(c)
        use = keep if len(keep) >= 3 else crops
        baseline = np.median(np.stack(use, axis=0), axis=0).astype(np.uint8)
        info["baseline"] = baseline
        info.pop("samples", None)
    return db

def servo_cmd_from_laser_ref(yaw_hold, pitch_hold, yaw_laser0, pitch_laser0):
    d_yaw   = wrap_deg(yaw_hold  - yaw_laser0) + CAL_YAW_OFFSET
    d_pitch = wrap_deg(pitch_hold - pitch_laser0) + CAL_PITCH_OFFSET

    target_yaw   = BASE_YAW_DEG   + YAW_SIGN   * (YAW_SCALE   * d_yaw)
    target_pitch = BASE_PITCH_DEG + PITCH_SIGN * (PITCH_SCALE * d_pitch)
    target_yaw   = max(0.0, min(180.0, target_yaw))
    target_pitch = max(0.0, min(180.0, target_pitch))
    return target_yaw, target_pitch

def depth_median_at(depth_m, x, y, r=3):
    h, w = depth_m.shape[:2]
    x0, x1 = max(0, x-r), min(w-1, x+r)
    y0, y1 = max(0, y-r), min(h-1, y+r)
    patch = depth_m[y0:y1+1, x0:x1+1]
    vals = patch[(patch > 0) & np.isfinite(patch)]
    if vals.size == 0: 
        return None
    return float(np.median(vals))

def _event_loop(size):
    W, H = size
    pose = PoseTracker(min_detection_confidence=0.5, model_complexity=1)
    blocked_state = {}
    out = None

    if SAVE_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(OUT_PATH, fourcc, OUT_FPS, (W, H))
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    t_prev = time.time()
    last_advanced_time = 0.0
    return pose, blocked_state, out, t_prev, last_advanced_time

def assign_indices_row_major(holds, row_tol=30):
    """
    매칭 없이 '한쪽' 목록에만 번호 부여:
    1) y기반 행 클러스터링(row_tol px)
    2) 각 행 안에서 x 오름차순
    3) 위 순서대로 hold_index = 0..N-1 부여
    return: 새 순서대로 정렬된 holds 리스트
    """
    if not holds:
        return []

    rows = _cluster_rows_by_y(holds, row_tol=row_tol)  # 이미 코드에 있는 함수 사용
    ordered = []
    hid = 0
    for row_y, idxs in rows:  # y 오름차순으로 정렬된 행
        # 행 내부는 x 오름차순
        idxs_sorted = sorted(idxs, key=lambda i: holds[i]["center"][0])
        for i in idxs_sorted:
            holds[i]["hold_index"] = hid
            ordered.append(holds[i])
            hid += 1
    return ordered

def build_servo_targets(by_id, yaw_laser0, pitch_laser0, X_laser, O):
    servo_targets = {}
    for hid, mr in by_id.items():
        yaw_h, pitch_h = mr["yaw_deg"], mr["pitch_deg"]
        ty, tp = servo_cmd_from_laser_ref(yaw_h, pitch_h, yaw_laser0, pitch_laser0)

        # ★ Pitch만 Z깊이 감쇠로 가중치 적용
        tp += extra_pitch_deg(mr.get("X"), O, X_laser, y_up_is_negative=Y_UP_IS_NEGATIVE)
        ty += extra_yaw_deg(mr.get("X"), O, X_laser, y_up_is_negative=Y_UP_IS_NEGATIVE)

        ty = max(0.0, min(180.0, ty))
        tp = max(0.0, min(180.0, tp))
        servo_targets[hid] = (ty, tp)
    return servo_targets

# === (NEW) CSV에서 (part, hold_id) 순서 로드 ===
def load_route_pairs_from_csv(path):
    route_pairs = []
    try:
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # 첫 코드 스키마: ["part","hold_id","cx","cy"]
                part = row.get("part")
                hid  = row.get("hold_id")
                if part is None or hid is None:
                    continue
                try:
                    hid = int(hid)
                except:
                    continue
                # (part, hold_id) 그대로 누적 (CSV 순서 보존, 중복허용)
                route_pairs.append((part, hid))
    except FileNotFoundError:
        print(f"[Warn] 경로 CSV '{path}' 없음 → 기본 순서 사용 불가")
    return route_pairs

def _resolve_part_name(tpart: str, coords: dict, grip_names: set):
    """
    CSV의 part 이름을 MediaPipe coords 키로 정규화.
    1) 완전일치
    2) 느슨한 매칭(서로 포함/접두)
    3) tip/dip/pip/mcp 꼬리표 제거 후 grip_parts와 근사 매칭
    못 찾으면 None
    """
    if not coords or not tpart:
        return None
    if tpart in coords:
        return tpart

    keys = list(coords.keys())
    # 느슨한 매칭
    for k in keys:
        if k.startswith(tpart) or tpart.startswith(k) or (tpart in k) or (k in tpart):
            return k

    # suffix 제거 후 근사
    base = tpart
    for suffix in ["_tip", "_dip", "_pip", "_mcp"]:
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    for g in (grip_names or []):
        if g == base or g in tpart or base in g:
            for k in keys:
                if g in k or k in g:
                    return k
    return None

def _run_frame_loop(cap, size, holds, matched_results,
                    servo_targets,auto_advance_enabled, pose, blocked_state,
                    out, t_prev, last_advanced_time, current_target_id, cur_yaw, cur_pitch, ctl,
                    route_pairs, route_pos, current_target_part,
                    hold_db, laser_px=None, yaw_laser0=0.0, pitch_laser0=0.0):
    W, H = size
    touch_streak = {}
    frame_id = 0

    holds_by_id = {h["hold_index"]: h for h in holds}

    # ===== 시간 기반 터치 트래킹/로깅 상태 =====
    TOUCH_TIME_S_LOCAL = globals().get("TOUCH_TIME_S", 0.350)  # 상단에 TOUCH_TIME_S 없으면 350ms 기본
    contact_start_ts = {}   # {(part, hold_id): 시작 시각}
    reported_350ms   = set()# 350ms 통과 후 이미 로그 출력한 키
    last_prog_print  = {}   # 진행 로그(스팸 방지용) 최근 출력 시각

    try:
        # 통계로 쓸 리스트 추가
        occlusion_logs = []
        while True:
            ok, Limg = cap.read()
            if not ok:
                print("[Warn] 프레임 캡쳐 실패"); break
            vis = Limg.copy()

            # === (NEW) 최초 레이저 위치 오버레이 (맨 마지막에 그려 가려지지 않게) ===
            if laser_px is not None:
                lx, ly = int(laser_px[0]), int(laser_px[1])
                H_, W_ = vis.shape[:2]
                if 0 <= lx < W_ and 0 <= ly < H_:
                    cv2.circle(vis, (lx, ly), 12, (255, 255, 255), 2, cv2.LINE_AA)
                    msg = f"Laser0 px=({lx},{ly})  yaw0={yaw_laser0:.2f}  pitch0={pitch_laser0:.2f}"
                else:
                    msg = f"Laser0 px off-frame: ({lx},{ly}) not in {W_}x{H_}"
                cv2.putText(vis, msg, (20, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2, cv2.LINE_AA)
                cv2.putText(vis, msg, (20, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2, cv2.LINE_AA)

            # (선택) 홀드/라벨 표시
            for h in holds:
                cv2.drawContours(vis, [h["contour"]], -1, h["color"], 2)
                cx, cy = h["center"]
                cv2.circle(vis, (cx, cy), 4, (255,255,255), -1)
                tag = f"ID:{h['hold_index']}"
                if h["hold_index"] == current_target_id:
                    tag = "[TARGET] " + tag
                cv2.putText(vis, tag, (cx - 10, cy + 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
                cv2.putText(vis, tag, (cx - 10, cy + 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, h["color"], 2, cv2.LINE_AA)

            # 3D/깊이 디버그 HUD(초기 계산값 사용)
            y_info = 60
            for mr in matched_results:
                X = mr["X"]; depth = X[2]
                txt3d = f"ID{mr['hid']} : X=({X[0]:.1f},{X[1]:.1f},{X[2]:.1f}) mm | depth(Z)={depth:.1f} mm"
                cv2.putText(vis, txt3d, (20, y_info),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2, cv2.LINE_AA)
                cv2.putText(vis, txt3d, (20, y_info),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
                y_info += 18

            # 포즈 추정
            coords = pose.process(Limg)
            draw_pose_points(vis, coords, offset_x=0)

            if coords:
                H_, W_ = vis.shape[:2]
                foot_mask = np.zeros((H_, W_), np.uint8)

                def _draw_capsule(mask, p1, p2, r):
                    # 선 두께를 2r로 그리면 내부가 채워진 캡슐이 됨
                    cv2.line(mask, p1, p2, 255, thickness=2*r, lineType=cv2.LINE_AA)
                    cv2.circle(mask, p1, r, 255, -1, cv2.LINE_AA)
                    cv2.circle(mask, p2, r, 255, -1, cv2.LINE_AA)

                for heel_name, index_name in [("left_heel","left_foot_index"),
                                            ("right_heel","right_foot_index")]:
                    if heel_name in coords and index_name in coords:
                        Ax, Ay = map(lambda v: int(round(v)), coords[heel_name])
                        Bx, By = map(lambda v: int(round(v)), coords[index_name])
                        L = float(np.hypot(Bx - Ax, By - Ay))
                        r = max(12, int(round(0.22 * L)))
                        _draw_capsule(foot_mask, (Ax, Ay), (Bx, By), r)

                if np.count_nonzero(foot_mask) > 0:
                    # 반투명 채움
                    overlay = vis.copy()
                    overlay[foot_mask > 0] = (0, 255, 255)  # 노란색
                    cv2.addWeighted(overlay, 0.35, vis, 0.65, 0, vis)

                    # 외곽선
                    cnts, _ = cv2.findContours(foot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if cnts:
                        cv2.drawContours(vis, cnts, -1, (0, 255, 255), 2, cv2.LINE_AA)

            # 타깃 홀드 이미지 차분
            if frame_id % DIFF_EVERY_N == 0 and current_target_id in hold_db:
                info = hold_db[current_target_id]
                x,y,w,hh = info["bbox"]
                grayL = cv2.cvtColor(Limg, cv2.COLOR_BGR2GRAY)
                patch = grayL[y:y+hh, x:x+w]
                base  = info["baseline"]

                mroi_full = info["mask"][y:y+hh, x:x+w]
                mroi_core = mroi_full
                if ERODE_ITERS > 0:
                    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
                    mroi_core = cv2.erode(mroi_full, k, iterations=ERODE_ITERS)

                diff = cv2.absdiff(base, patch)
                roi_idx = (mroi_core > 0)
                vals = diff[roi_idx]
                if vals.size == 0:
                    is_motion = False
                else:
                    med = float(np.median(vals))
                    mad = float(np.median(np.abs(vals - med))) + 1e-6
                    sigma = 1.4826 * mad
                    tau_dyn = max(TH_DIFF_HARD * 0.6, 3.0 * sigma)
                    frac_hard = float((vals >= TH_DIFF_HARD).mean())
                    frac_dyn  = float((vals >= tau_dyn).mean())
                    is_motion = (frac_hard >= FRAC_HARD_MIN) or (frac_dyn >= FRAC_DYN_MIN)

                is_block = False; block_part = None
                if is_motion:
                    label, who = classify_occluder(
                        center_xy=info["center"],
                        coords=coords,
                        hold_mask=info.get("mask")
                    )
                    is_block   = (label != "grip")
                    block_part = (who if is_block else None)

                if ADAPT_ALPHA > 0 and is_motion and not is_block:
                    base_f = base.astype(np.float32)
                    cv2.accumulateWeighted(patch.astype(np.float32), base_f, ADAPT_ALPHA, mask=mroi_core)
                    info["baseline"] = base_f.astype(np.uint8)

                if is_block:
                    part_to_report = block_part or "unknown"
                    if part_to_report != getattr(_run_frame_loop, "_prev_block_part", None):
                        print(f"[BLOCKED] 홀드 ID {current_target_id} — {part_to_report} 부위가 가림")
                        _run_frame_loop._prev_block_part = part_to_report
                        # === [지표2 로깅 추가] ===
                        occlusion_logs.append({
                            "frame_id": frame_id,
                            "timestamp": time.time(),
                            "hold_id": current_target_id,
                            "blocked_part": part_to_report,
                            "label": label,  # 'grip' or 'blocked'
                        })
                else:
                    # [NEW] grip 상태도 일정 간격으로 기록 (데이터 폭 방지용)
                    if frame_id % 5 == 0:  # 20FPS 기준 약 4Hz 샘플링
                        occlusion_logs.append({
                            "frame_id": frame_id,
                            "timestamp": time.time(),
                            "hold_id": current_target_id,
                            "blocked_part": "none",
                            "label": "grip",
                        })
                    _run_frame_loop._prev_block_part = None

            # 목표 (part, hold_id) 판정
            if coords and (current_target_id in holds_by_id):
                tid  = current_target_id
                tpart_csv = current_target_part
                hold = holds_by_id.get(tid)

                advanced_this_frame = False
                current_touched = set()
                tpart = _resolve_part_name(tpart_csv, coords, getattr(pose, "grip_parts", set()))

                if hold is None:
                    cv2.putText(vis, f"[WARN] hold_id {tid} not present", (20, 46),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)
                else:
                    # === (NEW) 손/발 그룹 후보 구성 ===
                    hand_set = getattr(pose, "hand_parts", set())
                    foot_set = getattr(pose, "foot_grip_parts", set())

                    csv_wants_hand = (tpart_csv in {"hand", "any_hand"})
                    csv_wants_foot = (tpart_csv in {"foot", "any_foot"})

                    is_hand_part = (tpart in hand_set)
                    is_foot_part = (tpart in foot_set)

                    if csv_wants_foot or is_foot_part:
                        group_label = "foot"
                        touched_any = False

                        # ★ 발은 마스크↔마스크 겹침으로 판정
                        foot_m = _foot_mask_from_coords(coords, (H, W))
                        info = hold_db.get(tid)
                        if info is not None:
                            hold_m = info.get("mask")
                            if hold_m is not None:
                                overlap = cv2.bitwise_and(foot_m, hold_m)
                                if np.count_nonzero(overlap) > 0:
                                    touched_any = True

                        display_key = (group_label, tid)

                    else:
                        # hand 또는 특정 파트는 기존처럼 포인트 in-polygon
                        if csv_wants_hand or is_hand_part:
                            candidate_parts = [n for n in hand_set if n in coords]
                            group_label = "hand"
                        else:
                            candidate_parts = [tpart] if (tpart in coords) else []
                            group_label = (tpart or tpart_csv)

                        display_key = (group_label, tid)
                        touched_any = False
                        for pname in candidate_parts:
                            x, y = coords[pname]
                            if cv2.pointPolygonTest(hold["contour"], (x, y), False) >= 0:
                                touched_any = True
                                break

                    # (공통) 그룹 키 기반 current_touched 설정
                    current_touched = set()
                    if touched_any:
                        current_touched.add(display_key)

                    # ===== 시간 기반 카운트/로그 갱신 (display_key 사용) =====
                    key_ts = display_key  # 시간 누적도 그룹 단위로
                    now_t  = time.time()
                    if touched_any:
                        if key_ts not in contact_start_ts:
                            contact_start_ts[key_ts] = now_t
                        dur = now_t - contact_start_ts[key_ts]

                        if dur < TOUCH_TIME_S_LOCAL and (now_t - last_prog_print.get(key_ts, 0)) > 0.10:
                            print(f"[DBG] touching {key_ts[0]}@{key_ts[1]}  {dur*1000:.0f}ms / {TOUCH_TIME_S_LOCAL*1000:.0f}ms")
                            last_prog_print[key_ts] = now_t

                        if dur >= TOUCH_TIME_S_LOCAL and key_ts not in reported_350ms:
                            print(f"[GRIP-350ms] confirmed: {key_ts[0]}@{key_ts[1]}  ({dur*1000:.0f} ms)")
                            reported_350ms.add(key_ts)
                    else:
                        contact_start_ts.pop(key_ts, None)
                        last_prog_print.pop(key_ts, None)
                        reported_350ms.discard(key_ts)

                    # HUD도 그룹 라벨로
                    cnt = touch_streak.get(display_key, 0)
                    cv2.putText(vis, f"[GRIP] {group_label}@{tid} {cnt}/{TOUCH_THRESHOLD}",
                                (20, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,0), 3, cv2.LINE_AA)
                    cv2.putText(vis, f"[GRIP] {group_label}@{tid} {cnt}/{TOUCH_THRESHOLD}",
                                (20, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50,220,50), 2, cv2.LINE_AA)

                    # 프레임 단위 카운트 누적/초기화
                    for key in current_touched:
                        touch_streak[key] = touch_streak.get(key, 0) + 1
                        if touch_streak[key] >= TOUCH_THRESHOLD:
                            now_t = time.time()
                            if (auto_advance_enabled
                                and (route_pos < len(route_pairs) - 1)
                                and (now_t - last_advanced_time) > ADV_COOLDOWN
                                and not advanced_this_frame):
                                next_part, next_tid = route_pairs[route_pos + 1]
                                ty_tp = servo_targets.get(next_tid)
                                if ty_tp is not None:
                                    target_yaw, target_pitch = ty_tp
                                    send_servo_angles(ctl, target_yaw, target_pitch)
                                    cur_yaw, cur_pitch = target_yaw, target_pitch
                                    route_pos += 1
                                    current_target_id   = next_tid
                                    current_target_part = next_part
                                    last_advanced_time = now_t
                                    advanced_this_frame = True
                                    touch_streak.clear()
                                    break

                    # 그룹 키가 아닌 것은 0으로
                    for key in list(touch_streak.keys()):
                        if key != display_key:
                            # 다른 키는 이번 프레임에 터치가 없으면 0으로
                            if key not in current_touched:
                                touch_streak[key] = 0

            # FPS/HUD
            t_now = time.time()
            fps = 1.0 / max(t_now - (t_prev), 1e-6); t_prev = t_now
            cv2.putText(vis, f"FPS: {fps:.1f} (Auto={'ON' if auto_advance_enabled else 'OFF'})",
                        (10, H-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)
            cv2.putText(vis, f"FPS: {fps:.1f} (Auto={'ON' if auto_advance_enabled else 'OFF'})",
                        (10, H-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1, cv2.LINE_AA)

            cv2.imshow(WINDOW_NAME, vis)
            if SAVE_VIDEO:
                if out is None:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(OUT_PATH, fourcc, OUT_FPS, (W, H))  # ← 단일 영상
                out.write(vis)

            frame_id += 1
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break

    finally:
        cap.release()
        if SAVE_VIDEO and out is not None:
            out.release(); print(f"[Info] 저장 완료: {OUT_PATH}")
        # 분석용 로그 추가
        if occlusion_logs:
            import pandas as pd
            import datetime
            timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_name = f"metric_occlusion_log_{timestamp_str}.csv"
            pd.DataFrame(occlusion_logs).to_csv(log_name, index=False)
            print(f"[Metric] Occlusion log saved: {log_name} ({len(occlusion_logs)} entries)")
            print(f"[Metric] Occlusion log saved ({len(occlusion_logs)} entries)")
        cv2.destroyAllWindows()
        try: pose.close()
        except: pass
        try: ctl.close()
        except: pass

    return current_target_id, cur_yaw, cur_pitch


def main():
    args = _parse_args()

    # (1) 레이저 찾기 (단일 윈도우; find_laser.py에서)
    laser_raw = _capture_laser_raw(args)  # {"left_raw": (x,y), "right_raw": ...} 또는 None

    # (2) D455 오픈 + YOLO 모델
    cap, model = _open_camera_and_model()

    # (3) 웹 스와치 색 선택 (CSV 결정)
    sel = choose_color_via_web()
    color_slug = sel if sel else "all"
    CSV_GRIPS_PATH_dyn = f"grip_records_{color_slug}.csv"
    print(f"[Route] 웹 선택 색상='{sel or 'ALL'}' → CSV='{CSV_GRIPS_PATH_dyn}'")

    # (4) 초기 홀드 검출
    ok, frame0 = cap.read()
    if not ok:
        print("[Init] 첫 프레임 실패"); return
    proc_size = (frame0.shape[1], frame0.shape[0])

    holds = initial_5frames_all_classes(cap, model, None,
                                        n_frames=5, mask_thresh=THRESH_MASK, merge_dist_px=18)
    if not holds:
        print("[Init] 홀드 검출 실패"); return

    # (5) 한 화면 클릭으로 타깃 홀드 선택
    idx = interactive_select_live_left_only(cap, holds, window=WINDOW_NAME)
    if not idx:
        print("[Select] 선택 없음"); return

    holds = [holds[i] for i in idx]
    # holds = assign_indices_row_major(holds, row_tol=ROW_TOL_Y)
    for new_id, h in enumerate(holds):
        h["hold_index"] = new_id
    for h in holds:
        if "mask" not in h:
            h["mask"] = _mask_from_contour((proc_size[1], proc_size[0]), h["contour"])

    hold_db = _build_hold_db_with_baseline(cap, proc_size, holds, n_frames=5, diff_gate=12.0)

    # (6) 레이저 원점(카메라 기준)
    L, O = compute_laser_origin_cam()

    # (7) 레이저 픽셀/3D → 기준각
    X_laser = None
    laser_px = None
    if laser_raw:
        # 1) 화면 시각화용 픽셀 좌표는 항상 먼저 세팅
        if laser_raw.get("left_raw"):
            lx, ly = laser_raw["left_raw"]
            laser_px = (int(lx), int(ly))

        # 2) 기준각/보정용 3D 좌표 세팅
        if laser_raw.get("X_laser_mm") is not None:
            X_laser = laser_raw["X_laser_mm"]  # mm, 이미 있음
        elif laser_px is not None:
            # 픽셀만 있을 땐 깊이로 역투영해서 3D 복원
            depth_m = cap.get_depth_meters()
            Zm = depth_median_at(depth_m, laser_px[0], laser_px[1], r=3)
            if Zm and Zm > 0:
                X_laser = cap.deproject(laser_px[0], laser_px[1], Zm) * 1000.0  # mm

    if X_laser is not None:
        yaw_laser0, pitch_laser0 = yaw_pitch_from_X(X_laser, O, Y_UP_IS_NEGATIVE)
    else:
        yaw_laser0, pitch_laser0 = 0.0, 0.0  # 폴백

    # (8) 모든 홀드 3D/각도 & 서보 타깃 생성(레이저 기준 적용)
    depth_m = cap.get_depth_meters()
    holds_3d = holds_to_3d(holds, depth_m, cap.deproject)
    matched_results = []
    for h in holds_3d:
        X = h["X"]
        yaw_deg, pitch_deg = yaw_pitch_from_X(X, O, Y_UP_IS_NEGATIVE)
        matched_results.append({
            "hid": h["hold_index"], "color": h["color"], "X": X,
            "d_left": float(np.linalg.norm(X - L)),
            "d_line": float(np.hypot(X[1], X[2])),
            "yaw_deg": yaw_deg, "pitch_deg": pitch_deg,
        })
    by_id = {mr["hid"]: mr for mr in matched_results}

    # (9) CSV 로드 → 경로
    route_pairs = load_route_pairs_from_csv(CSV_GRIPS_PATH_dyn)
    if not route_pairs:
        print("[Route] CSV 비어있음 — 종료"); return
    valid_hids = {h["hold_index"] for h in holds} & set(by_id.keys())
    route_pairs = [(p, hid) for (p, hid) in route_pairs if p and hid in valid_hids]
    if not route_pairs:
        print("[Route] 유효 (part,hold_id) 없음 — 종료"); return

    route_pos = 0
    current_target_part, current_target_id = route_pairs[0]

    # (10) 서보 준비 + 타깃 각도(레이저 기준) 계산
    ctl = DualServoController(args.port, args.baud) if HAS_SERVO else DualServoController()
    cur_yaw, cur_pitch = BASE_YAW_DEG, BASE_PITCH_DEG

    servo_targets = build_servo_targets(by_id, yaw_laser0, pitch_laser0, X_laser, O)

    # 첫 타깃으로 바로 조준하고 시작하고 싶으면:
    if current_target_id in by_id:
        ty, tp = servo_targets[current_target_id]
        send_servo_angles(ctl, ty, tp)
        cur_yaw, cur_pitch = ty, tp

    # (11) 루프 실행
    pose, blocked_state, out, t_prev, last_advanced_time = _event_loop(proc_size)
    _ = _run_frame_loop(
            cap, proc_size, holds, matched_results, servo_targets,
            (not args.no_auto_advance),
            pose, blocked_state, out, t_prev, last_advanced_time,
            current_target_id, cur_yaw, cur_pitch, ctl,
            route_pairs, route_pos, current_target_part, hold_db, laser_px=laser_px, yaw_laser0=yaw_laser0, pitch_laser0=pitch_laser0)

if __name__ == "__main__":
    main()