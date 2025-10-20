import time
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import csv
import argparse

# === MediaPipe 모듈 ===
from Climb_Mediapipe import PoseTracker, draw_pose_points, classify_occluder

# 레이저 찾기
from find_laser import capture_once_and_return

# hold 관련 코드
from A_hold_utils import initial_5frames_all_classes

# servo 관련 코드
from A_servo_utils import send_servo_angles, yaw_pitch_from_X

from A_click_select import interactive_select_both

from A_web import choose_color_via_web

# ========= 사용자 환경 경로 =========
NPZ_PATH       = r"./param/stereo_params_1024_576.npz"
MODEL_PATH     = r"./param/best_6.pt"

CAM1_INDEX     = 2   # 왼쪽 카메라
CAM2_INDEX     = 3   # 오른쪽 카메라

SWAP_DISPLAY   = False   # 화면 표시 좌/우 스와프

WINDOW_NAME    = "Rectified L | R"
THRESH_MASK    = 0.7
ROW_TOL_Y      = 30

# 자동 진행(터치→다음 홀드) 관련
TOUCH_THRESHOLD = 5     # in-polygon 연속 프레임 임계(기본 10)
ADV_COOLDOWN    = 0.5    # 연속 넘김 방지 쿨다운(sec)

# 저장 옵션
SAVE_VIDEO     = False
OUT_FPS        = 30
OUT_PATH       = "stereo_overlay.mp4"
CSV_GRIPS_PATH = "grip_records.csv"

# 런타임 보정 오프셋(레이저 실측)
CAL_YAW_OFFSET   = 0.0
CAL_PITCH_OFFSET = 0.0

# ---- 레이저 원점(LEFT 기준) 오프셋 (cm) ----
LASER_OFFSET_CM_LEFT = 1.15
LASER_OFFSET_CM_UP   = 5.2
LASER_OFFSET_CM_FWD  = -0.6
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

K_EXTRA_DEG = 3.0    # 최대 추가 피치 스케일(도)
H0_MM       = 1000.0 # 높이 정규화(1 m)
Z0_MM       = 4000.0 # 깊이 정규화(4 m에서 감쇠=1배)
BETA_Z      = 1.0    # 깊이 감쇠 기울기(0이면 감쇠 없음)
H_SOFT_MM  = 160.0   # 높이차가 이 정도 이하일 때 가산을 살짝 눌러줌
GAMMA_H    = 0.1     # 1.0~1.5 권장 (커지면 초반 더 세게 눌림)

# === 이미지 차분 파라미터 ===
DIFF_EVERY_N     = 10      # N프레임마다 차분(=부하 감소)
ADAPT_ALPHA      = 0.02   # 배경(베이스라인) 적응 속도(0=고정)
DILATE_KERNEL_SZ = 3      # 마스크 팽창 커널(노이즈↓/완충)

# === 강한 차분 기반 판정 파라미터(occlusion robust) ===
TH_DIFF_HARD   = 35        # 8-bit 강한 차분 임계(권장 30~45)
FRAC_HARD_MIN  = 0.30      # 강한 차분 픽셀 비율 최소(30% 이상이면 '크게 달라짐')
FRAC_DYN_MIN   = 0.55      # 동적 임계(노이즈 적응) 기준 비율
ERODE_ITERS    = 1         # 마스크 코어만 사용(경계 흔들림 억제). 0~1 권장

ROTATE_MAP = {
    2: cv2.ROTATE_90_COUNTERCLOCKWISE,  # LEFT
    3: cv2.ROTATE_90_CLOCKWISE,         # RIGHT
}

CAP_SIZE = (1024, 576)
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

def build_raw_projections_from_npz(npz_path):
    S = np.load(npz_path, allow_pickle=True)
    K1, D1 = S["K1"], S["D1"]
    K2, D2 = S["K2"], S["D2"]
    # stereoCalibrate 결과의 좌우 관계(왼→오)
    if "R" not in S or "T" not in S:
        raise RuntimeError("NPZ에 R,T가 필요합니다(레티파이 미사용 경로).")
    R, T = S["R"], S["T"].reshape(3,1)

    # P1 = K1 [I|0], P2 = K2 [R|T]
    P1 = K1 @ np.hstack([np.eye(3), np.zeros((3,1))])
    P2 = K2 @ np.hstack([R, T])
    return K1, D1, K2, D2, R, T, P1, P2

def undistort_points_px(pts_px, K, D):
    """ 입력/출력: 픽셀 좌표. 왜곡 보정 후 다시 픽셀 좌표로( P=K ). """
    pts = np.asarray(pts_px, dtype=np.float32).reshape(-1,1,2)
    und = cv2.undistortPoints(pts, K, D, P=K)  # → (N,1,2) in pixel domain
    return und.reshape(-1,2)

def triangulate_xy_raw(P1, P2, ptL_px, ptR_px, K1, D1, K2, D2):
    """레티파이 없이 직접 삼각측량(왜곡 보정만 수행)."""
    uL = undistort_points_px([ptL_px], K1, D1)[0]
    uR = undistort_points_px([ptR_px], K2, D2)[0]
    pL = np.array([[uL[0]],[uL[1]]], dtype=np.float64)
    pR = np.array([[uR[0]],[uR[1]]], dtype=np.float64)
    Xh = cv2.triangulatePoints(P1, P2, pL, pR)   # (4,1)
    X  = (Xh[:3,0] / Xh[3,0]).reshape(3)         # (3,)
    return X

def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM6")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--no_auto_advance", action="store_true")
    ap.add_argument("--no_web", action="store_true")
    ap.add_argument("--laser_on", action="store_true")
    return ap.parse_args()

def _verify_paths():
    for p in (NPZ_PATH, MODEL_PATH):
        if not Path(p).exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {p}")

def open_cams(idx1, idx2, size):
    W, H = size
    cap1 = cv2.VideoCapture(idx1, cv2.CAP_DSHOW)
    cap2 = cv2.VideoCapture(idx2, cv2.CAP_DSHOW)
    cap1.set(cv2.CAP_PROP_FRAME_WIDTH,  W); cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    cap2.set(cv2.CAP_PROP_FRAME_WIDTH,  W); cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    if not cap1.isOpened() or not cap2.isOpened():
        raise SystemExit("카메라 오픈 실패. 연결/권한 확인.")
    return cap1, cap2

def imshow_scaled(win, img, maxw=None):
    if not maxw:
        cv2.imshow(win, img)
        return
    h, w = img.shape[:2]
    if w > maxw:
        s = maxw / w
        img = cv2.resize(img, (int(w*s), int(h*s)))
    cv2.imshow(win, img)

def xoff_for(side, W, swap):
    return (W if swap else 0) if side=="L" else (0 if swap else W)

def wrap_deg(d):
    return (d + 180.0) % 360.0 - 180.0

def ypair_median_lock(ptL, ptR, clamp_h=None):
    """좌/우 픽셀 Y를 중앙값으로 강제 정렬한다.
    clamp_h가 주어지면 0..(clamp_h-1)로 클램핑."""
    xL, yL = int(ptL[0]), int(ptL[1])
    xR, yR = int(ptR[0]), int(ptR[1])
    y_med = int(round(np.median([yL, yR])))
    if clamp_h is not None:
        y_med = max(0, min(clamp_h - 1, y_med))
    return (xL, y_med), (xR, y_med)

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

    return K_EXTRA_DEG * height_term * depth_term

def _load_stereo_and_log():
    # 레티파이 맵/행렬 대신, 원시 투영행렬을 씀
    K1, D1, K2, D2, R, T, P1, P2 = build_raw_projections_from_npz(NPZ_PATH)
    # 이미지 사이즈는 카메라에서 얻는다(첫 프레임 사용)
    # main()에서 카메라 오픈 후 cap.get으로 확인
    print("[Info] (RAW) using P1=K1[I|0], P2=K2[R|T]")
    return K1, D1, K2, D2, R, T, P1, P2

def _capture_laser_raw(args):
    try:
        laser_raw = capture_once_and_return(
            port=args.port, baud=args.baud,
            wait_s=2.0, settle_n=8, show_preview=True,
            center_pitch=90.0, center_yaw=90.0, servo_settle_s=0.5, frame_size=(1024, 576),
        )
    except Exception as e:
        print(f"[A_Climbing] find_laser error: {e} → continue without laser")
        return None

    if laser_raw is None:
        print("[A_Climbing] 레이저 좌표 취득 실패(취소/에러). 계속 진행.")
        return None

    cam0_raw = laser_raw["cam0"]   # LEFT
    cam1_raw = laser_raw["cam1"]   # RIGHT
    print(f"[A_Climbing] 레이저(raw): L={cam0_raw}, R={cam1_raw}")
    return {"left_raw": cam0_raw, "right_raw": cam1_raw}

def compute_laser_origin_mid(T):
    L = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    M = (T.reshape(3) / 2.0)
    dx = -LASER_OFFSET_CM_LEFT * 10.0
    dy = (-1.0 if Y_UP_IS_NEGATIVE else 1.0) * LASER_OFFSET_CM_UP * 10.0
    dz = LASER_OFFSET_CM_FWD * 10.0
    O  = M + np.array([dx, dy, dz], dtype=np.float64)
    print(f"[Laser] Origin O (mm, MID-based) = {O}")
    return L, O

def _open_cameras_and_model(size):
    capL_idx, capR_idx = CAM1_INDEX, CAM2_INDEX
    cap1, cap2 = open_cams(capL_idx, capR_idx, size)
    model = YOLO(str(MODEL_PATH))
    return cap1, cap2, model

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

def _pair_and_assign_by_relpos(holdsL, selL, holdsR, selR, W, H, wx=1.0, wy=2.0, debug=False):
    """
    좌/우에서 '선택된' 홀드들만 가져와서, (x/W, y/H) 상대좌표 기반 1:1 최적 할당.
    - 비용: wy*|yL - yR| + wx*|xL_norm - xR_norm|
    - 결과 개수 = min(len(selL), len(selR))  (항상 1:1)
    - 각 쌍에 동일한 hold_index를 0..n-1 부여
    """
    # 1) 좌/우 선택 좌표 추출
    Lpts = []
    for i in selL:
        cx, cy = holdsL[i]["center"]
        Lpts.append((i, float(cx)/W, float(cy)/H))
    Rpts = []
    for j in selR:
        cx, cy = holdsR[j]["center"]
        Rpts.append((j, float(cx)/W, float(cy)/H))

    nL, nR = len(Lpts), len(Rpts)
    n = min(nL, nR)
    if n == 0:
        return [], []

    # 2) 비용 행렬 구성 (실제 y 픽셀 차보다 "상대 y"를 쓰고 싶으면 /H 하세요)
    #    여기서는 y도 0~1로 맞춰 스케일 편향 제거
    import math
    C = [[0.0]*nR for _ in range(nL)]
    for a, (_, xLn, yLn) in enumerate(Lpts):
        for b, (_, xRn, yRn) in enumerate(Rpts):
            cost = wy*abs(yLn - yRn) + wx*abs(xLn - xRn)
            C[a][b] = cost

    # 3) 헝가리안 (O(n^3)) — 외부 라이브러리 없이 경량 구현
    #    참고: n<=40 정도는 충분히 빠름
    def hungarian(cost):
        nL = len(cost)
        nR = len(cost[0]) if nL else 0
        n  = max(nL, nR)
        # 정사각으로 패딩
        BIG = 1e9
        M = [row[:] + [BIG]*(nR - len(row)) for row in cost] + [[BIG]*max(nR,n)]*(n - nL)
        # u, v potentials
        u = [0.0]*(n+1)
        v = [0.0]*(n+1)
        p = [0]*(n+1)
        way = [0]*(n+1)
        for i in range(1, n+1):
            p[0] = i
            j0 = 0
            minv = [float('inf')]*(n+1)
            used = [False]*(n+1)
            while True:
                used[j0] = True
                i0 = p[j0]
                delta = float('inf')
                j1 = 0
                for j in range(1, n+1):
                    if not used[j]:
                        cur = M[i0-1][j-1] - u[i0] - v[j]
                        if cur < minv[j]:
                            minv[j] = cur
                            way[j] = j0
                        if minv[j] < delta:
                            delta = minv[j]
                            j1 = j
                for j in range(0, n+1):
                    if used[j]:
                        u[p[j]] += delta
                        v[j] -= delta
                    else:
                        minv[j] -= delta
                j0 = j1
                if p[j0] == 0:
                    break
            while True:
                j1 = way[j0]
                p[j0] = p[j1]
                j0 = j1
                if j0 == 0:
                    break
        # p[j] = i  (i는 1..n) → (i-1) ↔ (j-1) 매칭
        matchLR = [-1]*n
        for j in range(1, n+1):
            if p[j] != 0:
                matchLR[j-1] = p[j]-1
        # 원래 크기로 절단
        pairs = []
        for j in range(min(nR, n)):
            i = matchLR[j]
            if 0 <= i < nL:
                pairs.append((i, j))
        return pairs

    pairs_idx = hungarian(C)

    # 4) 상위 n개만 사용 (비용 낮은 것부터) — 이미 최적이지만, nL!=nR인 경우 n개로 컷
    #    pairs_idx는 (iL_idx_in_sel, iR_idx_in_sel)
    #    sel 인덱스를 holds 인덱스로 변환
    #    정렬은 보기 좋게 좌측 y오름차순으로
    matched = []
    for i_sel, j_sel in pairs_idx:
        idxL = selL[i_sel]
        idxR = selR[j_sel]
        matched.append((idxL, idxR))
    # 좌측 y 기준 정렬(선택)
    matched.sort(key=lambda t: holdsL[t[0]]["center"][1])

    # 5) hold_index 부여 & 출력 리스트 구성
    L_out, R_out = [], []
    hid = 0
    for iL, iR in matched[:n]:
        holdsL[iL]["hold_index"] = hid
        holdsR[iR]["hold_index"] = hid
        L_out.append(holdsL[iL])
        R_out.append(holdsR[iR])
        hid += 1

    if debug:
        print(f"[Pair-RelPos] L_sel={len(selL)}, R_sel={len(selR)} → matched={len(L_out)}")
    return L_out, R_out

def _build_common_ids(holdsL, holdsR):
    idxL = {h["hold_index"]: h for h in holdsL}
    idxR = {h["hold_index"]: h for h in holdsR}
    common_ids = sorted(set(idxL.keys()) & set(idxR.keys()))
    if not common_ids:
        print("[Warn] 좌/우 공통 hold_index가 없습니다.")
        return idxL, idxR, []
    print(f"[Info] 공통 홀드 개수: {len(common_ids)}")
    return idxL, idxR, common_ids

def _compute_matched_results_raw(common_ids, idxL, idxR, P1, P2, K1, D1, K2, D2, L, O):
    matched_results = []
    for hid in common_ids:
        Lh = idxL[hid]; Rh = idxR[hid]
        # ★ Y 중앙값 보정
        ptL_corr, ptR_corr = ypair_median_lock(Lh["center"], Rh["center"])
        X = triangulate_xy_raw(P1, P2, ptL_corr, ptR_corr, K1, D1, K2, D2)
        d_left  = float(np.linalg.norm(X - L))
        d_line  = float(np.hypot(X[1], X[2]))
        yaw_deg, pitch_deg = yaw_pitch_from_X(X, O, Y_UP_IS_NEGATIVE)
        matched_results.append({
            "hid": hid, "color": Lh["color"],
            "X": X, "d_left": d_left, "d_line": d_line,
            "yaw_deg": yaw_deg, "pitch_deg": pitch_deg,
        })
    return matched_results

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

def _build_hold_db_with_baseline(capL, size, holdsL, n_frames=10, diff_gate=12.0):
    W, H = size
    # 1) 기본 구조
    db = {}
    for h in holdsL:
        x,y,w,hh = cv2.boundingRect(h["contour"])
        db[h["hold_index"]] = {
            "mask": _mask_from_contour((H,W), h["contour"]),
            "bbox": (x,y,w,hh),
            "center": tuple(h["center"]),
            "samples": [],  # 임시 저장
        }

    # 2) 초기 프레임 모으기
    frames = []
    for _ in range(n_frames):
        ok, f = capL.read()
        if not ok: continue
        f = rotate_image(f, ROTATE_MAP.get(CAM1_INDEX))
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    if not frames:
        raise SystemExit("[Diff] 초기 베이스라인 프레임을 하나도 못 얻었습니다.")
        
    # 3) 홀드별로 안정 프레임만 골라 median
    for hid, info in db.items():
        x,y,w,hh = info["bbox"]
        x = max(0, min(x, W-1)); y = max(0, min(y, H-1))
        w = max(1, min(w, W - x)); hh = max(1, min(hh, H - y))
        # 각 프레임에서 같은 위치 crop
        crops = [g[y:y+hh, x:x+w] for g in frames]
        if not crops:
            continue
        # 프레임 간 간단한 안정성 필터(프레임 t vs t-1 평균차)
        keep = []
        ref = crops[0]
        for c in crops:
            mean_diff = cv2.mean(cv2.absdiff(ref, c))[0]
            if mean_diff < diff_gate:  # 너무 흔들리는/가려진 프레임 제외
                keep.append(c)
        use = keep if len(keep) >= 3 else crops  # 최소 보장
        baseline = np.median(np.stack(use, axis=0), axis=0).astype(np.uint8)
        info["baseline"] = baseline
        info.pop("samples", None)
    return db

def _triangulate_laser_3d_raw(laser_px, P1, P2, K1, D1, K2, D2):
    if not laser_px: return None
    Lp = laser_px.get("left_raw"); Rp = laser_px.get("right_raw")
    if (Lp is None) or (Rp is None): return None
    # ★ Y 중앙값 보정
    Lp_corr, Rp_corr = ypair_median_lock(Lp, Rp)
    return triangulate_xy_raw(P1, P2, Lp_corr, Rp_corr, K1, D1, K2, D2)

def servo_cmd_from_laser_ref(yaw_hold, pitch_hold, yaw_laser0, pitch_laser0):
    d_yaw   = wrap_deg(yaw_hold  - yaw_laser0) + CAL_YAW_OFFSET
    d_pitch = wrap_deg(pitch_hold - pitch_laser0) + CAL_PITCH_OFFSET

    target_yaw   = BASE_YAW_DEG   + YAW_SIGN   * (YAW_SCALE   * d_yaw)
    target_pitch = BASE_PITCH_DEG + PITCH_SIGN * (PITCH_SCALE * d_pitch)
    target_yaw   = max(0.0, min(180.0, target_yaw))
    target_pitch = max(0.0, min(180.0, target_pitch))
    return target_yaw, target_pitch

def _init_servo_and_point_first(ctl, args, current_target_id, by_id, X_laser, O,
                                cur_yaw, cur_pitch):
    global YAW_LASER0, PITCH_LASER0

    # 서보를 90/90으로 일단 세팅
    try:
        ctl.set_angles(cur_pitch, cur_yaw)  # (pitch, yaw)
        try:
            if args.laser_on:
                ctl.laser_on()
        except:
            pass
    except Exception as e:
        print("[Init] Servo init error:", e)

    if (current_target_id is None) or (current_target_id not in by_id) or (X_laser is None):
        print("[Init] 레이저 3D 또는 첫 타깃 없음 → 폴백 초기 조준 사용")
        if current_target_id is not None:
            mr0 = by_id[current_target_id]
            yaw_hold0, pitch_hold0 = mr0["yaw_deg"], mr0["pitch_deg"]
            # 기준 레이저 각이 없으면 레이저 대신 90/90을 기준처럼 취급(폴백)
            yaw_l0 = 0.0 if YAW_LASER0 is None else YAW_LASER0
            pitch_l0 = 0.0 if PITCH_LASER0 is None else PITCH_LASER0
            yaw_cmd0, pitch_cmd0 = servo_cmd_from_laser_ref(yaw_hold0, pitch_hold0, yaw_l0, pitch_l0)
            try:
                ctl.set_angles(pitch_cmd0, yaw_cmd0)
                cur_yaw, cur_pitch = yaw_cmd0, pitch_cmd0
            except Exception as e:
                print("[Init-Point Fallback] Servo move error:", e)
        return cur_yaw, cur_pitch

    # === 기준(레이저0) 각 계산 & 저장 ===
    yaw_laser0,  pitch_laser0  = yaw_pitch_from_X(X_laser, O, Y_UP_IS_NEGATIVE)
    YAW_LASER0, PITCH_LASER0 = yaw_laser0, pitch_laser0

    # 첫 타깃으로 기준 기반 이동
    X_hold = by_id[current_target_id]["X"]
    yaw_hold0, pitch_hold0 = yaw_pitch_from_X(X_hold, O, Y_UP_IS_NEGATIVE)
    target_yaw, target_pitch = servo_cmd_from_laser_ref(
        yaw_hold0, pitch_hold0, YAW_LASER0, PITCH_LASER0
    )
    print(f"[Init-Target@Ref] laser0=({yaw_laser0:.2f},{pitch_laser0:.2f})°, "
          f"hold0=({yaw_hold0:.2f},{pitch_hold0:.2f})°  "
          f"→ servo Y/P=({target_yaw:.2f},{target_pitch:.2f})")

    try:
        ctl.set_angles(target_pitch, target_yaw)  # (pitch, yaw)
        cur_yaw, cur_pitch = target_yaw, target_pitch
    except Exception as e:
        print("[Init-Target] Servo move error:", e)

    return cur_yaw, cur_pitch

def _event_loop( size):
    W, H = size
    pose = PoseTracker(min_detection_confidence=0.5, model_complexity=1)
    blocked_state = {}   # (part, hold_id)별 차폐 상태

    out = None
    if SAVE_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(OUT_PATH, fourcc, OUT_FPS, (W*2, H))

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    t_prev = time.time()
    last_advanced_time = 0.0

    # 외부에서 현재 타깃, 서보 각도는 관리하므로 반환 값으로 전달
    return pose, blocked_state, out, t_prev, last_advanced_time

def build_servo_targets(by_id, yaw_laser0, pitch_laser0, X_laser, O):
    servo_targets = {}
    for hid, mr in by_id.items():
        yaw_h, pitch_h = mr["yaw_deg"], mr["pitch_deg"]
        ty, tp = servo_cmd_from_laser_ref(yaw_h, pitch_h, yaw_laser0, pitch_laser0)

        # ★ Pitch만 Z깊이 감쇠로 가중치 적용
        tp += extra_pitch_deg(mr.get("X"), O, X_laser, y_up_is_negative=Y_UP_IS_NEGATIVE)

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

def _run_frame_loop(cap1, cap2, size,
                    SWAP_DISPLAY, laser_px,
                    holdsL, holdsR, matched_results,
                    by_id, servo_targets,
                    auto_advance_enabled,
                    pose, blocked_state,
                    out, t_prev, last_advanced_time,
                    current_target_id, cur_yaw, cur_pitch, ctl,
                    route_pairs, route_pos, current_target_part,
                    hold_db):
    W, H = size
    touch_streak = {}  # (part_name, hold_id) -> 연속 프레임 수
    frame_id = 0
    prev_block_part = None

    holdsL_by_id = {h["hold_index"]: h for h in holdsL}

    try:
        while True:
            ok1, f1 = cap1.read(); ok2, f2 = cap2.read()
            if not (ok1 and ok2):
                print("[Warn] 프레임 캡쳐 실패"); break

            Limg = rotate_image(f1, ROTATE_MAP.get(CAM1_INDEX))
            Rimg = rotate_image(f2, ROTATE_MAP.get(CAM2_INDEX))

            # (옵션) 레이저 점 시각 확인
            if laser_px:
                if laser_px.get("left_raw") is not None:
                    lx, ly = laser_px["left_raw"]
                    cv2.circle(Limg, (lx, ly), 8, (0,0,255), 2, cv2.LINE_AA)
                if laser_px.get("right_raw") is not None:
                    rx, ry = laser_px["right_raw"]
                    cv2.circle(Rimg, (rx, ry), 8, (0,0,255), 2, cv2.LINE_AA)


            vis = np.hstack([Rimg, Limg]) if SWAP_DISPLAY else np.hstack([Limg, Rimg])

            for side, holds in (("L", holdsL), ("R", holdsR)):
                xoff = xoff_for(side, W, SWAP_DISPLAY)
                for h in holds:
                    cnt_shifted = h["contour"] + np.array([[[xoff, 0]]], dtype=h["contour"].dtype)
                    cx, cy = h["center"]

                    # 외곽선/라벨 표시
                    cv2.drawContours(vis, [cnt_shifted], -1, h["color"], 2)
                    cv2.circle(vis, (cx+xoff, cy), 4, (255,255,255), -1)
                    tag = f"ID:{h['hold_index']}"
                    if h["hold_index"] == current_target_id:
                        tag = "[TARGET] " + tag
                    cv2.putText(vis, tag, (cx + xoff - 10, cy + 26),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
                    cv2.putText(vis, tag, (cx + xoff - 10, cy + 26),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, h["color"], 2, cv2.LINE_AA)

            # --- 디버그 3D 좌표/깊이 표시 ---
            y_info = 60
            for mr in matched_results:
                X = mr["X"]
                depth = X[2]
                txt3d = (f"ID{mr['hid']} : X=({X[0]:.1f}, {X[1]:.1f}, {X[2]:.1f}) mm "
                         f" | depth(Z)={depth:.1f} mm")
                cv2.putText(vis, txt3d, (20, y_info),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2, cv2.LINE_AA)
                cv2.putText(vis, txt3d, (20, y_info),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
                y_info += 18

            # NEXT 텍스트 및 현재 각도 표시
            y0 = 28
            txt = ""
            if current_target_id in by_id:
                # route_pos가 현재 타깃을 가리킨다고 가정
                if route_pos < len(route_pairs) - 1:
                    next_part, next_hid = route_pairs[route_pos + 1]
                    ty, tp = servo_targets.get(next_hid, (None, None))
                    if ty is not None:
                        txt = (f"[NEXT] ({current_target_part}) {current_target_id} → "
                            f"({next_part}) {next_hid}  "
                            f"target_servo(Y/P)=({ty:.1f},{tp:.1f})°  "
                            f"[now Y/P=({cur_yaw:.1f},{cur_pitch:.1f})°]")
                    else:
                        txt = (f"[NEXT] ({current_target_part}) {current_target_id} → "
                            f"({next_part}) {next_hid}  (target pending)  "
                            f"[now Y/P=({cur_yaw:.1f},{cur_pitch:.1f})°]")
                else:
                    mr = by_id[current_target_id]
                    txt = (f"[LAST] ({current_target_part}) ID{mr['hid']}  "
                        f"hold_abs(Y/P)=({mr['yaw_deg']:.1f},{mr['pitch_deg']:.1f})°  "
                        f"[now Y/P=({cur_yaw:.1f},{cur_pitch:.1f})°]")
            cv2.putText(vis, txt, (20, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(vis, txt, (20, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,255), 1, cv2.LINE_AA)

            # === MediaPipe 포즈 추정 & 표시 ===
            coords = pose.process(Limg)
            left_xoff = xoff_for("L", W, SWAP_DISPLAY)
            draw_pose_points(vis, coords, offset_x=left_xoff)

            # === (NEW) 타깃 홀드 이미지 차분 + 최근접 관절 판단 (N프레임마다) ===
            if frame_id % DIFF_EVERY_N == 0 and current_target_id in hold_db:
                info = hold_db[current_target_id]
                x,y,w,hh = info["bbox"]
                grayL = cv2.cvtColor(Limg, cv2.COLOR_BGR2GRAY)
                patch = grayL[y:y+hh, x:x+w]
                base  = info["baseline"]

                mroi_full = info["mask"][y:y+hh, x:x+w]  # 딱 해당 홀드만
                mroi_core = mroi_full
                if ERODE_ITERS > 0:
                    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
                    mroi_core = cv2.erode(mroi_full, k, iterations=ERODE_ITERS)

                # 절대차(현재 프레임 vs 베이스라인)
                diff = cv2.absdiff(base, patch)
                # --- 통계 벡터 추출(ROI 코어만) ---
                roi_idx = (mroi_core > 0)
                vals = diff[roi_idx]
                if vals.size == 0:
                    is_motion = False
                else:
                    # Robust scale (MAD)로 동적 임계 만들기
                    med = float(np.median(vals))
                    mad = float(np.median(np.abs(vals - med))) + 1e-6
                    sigma = 1.4826 * mad
                    tau_dyn = max(TH_DIFF_HARD * 0.6, 3.0 * sigma)  # 너무 낮아지지 않게 바닥 보장

                    # "센 변화" 픽셀 비율 (완전 가리면 여기서 확 튄다)
                    frac_hard = float((vals >= TH_DIFF_HARD).mean())
                    frac_dyn  = float((vals >= tau_dyn).mean())

                    # 최종 모션 판정: 강한 변화 OR 동적 임계 초과
                    is_motion = (frac_hard >= FRAC_HARD_MIN) or (frac_dyn >= FRAC_DYN_MIN)

                # 최근접 관절
                center = info["center"]

                # 안전한 기본값
                is_block = False
                block_part = None

                # 최종 판정은 classify_occluder로 보정(덮어쓰기)
                if is_motion:
                    label, who = classify_occluder(
                        center_xy=center,
                        coords=coords,
                        shape_hw=(H, W),
                        grip_parts=pose.grip_parts,
                        hold_mask=info.get("mask", None)
                    )
                    is_block   = (label != "grip")
                    block_part = (who if is_block else None)

                    # === [강제 규칙 1] 타깃 그립이 홀드 안에 들어왔으면 무조건 'grip' ===
                    try:
                        # 타깃 파트 좌표
                        if coords and (current_target_part in coords):
                            gx, gy = coords[current_target_part]
                            ix, iy = int(round(gx)), int(round(gy))
                            if 0 <= ix < W and 0 <= iy < H:
                                # 코어 마스크 기준(경계 오탐 줄이기) — 없으면 full 마스크 사용
                                mcore = info.get("mask_core", None)
                                if mcore is None:
                                    mcore = info["mask"]
                                if mcore[iy, ix] > 0:
                                    # 타깃 파트가 홀드 내부 → 가림 아님 = 잡음
                                    label = "grip"
                                    is_block = False
                                    block_part = None
                    except Exception:
                        pass

                    # === [강제 규칙 2] classify가 who를 돌려줬고 그게 grip_parts에 해당하면 'grip' ===
                    if is_block and who:
                        # 정확 일치 + 느슨한 포함 둘 다 허용 (이름 표기 불일치 대비)
                        def _is_grip_name(nm: str) -> bool:
                            if not nm: return False
                            if nm in getattr(pose, "grip_parts", set()):
                                return True
                            # 예: 'right_index_tip' vs 'right_index' 등 유사명 처리
                            return any(nm.startswith(g) or g in nm for g in getattr(pose, "grip_parts", set()))
                        if _is_grip_name(who):
                            label = "grip"
                            is_block = False
                            block_part = None

                # 베이스라인 업데이트(가려진 동안엔 금지)
                if ADAPT_ALPHA > 0 and is_motion and not is_block:
                    base_f = base.astype(np.float32)
                    cv2.accumulateWeighted(patch.astype(np.float32), base_f, ADAPT_ALPHA, mask=mroi_core)
                    info["baseline"] = base_f.astype(np.uint8)

                # 로그: 같은 부위 연속이면 생략
                if is_block:
                    part_to_report = block_part or "unknown"
                    if part_to_report != prev_block_part:
                        print(f"[BLOCKED] 홀드 ID {current_target_id} — {part_to_report} 부위가 가림")
                        prev_block_part = part_to_report
                else:
                    prev_block_part = None
                                    
            # === (핵심) 목표 (part, hold_id)만 판정 ===
            if coords and (current_target_id in {h["hold_index"] for h in holdsL}):
                tid  = current_target_id
                tpart_csv = current_target_part
                hold = holdsL_by_id.get(tid)

                advanced_this_frame = False
                current_touched = set()

                # ▼ part 이름 정규화
                tpart = _resolve_part_name(tpart_csv, coords, getattr(pose, "grip_parts", set()))

                if hold is None:
                    cv2.putText(vis, f"[WARN] hold_id {tid} not on LEFT", (20, 64),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)
                    cv2.putText(vis, f"[WARN] hold_id {tid} not on LEFT", (20, 64),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,140,255), 1, cv2.LINE_AA)
                else:
                    if tpart is None:
                        # 좌표 키가 없으면 이번 프레임 스킵(눈에 보이게 경고)
                        cv2.putText(vis, f"[WARN] part '{tpart_csv}' not in coords", (20, 46),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)
                        cv2.putText(vis, f"[WARN] part '{tpart_csv}' not in coords", (20, 46),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,140,255), 1, cv2.LINE_AA)
                    else:
                        x, y = coords[tpart]
                        inside = cv2.pointPolygonTest(hold["contour"], (x, y), False) >= 0
                        key = (tpart, tid)
                        if inside:
                            current_touched.add(key)

                        # blocking 로직 유지
                        if inside and (tpart in pose.blocking_parts):
                            if not blocked_state.get(key, False):
                                blocked_state[key] = True
                        else:
                            blocked_state[key] = False

                    # 스트릭 진행상황 HUD (디버깅용)
                    cnt = touch_streak.get((tpart or tpart_csv, tid), 0)
                    cv2.putText(vis, f"[GRIP] {tpart or tpart_csv}@{tid} {cnt}/{TOUCH_THRESHOLD}",
                                (20, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,0), 3, cv2.LINE_AA)
                    cv2.putText(vis, f"[GRIP] {tpart or tpart_csv}@{tid} {cnt}/{TOUCH_THRESHOLD}",
                                (20, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50,220,50), 2, cv2.LINE_AA)

                    # 스트릭 갱신/체크
                    for key in current_touched:
                        touch_streak[key] = touch_streak.get(key, 0) + 1

                        if touch_streak[key] >= TOUCH_THRESHOLD:
                            now_t = time.time()
                            if (auto_advance_enabled
                                and (route_pos < len(route_pairs) - 1)
                                and (now_t - last_advanced_time) > ADV_COOLDOWN
                                and not advanced_this_frame):

                                next_part, next_tid = route_pairs[route_pos + 1]

                                # 서보 타깃 방어적 접근
                                ty_tp = servo_targets.get(next_tid)
                                if ty_tp is None:
                                    print(f"[Route] servo target missing for id {next_tid} — skip advance this frame")
                                else:
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

                    # 이번 프레임에 안 닿은 키 리셋
                    for key in list(touch_streak.keys()):
                        if key not in current_touched:
                            touch_streak[key] = 0

            # FPS
            t_now = time.time()
            fps = 1.0 / max(t_now - (t_prev), 1e-6); t_prev = t_now
            cv2.putText(vis, f"FPS: {fps:.1f} (Auto={'ON' if auto_advance_enabled else 'OFF'})",
                        (10, H-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)
            cv2.putText(vis, f"FPS: {fps:.1f} (Auto={'ON' if auto_advance_enabled else 'OFF'})",
                        (10, H-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1, cv2.LINE_AA)

            imshow_scaled(WINDOW_NAME, vis, None)
            if SAVE_VIDEO:
                if out is None:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(OUT_PATH, fourcc, OUT_FPS, (W*2, H))
                out.write(vis)

            frame_id += 1
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break

    finally:
        cap1.release(); cap2.release()
        if SAVE_VIDEO and out is not None:
            out.release(); print(f"[Info] 저장 완료: {OUT_PATH}")
        cv2.destroyAllWindows()
        try: pose.close()
        except: pass
        try: ctl.close()
        except: pass

    # 상태 업데이트 반환(필요시 확장용)
    return current_target_id, cur_yaw, cur_pitch

# ---------- 메인 ----------
def main():
    args = _parse_args()

    # 경로 검증
    _verify_paths()

    # 스테레오 로드
    K1, D1, K2, D2, R, T, P1, P2 = _load_stereo_and_log()

    # 레이저 좌표 먼저 측정 (find_laser) → 보정좌표로 변환
    laser_px = _capture_laser_raw(args)

    L, O = compute_laser_origin_mid(T)

    # 웹 스와치에서 색상 선택 ("" = 전체)
    sel = choose_color_via_web()   # "green" / "red" / ... / ""(전체)

    # 파일명용 슬러그: ""이면 "all"로 저장
    color_slug = sel if sel else "all"
    CSV_GRIPS_PATH_dyn = f"grip_records_{color_slug}.csv"
    print(f"[Route] 웹 선택 색상='{sel or 'ALL'}' → CSV='{CSV_GRIPS_PATH_dyn}'")

    # 카메라 & 모델
    cap1, cap2, model = _open_cameras_and_model(CAP_SIZE)

    # ★ 회전 후 처리 해상도 확정
    _ , f1_raw = cap1.read(); _ , f2_raw = cap2.read()
    f1r = rotate_image(f1_raw, ROTATE_MAP.get(CAM1_INDEX))
    H, W = f1r.shape[:2]
    proc_size = (W, H)

    # 초기 5프레임: YOLO seg & merge
    # 1) 모든 클래스 5프레임 세그 + 병합(인덱스X)
    holdsL = initial_5frames_all_classes(cap1, model, ROTATE_MAP.get(CAM1_INDEX),
                                        n_frames=5, mask_thresh=THRESH_MASK, merge_dist_px=18)
    holdsR = initial_5frames_all_classes(cap2, model, ROTATE_MAP.get(CAM2_INDEX),
                                        n_frames=5, mask_thresh=THRESH_MASK, merge_dist_px=18)
    if not holdsL or not holdsR:
        print("[Init] 홀드를 검출하지 못했습니다.")
        return

    # 2) 한 프레임씩 캡처(회전 포함) → 합성 화면에서 양쪽 클릭 선택
    okL, frameL = cap1.read()
    okR, frameR = cap2.read()
    if not (okL and okR):
        print("[Init] 프레임 캡처 실패"); return
    Limg_now = rotate_image(frameL, ROTATE_MAP.get(CAM1_INDEX))
    Rimg_now = rotate_image(frameR, ROTATE_MAP.get(CAM2_INDEX))

    idxL, idxR = interactive_select_both(Limg_now, Rimg_now, holdsL, holdsR, swap_display=SWAP_DISPLAY,
                                        window="Select Holds (L|R)")
    if not idxL or not idxR:
        print("[Select] 양쪽 선택이 필요합니다(선택 없음)."); return

    holdsL, holdsR = _pair_and_assign_by_relpos(
    holdsL, idxL, holdsR, idxR, W=proc_size[0], H=proc_size[1],
    wx=1.0, wy=2.0, debug=True)

    hold_db = _build_hold_db_with_baseline(cap1, proc_size, holdsL, n_frames=5, diff_gate=12.0)

    # 공통 ID
    idxL, idxR, common_ids = _build_common_ids(holdsL, holdsR)
    if not common_ids:
        return

    # 3D/각도 계산
    matched_results = _compute_matched_results_raw(common_ids, idxL, idxR, P1, P2, K1, D1, K2, D2, L, O)

    # Δ 테이블 (CSV 순서 기반)
    by_id  = {mr["hid"]: mr for mr in matched_results}
    # ✅ CSV에서 (part, hold_id) 순서대로 읽기
    route_pairs = load_route_pairs_from_csv(CSV_GRIPS_PATH_dyn)
    if not route_pairs:
        print("[Route] CSV 경로가 비었거나 없음 — (part, hold_id) 기반 진행 불가")
        return

    # route 유효성 필터: 실제 공통 ID(=by_id.keys) & LEFT에 존재하는 것만
    valid_hids = set({h["hold_index"] for h in holdsL}) & set(by_id.keys())
    orig_n = len(route_pairs)
    route_pairs = [(p, hid) for (p, hid) in route_pairs
                if (p is not None and p != "") and (hid in valid_hids)]
    if len(route_pairs) < orig_n:
        print(f"[Route] CSV {orig_n} → 유효 {len(route_pairs)} (없는 id/빈 part 제거)")
    if not route_pairs:
        print("[Route] 유효 (part,hold_id) 없음 — 종료")
        return

    # 시작 포인터/타깃
    route_pos = 0
    current_target_part, current_target_id = route_pairs[0]

    # Servo 초기화 & '레이저→첫 홀드' Δ각 기반 초기 조준
    ctl = DualServoController(args.port, args.baud) if HAS_SERVO else DualServoController()

    # 2) 레이저 3D
    X_laser = _triangulate_laser_3d_raw(laser_px, P1, P2, K1, D1, K2, D2)

    # 3) 서보는 중립 90/90에서 시작(규약), Δ각을 적용해 첫 홀드로
    cur_yaw, cur_pitch = BASE_YAW_DEG, BASE_PITCH_DEG
    cur_yaw, cur_pitch = _init_servo_and_point_first(ctl, args, current_target_id, by_id, X_laser, O, cur_yaw, cur_pitch)

    servo_targets = build_servo_targets(by_id, YAW_LASER0, PITCH_LASER0, X_laser, O)

    # 루프 준비 객체 생성 (상태는 아래 프레임 루프에서 유지/갱신)
    pose, blocked_state, out, t_prev, last_advanced_time = _event_loop(proc_size)

    # 본 루프 실행
    _ = _run_frame_loop(cap1, cap2, proc_size,
                    SWAP_DISPLAY, laser_px,
                    holdsL, holdsR, matched_results,
                    by_id, servo_targets,
                    (not args.no_auto_advance),
                    pose, blocked_state,
                    out, t_prev, last_advanced_time,
                    current_target_id, cur_yaw, cur_pitch, ctl,
                    route_pairs, route_pos, current_target_part,
                    hold_db)

if __name__ == "__main__":
    main()
