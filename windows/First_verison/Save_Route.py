#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import csv
import argparse

# === MediaPipe ===
from Climb_Mediapipe import PoseTracker, draw_pose_points

# === 웹 색상 선택 (파일명 라벨만; 감지는 항상 전체) ===
_USE_WEB = True
try:
    from web import choose_color_via_web
except Exception:
    _USE_WEB = False
    def choose_color_via_web(*a, **k):
        raise RuntimeError("A_web 모듈이 로드되지 않았습니다.")

# === 클릭 선택(단일 화면) ===
from click_select import interactive_select_live_left_only

# === RealSense 어댑터(컬러만 사용, depth는 무시) ===
from realsense_adapter import RealSenseColorDepth

# === 홀드 유틸 ===
from hold_utils import initial_5frames_all_classes, assign_indices

# ========= 사용자 경로 =========
MODEL_PATH     = r"C:\Users\jshkr\OneDrive\문서\JSH_CAPSTONE_CODE\windows\param\best_6.pt"

# ========= 런타임 파라미터 =========
WINDOW_NAME    = "SaveRoute (D455 color only)"
THRESH_MASK    = 0.7
ROW_TOL_Y      = 1
SELECTED_COLOR = None    # 예) "orange" (None=전체). ← 파일명 라벨만!

# 터치 판정(프레임 기반)
TOUCH_THRESHOLD = 10

# ==== 발 시각화/판정 옵션 ====
SHOW_FEET      = True
FOOT_ALPHA     = 0.35
FOOT_OUTLINE   = 2
FOOT_MIN_R_PX  = 12
FOOT_R_SCALE   = 0.22
FOOT_COLOR     = (0, 255, 255)

# ==== 색상 맵(시각화용) ====
COLOR_MAP = {
    'Hold_Red':(0,0,255),'Hold_Orange':(0,165,255),'Hold_Yellow':(0,255,255),
    'Hold_Green':(0,255,0),'Hold_Blue':(255,0,0),'Hold_Purple':(204,50,153),
    'Hold_Pink':(203,192,255),'Hold_Lime':(50,255,128),'Hold_Sky':(255,255,0),
    'Hold_White':(255,255,255),'Hold_Black':(30,30,30),'Hold_Gray':(150,150,150),
}
ALL_COLORS = {
    'red':'Hold_Red','orange':'Hold_Orange','yellow':'Hold_Yellow','green':'Hold_Green',
    'blue':'Hold_Blue','purple':'Hold_Purple','pink':'Hold_Pink','white':'Hold_White',
    'black':'Hold_Black','gray':'Hold_Gray','lime':'Hold_Lime','sky':'Hold_Sky',
}

def ask_color_and_map_to_class(all_colors_dict):
    print("가능한 색상:", ", ".join(all_colors_dict.keys()))
    s = input("CSV 파일명에 사용할 색상 라벨 입력(엔터=all): ").strip().lower()
    if not s:
        print("→ 파일 라벨 'all' 사용")
        return "all"
    if s not in all_colors_dict:
        print(f"입력 '{s}' 은(는) 유효하지 않은 색입니다. 'all' 사용")
        return "all"
    print(f"선택된 라벨: {s}")
    return s

def _sanitize_label(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum() or ch in ("_", "-"))

# ========== 발(좌/우) 마스크 ==========
def _draw_capsule(mask, p1, p2, r_px):
    r = int(round(float(r_px)))
    if r <= 0: return
    p1 = tuple(map(int, map(round, p1)))
    p2 = tuple(map(int, map(round, p2)))
    cv2.line(mask, p1, p2, 255, thickness=2*r, lineType=cv2.LINE_AA)
    cv2.circle(mask, p1, r, 255, -1, cv2.LINE_AA)
    cv2.circle(mask, p2, r, 255, -1, cv2.LINE_AA)

def build_foot_masks_sided(coords, shape_hw):
    H, W = shape_hw
    masks = {"left_foot": np.zeros((H,W), np.uint8),
             "right_foot": np.zeros((H,W), np.uint8)}
    # left
    if "left_heel" in coords and "left_foot_index" in coords:
        A = coords["left_heel"]; B = coords["left_foot_index"]
        L = float(np.hypot(B[0]-A[0], B[1]-A[1]))
        r = max(FOOT_MIN_R_PX, int(round(FOOT_R_SCALE * L)))
        _draw_capsule(masks["left_foot"], A, B, r)
    # right
    if "right_heel" in coords and "right_foot_index" in coords:
        A = coords["right_heel"]; B = coords["right_foot_index"]
        L = float(np.hypot(B[0]-A[0], B[1]-A[1]))
        r = max(FOOT_MIN_R_PX, int(round(FOOT_R_SCALE * L)))
        _draw_capsule(masks["right_foot"], A, B, r)
    return masks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no_web", action="store_true", help="웹 색상 선택 비활성화(콘솔 입력)")
    ap.add_argument("--rotate90", action="store_true", help="컬러 프레임을 시계 90° 회전 (어댑터 내부 처리)")
    ap.add_argument("--csv_prefix", default="grip_records", help="CSV 파일 접두사")
    args = ap.parse_args()

    # 경로 검증
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {MODEL_PATH}")

    # ================= CSV 색상 라벨(파일명 용도만) =================
    selected_color_label = "all"   # 파일명 라벨

    # 1) 웹 선택 (라벨만)
    if (not args.no_web) and _USE_WEB:
        try:
            chosen = choose_color_via_web(
                all_colors=list(ALL_COLORS.keys()),
                defaults={}
            )  # ""이면 전체
            if chosen:
                if chosen.lower() in ALL_COLORS:
                    print(f"[Label] 웹 선택: {chosen}")
                    selected_color_label = chosen.lower()
                else:
                    print(f"[Label] 웹 선택 '{chosen}' 무효 → 'all' 사용")
            else:
                print("[Label] 웹에서 전체 선택 → 'all'")
        except Exception as e:
            print(f"[Label] 웹 선택 실패 → 콘솔 대체: {e}")

    # 2) 고정 라벨
    if selected_color_label == "all" and (SELECTED_COLOR is not None):
        sc = SELECTED_COLOR.strip().lower()
        if sc in ALL_COLORS:
            print(f"[Label] 고정 라벨 사용: {sc}")
            selected_color_label = sc
        else:
            print(f"[Label] SELECTED_COLOR='{SELECTED_COLOR}' 무효 → 콘솔에서 입력")

    # 3) 콘솔 입력(여전히 라벨만)
    if selected_color_label == "all" and (args.no_web or not _USE_WEB):
        selected_color_label = ask_color_and_map_to_class(ALL_COLORS)

    # CSV 파일명(색상별 분리; 감지는 전체)
    csv_label = _sanitize_label(selected_color_label) if selected_color_label else "all"
    CSV_GRIPS_PATH = f"{args.csv_prefix}_{csv_label}.csv"
    print(f"[Info] 그립 CSV 파일: {CSV_GRIPS_PATH}  (감지는 항상 전체 클래스)")

    # === D455(컬러만 사용) & YOLO ===
    # 회전은 어댑터 내부에서 처리 → 이후 프레임은 항상 같은 좌표계로 옴
    cap = RealSenseColorDepth(color=(1280,720,30), depth=(848,480,30),
                              align_to_color=True, rotate90=args.rotate90)
    model = YOLO(str(MODEL_PATH))

    # ====== 초기 N프레임: YOLO seg & merge (원본 좌표계 유지) ======
    print(f"[Init] First 10 frames: YOLO seg & merge (detect=ALL, no-resize, no extra-rotate) ...")
    holds = initial_5frames_all_classes(
        cap, model, rotate_code=None,   # 어댑터가 이미 회전 적용함
        n_frames=10,
        mask_thresh=THRESH_MASK,
        merge_dist_px=18
    )
    if not holds:
        print("[Warn] 홀드가 검출되지 않았습니다."); return

    # ==== 선택: 단일 화면에서 클릭으로 골라서 후보 추출 ====
    sel_indices = interactive_select_live_left_only(cap, holds, window=WINDOW_NAME)
    if not sel_indices:
        print("[Select] 선택 없음"); return

    selected_holds = [holds[i] for i in sel_indices]

    # ==== 행(y)→열(x) 정렬로 hold_index 재부여 ====
    holds = assign_indices(selected_holds, row_tol=ROW_TOL_Y)

    # ==== 상태 ====
    pose = PoseTracker(min_detection_confidence=0.5, model_complexity=1)
    grip_records = []          # [(part, hold_id)]
    logged = set()             # {(part, hold_id)}  중복 방지
    streak = {}                # {(part, hold_id): frame_count}

    # (선택된 holds에 대해) 홀드 마스크 1회 생성 예약
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    t_prev = time.time()
    masks_built = False

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[Warn] 프레임 캡쳐 실패"); break

            vis = frame.copy()
            H_cur, W_cur = vis.shape[:2]

            # 선택된 홀드 마스크 1회 생성
            if not masks_built:
                for h in holds:
                    m = np.zeros((H_cur, W_cur), np.uint8)
                    cv2.drawContours(m, [h["contour"]], -1, 255, -1)
                    h["mask"] = m
                masks_built = True

            # === 포즈 ===
            coords = pose.process(frame)
            draw_pose_points(vis, coords, offset_x=0)

            # === 발 마스크(좌/우) 시각화 ===
            foot_masks = None
            if SHOW_FEET and coords:
                foot_masks = build_foot_masks_sided(coords, (H_cur, W_cur))
                overlay = vis.copy()
                for m in foot_masks.values():
                    if np.count_nonzero(m) > 0:
                        overlay[m > 0] = FOOT_COLOR
                cv2.addWeighted(overlay, FOOT_ALPHA, vis, 1.0 - FOOT_ALPHA, 0, vis)
                # 외곽선
                for m in foot_masks.values():
                    if np.count_nonzero(m) > 0:
                        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if cnts:
                            cv2.drawContours(vis, cnts, -1, FOOT_COLOR, FOOT_OUTLINE, cv2.LINE_AA)

            # === 그룹 단위(정확히: 대표 파트명) 터치 판단 ===
            live_filled_ids = set()
            touched_now = set()  # {('left_index', hid), ('right_index', hid), ('left_heel', hid), ('right_heel', hid)}

            if coords:
                left_hand_parts  = {"left_wrist","left_index","left_thumb","left_pinky"}
                right_hand_parts = {"right_wrist","right_index","right_thumb","right_pinky"}
                idx_map = {h["hold_index"]: h for h in holds}

                for hid, hold in idx_map.items():
                    # --- 손(좌): 네 파트 중 하나라도 in-polygon → 기록 키는 'left_index'
                    for pname in [p for p in left_hand_parts if p in coords]:
                        px, py = coords[pname]
                        if cv2.pointPolygonTest(hold["contour"], (px, py), False) >= 0:
                            touched_now.add(("left_index", hid))
                            live_filled_ids.add(hid)
                            break

                    # --- 손(우): 기록 키는 'right_index'
                    for pname in [p for p in right_hand_parts if p in coords]:
                        px, py = coords[pname]
                        if cv2.pointPolygonTest(hold["contour"], (px, py), False) >= 0:
                            touched_now.add(("right_index", hid))
                            live_filled_ids.add(hid)
                            break

                    # --- 발(좌/우): 마스크 겹침 → 기록 키는 'left_heel' / 'right_heel'
                    if foot_masks is not None and "mask" in hold:
                        hm = hold["mask"]
                        if np.count_nonzero(cv2.bitwise_and(foot_masks["left_foot"], hm)) > 0:
                            touched_now.add(("left_heel", hid))
                            live_filled_ids.add(hid)
                        if np.count_nonzero(cv2.bitwise_and(foot_masks["right_foot"], hm)) > 0:
                            touched_now.add(("right_heel", hid))
                            live_filled_ids.add(hid)

                # === 프레임 누적 → 로깅(중복 금지) ===
                for key in touched_now:
                    streak[key] = streak.get(key, 0) + 1
                    if streak[key] >= TOUCH_THRESHOLD and key not in logged:
                        part, hid = key     # part 는 mediapipe 이름(예: 'left_index', 'right_heel')
                        grip_records.append((part, hid))
                        logged.add(key)

                # 이번 프레임에서 터치 안 된 키는 0으로 리셋
                for key in list(streak.keys()):
                    if key not in touched_now:
                        streak[key] = 0
            else:
                # 포즈 실종 시 카운트 리셋 (로그는 유지)
                for key in list(streak.keys()):
                    streak[key] = 0

            # === 드로잉(컨투어/라벨/채움) ===
            for h in holds:
                cls_color = COLOR_MAP.get(h.get("class_name",""), (255,255,255))
                cv2.drawContours(vis, [h["contour"]], -1, cls_color, 2)
                cx, cy = h["center"]
                if h["hold_index"] in live_filled_ids:
                    cv2.drawContours(vis, [h["contour"]], -1, cls_color, thickness=cv2.FILLED)
                tag = f"ID:{h['hold_index']}"
                cv2.circle(vis, (cx, cy), 4, (255,255,255), -1)
                cv2.putText(vis, tag, (cx - 10, cy + 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
                cv2.putText(vis, tag, (cx - 10, cy + 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

            # FPS
            t_now = time.time()
            fps = 1.0 / max(t_now - (t_prev), 1e-6); t_prev = t_now
            cv2.putText(vis, f"FPS: {fps:.1f}",
                        (10, H_cur-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)
            cv2.putText(vis, f"FPS: {fps:.1f}",
                        (10, H_cur-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1, cv2.LINE_AA)

            cv2.imshow(WINDOW_NAME, vis)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break

    finally:
        try: cap.release()
        except: pass
        cv2.destroyAllWindows()
        try: pose.close()
        except: pass

    # === CSV 저장 (part, hold_id) ===
    with open(CSV_GRIPS_PATH, "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["part", "hold_id"])               # ← 헤더는 러너와 동일
        writer.writerows(grip_records)                      # 예: ('left_index', 7), ('right_heel', 11) ...

    print(f"[Info] 그립 CSV 저장 완료: {CSV_GRIPS_PATH} (총 {len(grip_records)}개)")

if __name__ == "__main__":
    main()
