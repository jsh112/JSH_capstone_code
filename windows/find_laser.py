import cv2, numpy as np, platform

# ========= 설정 =========
CAM0, CAM1 = 1, 2          # 카메라 인덱스
W, H = 1280, 720        # 해상도
USE_ECC_ALIGN = True       # 전/후 미세 흔들림(translation) 정합
BLUR = 0                   # 1픽셀 유지면 0, 약간 번지면 3
BORDER_IGNORE = 2          # 프레임 테두리 n픽셀 무시
REQUIRE_POSITIVE_DIFF = False  # 최대값이 0이어도 한 점 찍을지(False=찍음)

# 히트맵/표시
ALPHA = 0.6                        # 오버레이 강도
USE_TURBO = True                   # 팔레트: TURBO(권장) / JET
POINT_RADIUS = 10                  # 표시 점 크기
POINT_FILL_COLOR = (0, 0, 255)     # 빨강(가시성↑)
POINT_EDGE_COLOR = (255, 255, 255) # 흰 테두리
POINT_EDGE_THICK = 2

ROTATE_MAP = {
    1: cv2.ROTATE_90_COUNTERCLOCKWISE,  # LEFT
    2: cv2.ROTATE_90_CLOCKWISE,         # RIGHT
}
# =======================

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

def open_cam(idx):
    be = 0
    if platform.system() == "Windows": be = cv2.CAP_DSHOW
    elif platform.system() == "Linux": be = cv2.CAP_V4L2
    cap = cv2.VideoCapture(idx, be)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap

def to_gray_norm(bgr):
    """BGR→gray(float32) 후 z-score 정규화로 조도 변화 영향 축소."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    m, s = cv2.meanStdDev(g); s = max(float(s[0][0]), 1e-6)
    return (g - float(m[0][0])) / s

def align_translation(src, dst):
    """dst 기준으로 src를 translation 정합(ECC). 입력은 float32 gray."""
    try:
        # ECC 수렴 안정화(0..1 정규화)
        src_n = cv2.normalize(src, None, 0.0, 1.0, cv2.NORM_MINMAX)
        dst_n = cv2.normalize(dst, None, 0.0, 1.0, cv2.NORM_MINMAX)
        warp = np.eye(2, 3, dtype=np.float32)
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4)
        cv2.findTransformECC(dst_n, src_n, warp, cv2.MOTION_TRANSLATION, crit, None, 5)
        return cv2.warpAffine(src, warp, (dst.shape[1], dst.shape[0]),
                              flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_REPLICATE)
    except Exception:
        return src

def diff_maps(before_bgr, after_bgr):
    """
    반환:
      d8 : 0..255 uint8 절대차(히트맵용)
      d  : float32 절대차(최대점 탐색용)
    """
    gb = to_gray_norm(before_bgr)
    ga = to_gray_norm(after_bgr)
    if USE_ECC_ALIGN:
        ga = align_translation(ga, gb)
    d = np.abs(ga - gb).astype(np.float32)
    if BLUR >= 3 and BLUR % 2 == 1:
        d = cv2.GaussianBlur(d, (BLUR, BLUR), 0)
    d8 = cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return d8, d

def max_change_pixel(d_float, border_ignore=0, require_positive=False):
    """모든 픽셀 중 전역 최대 한 점 좌표 리턴."""
    if d_float is None or d_float.size == 0:
        return None
    roi = d_float.copy()
    bi = max(0, int(border_ignore))
    if bi > 0:
        roi[:bi, :] = 0; roi[-bi:, :] = 0; roi[:, :bi] = 0; roi[:, -bi:] = 0
    _, maxV, _, maxLoc = cv2.minMaxLoc(roi)
    if require_positive and maxV <= 0:
        return None
    return maxLoc  # (x, y)

def make_heatmap(d8):
    """순수 히트맵 BGR 이미지 반환."""
    if USE_TURBO and hasattr(cv2, "COLORMAP_TURBO"):
        return cv2.applyColorMap(d8, cv2.COLORMAP_TURBO)
    return cv2.applyColorMap(d8, cv2.COLORMAP_JET)

def overlay_heat(after_bgr, d8):
    """after 위에 히트맵 반투명 오버레이."""
    heat = make_heatmap(d8)
    return cv2.addWeighted(after_bgr, 1.0, heat, ALPHA, 0), heat

def draw_point(frame, pt):
    if pt is None: return frame
    cv2.circle(frame, pt, POINT_RADIUS, POINT_FILL_COLOR, -1, cv2.LINE_AA)
    cv2.circle(frame, pt, POINT_RADIUS + 2, POINT_EDGE_COLOR, POINT_EDGE_THICK, cv2.LINE_AA)
    return frame

def capture_once_and_return(
    port="COM15", baud=115200,
    wait_s=2.0, settle_n=8, show_preview=True,
    center_pitch=90.0, center_yaw=90.0, servo_settle_s=0.5,
    frame_size=None,          # ← (W, H) 튜플; 지정 시 해당 해상도로 캡쳐
):
    import time
    from servo_control import DualServoController

    # --- 로컬 카메라 오픈(전역 W/H에 의존하지 않음) ---
    def open_cam_with_size(idx, size=None):
        be = 0
        import platform
        if platform.system() == "Windows": be = cv2.CAP_DSHOW
        elif platform.system() == "Linux": be = cv2.CAP_V4L2
        cap = cv2.VideoCapture(idx, be)
        if size:
            W, H = size
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  int(W))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(H))
        cap.set(cv2.CAP_PROP_FPS, 30)
        return cap

    ROI_HALF = 30

    # --- 카메라 오픈 & 미리보기 창 ---
    cap0 = open_cam_with_size(CAM0, frame_size)
    cap1 = open_cam_with_size(CAM1, frame_size)
    if show_preview:
        cv2.namedWindow("cam0_preview", cv2.WINDOW_NORMAL)
        cv2.namedWindow("cam1_preview", cv2.WINDOW_NORMAL)
        cv2.namedWindow("diff0", cv2.WINDOW_NORMAL)
        cv2.namedWindow("diff1", cv2.WINDOW_NORMAL)

    # --- 보조: 회전 포함 프레임 획득 ---
    def _grab_rotated():
        r0, f0 = cap0.read()
        r1, f1 = cap1.read()
        if not (r0 and r1):
            raise RuntimeError("[find_laser] 카메라 프레임 획득 실패")
        f0 = rotate_image(f0, ROTATE_MAP.get(CAM0))
        f1 = rotate_image(f1, ROTATE_MAP.get(CAM1))
        return f0, f1

    # --- 보조: N프레임 워밍업 후 최종 한 장 반환 ---
    def _settle_and_grab(n=8):
        f0 = f1 = None
        for _ in range(max(1, int(n))):
            f0, f1 = _grab_rotated()
        return f0, f1

    # --- 서보 컨트롤러 ---
    ctl = DualServoController(port, baud)
    try:
        # 0) 중립 각 → 안정화
        try:
            ctl.set_angles(center_pitch, center_yaw)  # (pitch, yaw)
        except Exception as e:
            print("[find_laser] center set_angles error:", e)
        time.sleep(max(0.0, float(servo_settle_s)))

        # 1) 미리보기에서 양쪽 ROI 중심 클릭(ENTER/SPACE 불필요)
        roi_centers = {"cam0": None, "cam1": None}

        if show_preview:
            def _on_mouse_cam0(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    roi_centers["cam0"] = (x, y)
            def _on_mouse_cam1(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    roi_centers["cam1"] = (x, y)

            cv2.setMouseCallback("cam0_preview", _on_mouse_cam0)
            cv2.setMouseCallback("cam1_preview", _on_mouse_cam1)
            print("[find_laser] cam0, cam1 화면을 각각 클릭해서 ROI를 지정하세요. (정사각형, 한 변 = roi_box)")

            while roi_centers["cam0"] is None or roi_centers["cam1"] is None:
                try:
                    prev0, prev1 = _grab_rotated()
                except Exception:
                    continue

                H0, W0 = prev0.shape[:2]
                H1, W1 = prev1.shape[:2]
                disp0, disp1 = prev0.copy(), prev1.copy()

                # cam0 ROI 박스
                mx0, my0 = roi_centers["cam0"] if roi_centers["cam0"] else (W0 // 2, H0 // 2)
                x10 = max(0, mx0 - ROI_HALF); y10 = max(0, my0 - ROI_HALF)
                x20 = min(W0 - 1, mx0 + ROI_HALF); y20 = min(H0 - 1, my0 + ROI_HALF)
                cv2.rectangle(disp0, (x10, y10), (x20, y20),
                              (0, 0, 0) if roi_centers["cam0"] else (0, 0, 255), 2, cv2.LINE_AA)
                msg0 = "cam0: 클릭하여 ROI 확정" if roi_centers["cam0"] is None else "cam0: 확정됨"
                cv2.putText(disp0, msg0, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2, cv2.LINE_AA)

                # cam1 ROI 박스
                mx1, my1 = roi_centers["cam1"] if roi_centers["cam1"] else (W1 // 2, H1 // 2)
                x11 = max(0, mx1 - ROI_HALF); y11 = max(0, my1 - ROI_HALF)
                x21 = min(W1 - 1, mx1 + ROI_HALF); y21 = min(H1 - 1, my1 + ROI_HALF)
                cv2.rectangle(disp1, (x11, y11), (x21, y21),
                              (0, 0, 0) if roi_centers["cam1"] else (0, 0, 255), 2, cv2.LINE_AA)
                msg1 = "cam1: 클릭하여 ROI 확정" if roi_centers["cam1"] is None else "cam1: 확정됨"
                cv2.putText(disp1, msg1, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2, cv2.LINE_AA)

                cv2.imshow("cam0_preview", disp0)
                cv2.imshow("cam1_preview", disp1)
                cv2.waitKey(10)  # 이벤트 펌프

            # 더 이상 클릭 안 받음
            cv2.setMouseCallback("cam0_preview", lambda *a: None)
            cv2.setMouseCallback("cam1_preview", lambda *a: None)

        # 2) 프레임 크기 확정(회전 후)
        ref0, ref1 = _grab_rotated()
        H0, W0 = ref0.shape[:2]
        H1, W1 = ref1.shape[:2]

        mx0, my0 = roi_centers["cam0"] if roi_centers["cam0"] else (W0 // 2, H0 // 2)
        x10 = max(0, mx0 - ROI_HALF); y10 = max(0, my0 - ROI_HALF)
        x20 = min(W0 - 1, mx0 + ROI_HALF); y20 = min(H0 - 1, my0 + ROI_HALF)

        mx1, my1 = roi_centers["cam1"] if roi_centers["cam1"] else (W1 // 2, H1 // 2)
        x11 = max(0, mx1 - ROI_HALF); y11 = max(0, my1 - ROI_HALF)
        x21 = min(W1 - 1, mx1 + ROI_HALF); y21 = min(H1 - 1, my1 + ROI_HALF)

        # 3) ON → OFF 캡처(해당 ROI만 차분 사용)
        try: ctl.laser_on()
        except Exception as e: print("[find_laser] laser_on error:", e)
        time.sleep(max(0.0, float(wait_s)))
        ON0, ON1 = _settle_and_grab(settle_n)

        try: ctl.laser_off()
        except Exception as e: print("[find_laser] laser_off error:", e)
        time.sleep(max(0.0, float(wait_s)))
        OFF0, OFF1 = _settle_and_grab(settle_n)

        # 절대차(전체 계산 후 ROI에서 최대점)
        _, d0_full = diff_maps(OFF0, ON0)
        _, d1_full = diff_maps(OFF1, ON1)

        pt0 = None
        roi0 = d0_full[y10:y20, x10:x20]
        if roi0.size > 0:
            loc0 = max_change_pixel(roi0, border_ignore=0, require_positive=False)
            if loc0 is not None:
                pt0 = (x10 + loc0[0], y10 + loc0[1])

        pt1 = None
        roi1 = d1_full[y11:y21, x11:x21]
        if roi1.size > 0:
            loc1 = max_change_pixel(roi1, border_ignore=0, require_positive=False)
            if loc1 is not None:
                pt1 = (x11 + loc1[0], y11 + loc1[1])

        # 시각화
        if show_preview:
            d8_0 = cv2.normalize(d0_full, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            d8_1 = cv2.normalize(d1_full, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            over0, heat0 = overlay_heat(ON0.copy(), d8_0)
            over1, heat1 = overlay_heat(ON1.copy(), d8_1)

            cv2.rectangle(over0, (x10, y10), (x20, y20), (0, 0, 0), 2, cv2.LINE_AA)
            cv2.rectangle(over1, (x11, y11), (x21, y21), (0, 0, 0), 2, cv2.LINE_AA)

            over0 = draw_point(over0, pt0)
            over1 = draw_point(over1, pt1)
            if pt0 is not None:
                cv2.drawMarker(heat0, pt0, (0,255,0), cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)
            if pt1 is not None:
                cv2.drawMarker(heat1, pt1, (0,255,0), cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)

            cv2.imshow("cam0_preview", over0)
            cv2.imshow("cam1_preview", over1)
            cv2.imshow("diff0", heat0)
            cv2.imshow("diff1", heat1)
            cv2.waitKey(1)

    finally:
        try: cap0.release(); cap1.release()
        except: pass
        try: cv2.destroyAllWindows()
        except: pass
        try: ctl.close()
        except: pass

    return {
        "image_size": (W0, H0),  # cam0 기준(회전 후) 실제 크기 반환
        "cam0": (int(pt0[0]), int(pt0[1])) if pt0 is not None else None,
        "cam1": (int(pt1[0]), int(pt1[1])) if pt1 is not None else None,
    }

def main():
    cap0, cap1 = open_cam(CAM0), open_cam(CAM1)
    cv2.namedWindow("cam0",  cv2.WINDOW_NORMAL)
    cv2.namedWindow("cam1",  cv2.WINDOW_NORMAL)
    cv2.namedWindow("diff0", cv2.WINDOW_NORMAL)  # 순수 히트맵
    cv2.namedWindow("diff1", cv2.WINDOW_NORMAL)

    before0 = before1 = None
    print("[Space] 전→후(히트맵+최대 한 점) | [x] 초기화 | [q] 종료 — 저장 없음")

    while True:
        r0, f0 = cap0.read()
        r1, f1 = cap1.read()
        if not (r0 and r1): break
        
        f0 = rotate_image(f0, ROTATE_MAP.get(CAM0))
        f1 = rotate_image(f1, ROTATE_MAP.get(CAM1))

        # 상태 표시
        v0, v1 = f0.copy(), f1.copy()
        status = "READY: Space=BEFORE 촬영" if before0 is None else "BEFORE OK: Space=AFTER 촬영→표시  /  x=초기화"
        for img in (v0, v1):
            cv2.putText(img, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("cam0", v0); cv2.imshow("cam1", v1)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        elif k == ord('x'):
            before0 = before1 = None
            print("[초기화]")
        elif k == 32:  # Space
            if before0 is None:
                before0, before1 = f0.copy(), f1.copy()
                print("[BEFORE] 촬영")
            else:
                after0, after1 = f0.copy(), f1.copy()
                print("[AFTER]  촬영 → 히트맵+최대 변화 한 점 표시")

                # 1) 절대차 맵들
                d8_0, d0 = diff_maps(before0, after0)
                d8_1, d1 = diff_maps(before1, after1)

                # 2) 모든 픽셀 중 전역 최대 한 점
                pt0 = max_change_pixel(d0, BORDER_IGNORE, REQUIRE_POSITIVE_DIFF)
                pt1 = max_change_pixel(d1, BORDER_IGNORE, REQUIRE_POSITIVE_DIFF)

                # 3) after 위 히트맵 오버레이 + 한 점 표시
                over0, heat0 = overlay_heat(after0, d8_0)
                over1, heat1 = overlay_heat(after1, d8_1)
                over0 = draw_point(over0, pt0)
                over1 = draw_point(over1, pt1)

                # 4) 순수 히트맵에도 동일 지점 마커(가독성↑)
                if pt0 is not None:
                    cv2.drawMarker(heat0, pt0, (0,255,0), cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)
                if pt1 is not None:
                    cv2.drawMarker(heat1, pt1, (0,255,0), cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)

                # 5) 표출
                cv2.imshow("cam0", over0)
                cv2.imshow("cam1", over1)
                cv2.imshow("diff0", heat0)
                cv2.imshow("diff1", heat1)

    cap0.release(); cap1.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
