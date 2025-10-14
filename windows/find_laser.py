import cv2, numpy as np, platform

# ========= 설정 =========
CAM0, CAM1 = 2, 3          # 카메라 인덱스
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
    2: cv2.ROTATE_90_COUNTERCLOCKWISE,  # LEFT
    3: cv2.ROTATE_90_CLOCKWISE,         # RIGHT
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

# def capture_once_and_return(port="COM15", baud=115200,
#                             wait_s=2.0, settle_n=8, show_preview=True,
#                             center_pitch=90.0, center_yaw=90.0, servo_settle_s=0.5):
#     from servo_control import DualServoController
#     import time, cv2
#
#     def _settle_and_grab(cap0, cap1, n=8):
#         f0 = f1 = None
#         for _ in range(max(1, int(n))):
#             r0, f0 = cap0.read()
#             r1, f1 = cap1.read()
#             if not (r0 and r1):
#                 raise RuntimeError("[find_laser] 카메라 프레임 획득 실패")
#         return f0, f1
#
#     cap0, cap1 = open_cam(CAM0), open_cam(CAM1)
#     if show_preview:
#         cv2.namedWindow("cam0_preview", cv2.WINDOW_NORMAL)
#         cv2.namedWindow("cam1_preview", cv2.WINDOW_NORMAL)
#
#     ctl = DualServoController(port, baud)
#     try:
#         # 0) 서보 중립(90/90) 세팅 → 안정화 대기
#         try:
#             # set_angles(pitch, yaw) 순서 주의
#             ctl.set_angles(center_pitch, center_yaw)
#         except Exception as e:
#             print("[find_laser] center set_angles error:", e)
#         time.sleep(max(0.0, float(servo_settle_s)))
#
#         # 1) 레이저 OFF → 대기 → BEFORE
#         ctl.laser_off()
#         time.sleep(max(0.0, float(wait_s)))
#         before0, before1 = _settle_and_grab(cap0, cap1, settle_n)
#         before0 = rotate_image(before0, ROTATE_MAP.get(CAM0))
#         before1 = rotate_image(before1, ROTATE_MAP.get(CAM1))
#         if show_preview:
#             cv2.imshow("cam0_preview", before0)
#             cv2.imshow("cam1_preview", before1)
#             cv2.waitKey(1)
#
#         # 2) 레이저 ON → 대기 → AFTER
#         ctl.laser_on()
#         time.sleep(max(0.0, float(wait_s)))
#         after0, after1 = _settle_and_grab(cap0, cap1, settle_n)
#         after0 = rotate_image(after0, ROTATE_MAP.get(CAM0))
#         after1 = rotate_image(after1, ROTATE_MAP.get(CAM1))
#         if show_preview:
#             cv2.imshow("cam0_preview", after0)
#             cv2.imshow("cam1_preview", after1)
#             cv2.waitKey(1)
#
#     finally:
#         try: ctl.close()
#         except: pass
#
#     # 3) 절대차 → 최대 변화점
#     _, d0 = diff_maps(before0, after0)
#     _, d1 = diff_maps(before1, after1)
#     pt0 = max_change_pixel(d0, BORDER_IGNORE, REQUIRE_POSITIVE_DIFF)
#     pt1 = max_change_pixel(d1, BORDER_IGNORE, REQUIRE_POSITIVE_DIFF)
#
#     if show_preview:
#         d8_0 = cv2.normalize(d0, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
#         d8_1 = cv2.normalize(d1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
#         over0, heat0 = overlay_heat(after0, d8_0)
#         over1, heat1 = overlay_heat(after1, d8_1)
#         over0 = draw_point(over0, pt0); over1 = draw_point(over1, pt1)
#         if pt0 is not None: cv2.drawMarker(heat0, pt0, (0,255,0), cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)
#         if pt1 is not None: cv2.drawMarker(heat1, pt1, (0,255,0), cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)
#         cv2.imshow("cam0_preview", over0); cv2.imshow("cam1_preview", over1)
#         cv2.imshow("diff0", heat0);        cv2.imshow("diff1", heat1)
#         cv2.waitKey(1)
#
#     try:
#         cap0.release(); cap1.release(); cv2.destroyAllWindows()
#     except: pass
#     H0, W0 = after0.shape[:2]
#     H1, W1 = after1.shape[:2]
#     return {
#         "image_size": (W0, H0),  # cam0 회전 후 크기 (W,H)
#         "cam0": (int(pt0[0]), int(pt0[1])) if pt0 is not None else None,
#         "cam1": (int(pt1[0]), int(pt1[1])) if pt1 is not None else None,
#     }
def capture_once_and_return(port="COM15", baud=115200,
                            center_pitch=90.0, center_yaw=90.0, servo_settle_s=0.5):
    from servo_control import DualServoController
    import time, cv2

    pt0 = None
    pt1 = None
    f0r, fr1 = None, None

    cap0, cap1= open_cam(CAM0), open_cam(CAM1)
    # ====== 해상도 테스트 ==========================
    # ✅ 실제 적용된 해상도 확인
    real_w0 = cap0.get(cv2.CAP_PROP_FRAME_WIDTH)
    real_h0 = cap0.get(cv2.CAP_PROP_FRAME_HEIGHT)
    real_w1 = cap1.get(cv2.CAP_PROP_FRAME_WIDTH)
    real_h1 = cap1.get(cv2.CAP_PROP_FRAME_HEIGHT)

    print(f"[DEBUG] cam0 actual size: {int(real_w0)}x{int(real_h0)}")
    print(f"[DEBUG] cam1 actual size: {int(real_w1)}x{int(real_h1)}")

    # ✅ FOURCC 확인
    print(f"[DEBUG] cam0 FOURCC:", cap0.get(cv2.CAP_PROP_FOURCC))
    print(f"[DEBUG] cam1 FOURCC:", cap1.get(cv2.CAP_PROP_FOURCC))
    # ====== 해상도 테스트 ==========================

    print(f"[DEBUG] 시리얼 연결 시도: {port}, {baud}bps")
    ctl = DualServoController(port, baud)
    print("[DEBUG] 서보 컨트롤러 초기화 완료")

    print(f"[DEBUG] 서보 각도 초기화 중: pitch={center_pitch}, yaw={center_yaw}")
    ctl.set_angles(center_pitch, center_yaw)
    time.sleep(servo_settle_s) # 0.5초
    print("[DEBUG] 서보 초기 위치 설정 완료")

    print("[DEBUG] 레이저 ON 명령 전송 중...")
    ctl.laser_on()
    time.sleep(0.2)
    print("🔴 레이저 켜짐 — 클릭해서 레이저 점을 선택하세요")

    cv2.namedWindow("cam0_preview", cv2.WINDOW_NORMAL)
    cv2.namedWindow("cam1_preview", cv2.WINDOW_NORMAL)

    def callback_left(event, x, y, flags, param):
        nonlocal pt0
        if event == cv2.EVENT_LBUTTONDOWN:
            pt0 = (x, y)
            print(f"🟢 Left cam0: {pt0}")

    # 오른쪽 카메라 클릭 → cam1 좌표
    def callback_right(event, x, y, flags, param):
        nonlocal pt1
        if event == cv2.EVENT_LBUTTONDOWN:
            pt1 = (x, y)
            print(f"🔵 Right cam1: {pt1}")

    cv2.setMouseCallback("cam0_preview", callback_left)
    cv2.setMouseCallback("cam1_preview", callback_right)

    for _ in range(10):
        cap0.read();cap1.read()
    time.sleep(0.5)

 # --- 루프: 양쪽 영상 프리뷰 ---
    while True:
        r0, f0 = cap0.read()
        r1, f1 = cap1.read()
        if not (r0 and r1):
            print("⚠️ 카메라 프레임 읽기 실패")
            time.sleep(0.2)
            continue

        f0r = rotate_image(f0, ROTATE_MAP.get(CAM0))
        f1r = rotate_image(f1, ROTATE_MAP.get(CAM1))

        # 미리보기 표시
        cv2.imshow("cam0_preview", f0r)
        cv2.imshow("cam1_preview", f1r)

        k = cv2.waitKey(1) & 0xFF

        # Enter로 확정 (양쪽 클릭 완료)
        if k == 13 and pt0 is not None:
            print(f"✅ 레이저 확정: L={pt0}, R={pt1}")
            break

        # ESC로 초기화
        elif k == 27:
            pt0,pt1 = None, None
            print("🔁 좌표 초기화")
    print(f'============Ready to Laser off =============')
    ctl.laser_off()
    print(f'============ Laser off =============')
    cap0.release()
    cap1.release()
    cv2.destroyAllWindows()

    # --- 안전한 크기 추출 ---
    H0, W0 = f0r.shape[:2]

    return {
        "image_size": (W0, H0),  # cam0 회전 후 크기 (W,H)
        "cam0": (int(pt0[0]), int(pt0[1])),
        "cam1": (int(pt1[0]), int(pt1[1]))
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
