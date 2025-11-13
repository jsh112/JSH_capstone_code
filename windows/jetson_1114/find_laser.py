import cv2, numpy as np, time
from servo_control import DualServoController
from realsense_adapter import RealSenseColorDepth

# ========= 시각화 설정 =========
ALPHA = 0.6
USE_TURBO = True
POINT_RADIUS = 10
POINT_FILL_COLOR = (0, 0, 255)
POINT_EDGE_COLOR = (255, 255, 255)
POINT_EDGE_THICK = 2
# ==============================

def to_gray_norm(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    m, s = cv2.meanStdDev(g)
    s = max(float(s[0][0]), 1e-6)
    return (g - float(m[0][0])) / s

def align_translation(src, dst):
    try:
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
    gb = to_gray_norm(before_bgr)
    ga = to_gray_norm(after_bgr)
    ga = align_translation(ga, gb)
    d = np.abs(ga - gb).astype(np.float32)
    d8 = cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return d8, d

def max_change_pixel(d_float, border_ignore=0, require_positive=False):
    if d_float is None or d_float.size == 0:
        return None
    roi = d_float.copy()
    bi = max(0, int(border_ignore))
    if bi > 0:
        roi[:bi, :] = 0; roi[-bi:, :] = 0; roi[:, :bi] = 0; roi[:, -bi:] = 0
    _, maxV, _, maxLoc = cv2.minMaxLoc(roi)
    if require_positive and maxV <= 0:
        return None
    return maxLoc

def make_heatmap(d8):
    if USE_TURBO and hasattr(cv2, "COLORMAP_TURBO"):
        return cv2.applyColorMap(d8, cv2.COLORMAP_TURBO)
    return cv2.applyColorMap(d8, cv2.COLORMAP_JET)

def overlay_heat(after_bgr, d8):
    heat = make_heatmap(d8)
    return cv2.addWeighted(after_bgr, 1.0, heat, ALPHA, 0), heat

def draw_point(frame, pt):
    if pt is None: return frame
    cv2.circle(frame, pt, POINT_RADIUS, POINT_FILL_COLOR, -1, cv2.LINE_AA)
    cv2.circle(frame, pt, POINT_RADIUS + 2, POINT_EDGE_COLOR, POINT_EDGE_THICK, cv2.LINE_AA)
    return frame

# === D455 전용 단일창 레이저 캡처 ===
def capture_once_and_return(
    port="COM4", baud=115200,
    wait_s=2.0, settle_n=8, show_preview=True,
    center_pitch=90.0, center_yaw=90.0, servo_settle_s=0.5,
    frame_size=(1280, 720),
):
    cap = RealSenseColorDepth(
        color=(frame_size[0], frame_size[1], 30),
        depth=(848, 480, 30),
        align_to_color=True,
        rotate90=False,
    )
    ctl = DualServoController(port, baud)

    try:
        ctl.set_angles(center_pitch, center_yaw)
        time.sleep(max(0.0, float(servo_settle_s)))

        WINDOW = "laser_finder_d455"
        ROI_HALF = 20
        roi_center = None

        if show_preview:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

            def _on_mouse(event, x, y, flags, param):
                nonlocal roi_center
                if event == cv2.EVENT_LBUTTONDOWN:
                    roi_center = (int(x), int(y))

            cv2.setMouseCallback(WINDOW, _on_mouse)

            while roi_center is None:
                ok, color = cap.read()
                if not ok: continue
                disp = color.copy()
                h, w = disp.shape[:2]
                mx, my = w // 2, h // 2
                cv2.rectangle(disp, (mx-ROI_HALF, my-ROI_HALF),
                              (mx+ROI_HALF, my+ROI_HALF), (0,0,255), 2)
                cv2.putText(disp, "click to set ROI", (10, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
                cv2.imshow(WINDOW, disp)
                if (cv2.waitKey(10) & 0xFF) in (27, ord('q')):
                    roi_center = (mx, my)

        ok, ref_color = cap.read()
        if not ok:
            raise RuntimeError("[find_laser] D455 첫 프레임 실패")
        H, W = ref_color.shape[:2]
        if roi_center is None:
            roi_center = (W//2, H//2)

        mx, my = roi_center
        x1, y1 = max(0, mx-ROI_HALF), max(0, my-ROI_HALF)
        x2, y2 = min(W-1, mx+ROI_HALF), min(H-1, my+ROI_HALF)

        ctl.laser_on()
        time.sleep(wait_s)
        for _ in range(int(settle_n)):
            ok, on_color = cap.read()
        ctl.laser_off()
        time.sleep(wait_s)
        for _ in range(int(settle_n)):
            ok, off_color = cap.read()

        d8, d_full = diff_maps(off_color, on_color)
        roi = d_full[y1:y2, x1:x2]
        ptL = None
        if roi.size > 0:
            loc = max_change_pixel(roi)
            if loc is not None:
                ptL = (x1 + int(loc[0]), y1 + int(loc[1]))

        if show_preview:
            over, heat = overlay_heat(on_color.copy(), d8)
            if ptL is not None:
                draw_point(over, ptL)
            vis = cv2.vconcat([over, heat])
            cv2.imshow(WINDOW, vis)
            cv2.waitKey(1000)       # 1ms만 처리하고 바로 진행 (엔터 불필요)
            cv2.destroyWindow(WINDOW)  # 창 깔끔히 닫고 다음 단계로

        X_laser_mm = None
        if ptL is not None:
            depth_m = cap.get_depth_meters()
            x, y = ptL
            r = 3
            patch = depth_m[max(0,y-r):min(H-1,y+r), max(0,x-r):min(W-1,x+r)]
            vals = patch[(patch > 0) & np.isfinite(patch)]
            if vals.size > 0:
                Zm = float(np.median(vals))
                X_m = cap.deproject(x, y, Zm)
                X_laser_mm = X_m * 1000.0

    finally:
        cap.release()
        cv2.destroyAllWindows()
        ctl.close()

    return {
        "image_size": (W, H),
        "left_raw": (int(ptL[0]), int(ptL[1])) if ptL is not None else None,
        "X_laser_mm": X_laser_mm,
    }
