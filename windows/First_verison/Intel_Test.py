# realsense_rgb_depth_preview.py
import pyrealsense2 as rs
import numpy as np
import cv2
import time

# ===== 설정 =====
COLOR_W, COLOR_H, COLOR_FPS = 1280, 720, 30
DEPTH_W, DEPTH_H, DEPTH_FPS = 848,  480, 30
ALIGN_TO_COLOR = True     # Depth→Color 정렬 사용
ROTATE_90 = False         # 카메라를 세로로 달았다면 True로 바꾸세요

def maybe_rotate(img):
    return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) if ROTATE_90 else img

pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, COLOR_W, COLOR_H, rs.format.bgr8, COLOR_FPS)
cfg.enable_stream(rs.stream.depth, DEPTH_W, DEPTH_H, rs.format.z16, DEPTH_FPS)
profile = pipe.start(cfg)

align = rs.align(rs.stream.color) if ALIGN_TO_COLOR else None
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()  # depth_raw * depth_scale = meters

cv2.namedWindow('color', cv2.WINDOW_NORMAL)
cv2.namedWindow('depth(aligned)', cv2.WINDOW_NORMAL)

# 클릭한 위치의 거리(m) 표시용
clicked = None
def on_mouse(event, x, y, flags, param):
    global clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked = (x, y)
cv2.setMouseCallback('color', on_mouse)

t0, n = time.time(), 0
try:
    while True:
        frames = pipe.wait_for_frames()
        if align:
            frames = align.process(frames)
        cf, df = frames.get_color_frame(), frames.get_depth_frame()
        if not cf or not df:
            continue

        color = np.asanyarray(cf.get_data())        # BGR
        depth = np.asanyarray(df.get_data())        # uint16 (mm 단위 아님!)
        # 보기 쉬운 컬러맵(가시화용) — 0~4m 기준 스케일
        depth_vis = cv2.convertScaleAbs(depth, alpha=255.0/4000.0)
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

        # 회전(세로 장착 시 color/depth 둘 다 동일하게)
        color = maybe_rotate(color)
        depth_vis = maybe_rotate(depth_vis)
        depth_for_measure = maybe_rotate(depth)  # 실제 거리 읽을 때 사용

        # 클릭한 위치의 실제 거리(m) 표시
        if clicked is not None:
            x, y = clicked
            if 0 <= y < depth_for_measure.shape[0] and 0 <= x < depth_for_measure.shape[1]:
                Zm = float(depth_for_measure[y, x]) * depth_scale
                if Zm > 0:
                    cv2.circle(color, (x, y), 5, (0, 0, 255), -1)
                    cv2.putText(color, f"{Zm:.2f} m", (x+8, y-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        cv2.imshow('color', color)
        cv2.imshow('depth(aligned)', depth_vis)

        n += 1
        if n % 60 == 0:
            fps = n / (time.time() - t0)
            print(f"~{fps:.1f} FPS")
            t0, n = time.time(), 0

        key = cv2.waitKey(1)
        if key in (27, ord('q')):  # ESC or q
            break
finally:
    pipe.stop()
    cv2.destroyAllWindows()
