# realsense_adapter.py (상단)
import pyrealsense2 as rs
import numpy as np
import cv2

class RealSenseColorDepth:
    def __init__(self, color=(1280,720,30), depth=(848,480,30),
                 align_to_color=True, rotate90=False):
        self.rotate90 = rotate90

        self.pipe = rs.pipeline()
        cfg = rs.config()

        # --- 견고한 enable_stream 래퍼 ---
        def _enable(cfg, stream, spec, default_fmt, stream_index=0):
            # 허용 입력:
            # (w, h, fps)
            # (w, h, fmt, fps)
            # (idx, w, h, fmt, fps)
            if not isinstance(spec, (tuple, list)):
                raise ValueError(f"Stream spec must be tuple, got {type(spec)}")
            if len(spec) == 3:
                w, h, fps = spec
                cfg.enable_stream(stream, int(w), int(h), default_fmt, int(fps))
            elif len(spec) == 4:
                w, h, fmt, fps = spec
                cfg.enable_stream(stream, int(w), int(h), fmt, int(fps))
            elif len(spec) == 5:
                idx, w, h, fmt, fps = spec
                cfg.enable_stream(stream, int(idx), int(w), int(h), fmt, int(fps))
            else:
                raise ValueError(f"Invalid stream spec length: {len(spec)}")

        # color/depth 스트림 켜기
        _enable(cfg, rs.stream.color, color, rs.format.bgr8)
        _enable(cfg, rs.stream.depth, depth, rs.format.z16)

        self.profile = self.pipe.start(cfg)

        # 정합(align)
        self.align = rs.align(rs.stream.color) if align_to_color else None

    def read(self):
        frames = self.pipe.wait_for_frames()
        if self.align is not None:
            frames = self.align.process(frames)

        c = frames.get_color_frame()
        d = frames.get_depth_frame()
        if not c or not d:
            return False, None

        img = np.asanyarray(c.get_data())
        if self.rotate90:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        self._last_depth = d
        return True, img

    def get_depth_meters(self):
        # depth frame → meters float32
        if not hasattr(self, "_last_depth") or self._last_depth is None:
            return np.zeros((720,1280), np.float32)
        d = np.asanyarray(self._last_depth.get_data()).astype(np.float32)
        scale = self.profile.get_device().first_depth_sensor().get_depth_scale()  # meters
        return d * float(scale)

    def deproject(self, x, y, z_m):
        # 픽셀/깊이 → 카메라 좌표계 (미터)
        intr = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        X = rs.rs2_deproject_pixel_to_point(intr, [float(x), float(y)], float(z_m))
        return np.array(X, dtype=np.float32)

    def release(self):
        try:
            self.pipe.stop()
        except Exception:
            pass
