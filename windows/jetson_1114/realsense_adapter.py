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

        # 1차 시도: 사용자가 요청한 사양
        _enable(cfg, rs.stream.color, color, rs.format.bgr8)
        _enable(cfg, rs.stream.depth, depth, rs.format.z16)

        try:
            self.profile = self.pipe.start(cfg)

        except RuntimeError as e:
            # ---- 진단 로그: USB 타입, 사용 가능 프로필 출력 ----
            try:
                ctx = rs.context()
                dev = ctx.query_devices()[0]
                usb = dev.get_info(rs.camera_info.usb_type_descriptor)
                print(f"[RealSense] start() failed: {e}")
                print(f"[RealSense] USB link: {usb} (USB 3.x가 아니면 대역폭 부족 가능성↑)")
                for s in dev.query_sensors():
                    try:
                        name = s.get_info(rs.camera_info.name)
                    except Exception:
                        name = "sensor"
                    print(f"  - Sensor: {name}")
                    for p in s.get_stream_profiles():
                        vp = p.as_video_stream_profile()
                        fmt = vp.format()
                        print(f"    {vp.stream_type()} {vp.width()}x{vp.height()}@{vp.fps()} {fmt}")
            except Exception:
                pass

            # ---- 폴백: 가장 안전한 조합으로 재시도 ----
            cfg.disable_all_streams()
            cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            print("[RealSense] Fallback to color/depth 640x480@30 ...")
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
        if not hasattr(self, "_last_depth") or self._last_depth is None:
            return None
        d = np.asanyarray(self._last_depth.get_data()).astype(np.float32)
        scale = self.profile.get_device().first_depth_sensor().get_depth_scale()
        return d * float(scale)  # (H,W) 그대로 반환

    def deproject(self, x, y, z_m):
        intr = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        X = rs.rs2_deproject_pixel_to_point(intr, [float(x), float(y)], float(z_m))
        return np.array(X, dtype=np.float32)

    def release(self):
        try:
            self.pipe.stop()
        except Exception:
            pass
