import pyrealsense2 as rs
import numpy as np

class RealSenseColorDepth:
    def __init__(self, color=(1280,720,30), depth=(848,480,30), align_to_color=True, rotate90=False):
        self.pipe = rs.pipeline()
        cfg = rs.config()

        color_w, color_h, color_fps = color
        depth_w, depth_h, depth_fps = depth

        # ✅ 올바른 순서로 스트림 활성화
        cfg.enable_stream(rs.stream.color, color_w, color_h, rs.format.bgr8, color_fps)
        cfg.enable_stream(rs.stream.depth, depth_w, depth_h, rs.format.z16, depth_fps)

        self.profile = self.pipe.start(cfg)
        self.align = rs.align(rs.stream.color) if align_to_color else None

        # 💡 Post-processing 필터 체인
        self.decimation = rs.decimation_filter()
        self.spatial = rs.spatial_filter()
        self.temporal = rs.temporal_filter()
        self.hole_filling = rs.hole_filling_filter()

        # ✅ depth scale 안전하게 얻기
        sensor = self.profile.get_device().first_depth_sensor()
        if sensor.supports(rs.option.depth_units):
            self.depth_scale = sensor.get_depth_scale()
        else:
            # fallback (대부분 0.001)
            self.depth_scale = 0.001

        self.rotate90 = rotate90
        self._last_depth = None

    def read(self):
        frames = self.pipe.wait_for_frames()
        if self.align:
            frames = self.align.process(frames)
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        # 필터 적용
        depth_frame = self.decimation.process(depth_frame)
        depth_frame = self.spatial.process(depth_frame)
        depth_frame = self.temporal.process(depth_frame)
        depth_frame = self.hole_filling.process(depth_frame)

        color = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data()) * self.depth_scale
        self._last_depth = depth

        # ✅ 변경: depth는 내부에 저장하고 리턴하지 않음
        return True, color

    def get_depth_meters(self):
        """최근 읽은 depth를 m 단위로 반환"""
        return self._last_depth

    def deproject(self, x, y, depth_m):
        """(x, y, depth) → 3D (m)"""
        intr = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        X = rs.rs2_deproject_pixel_to_point(intr, [x, y], depth_m)
        return np.array(X, dtype=np.float64)

    def release(self):
        self.pipe.stop()
