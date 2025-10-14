from A_A_Climbing import triangulate_xy_raw, yaw_pitch_from_X, compute_laser_origin_mid
import numpy as np, cv2

S = np.load("param/stereo_params_scaled_1012.npz", allow_pickle=True)
K1, D1 = S["K1"], S["D1"]
K2, D2 = S["K2"], S["D2"]
R, T   = S["R"], S["T"].reshape(3,1)

# 이미지 불러오기
P1 = K1 @ np.hstack([np.eye(3), np.zeros((3,1))])  # Left camera: [I | 0]
P2 = K2 @ np.hstack([R, T])                        # Right camera: [R | T]

X1 = triangulate_xy_raw(P1,P2,ptL1,ptR1, K1, D1, K2, D2)
X2 = triangulate_xy_raw(P1,P2,ptL2,ptR2, K1, D1, K2, D2)
distance = np.linalg.norm(X1 - X2)
print(f"distance = {distance}")
