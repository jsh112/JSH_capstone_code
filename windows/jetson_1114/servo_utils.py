import numpy as np

# === 서보 기준(중립 90/90) & 부호/스케일 ===
BASE_YAW_DEG   = 90.0   # 서보 중립
BASE_PITCH_DEG = 90.0   # 서보 중립
YAW_SIGN       = -1.0   # 반대로 가면 -1.0
PITCH_SIGN     = +1.0   # 반대로 가면 -1.0

def send_servo_angles(ctl, yaw_cmd, pitch_cmd):
    try:
        print(f"[Servo] send: yaw={yaw_cmd:.2f}°, pitch={pitch_cmd:.2f}°")
        ctl.set_angles(pitch_cmd, yaw_cmd)  # (pitch, yaw) 순서
    except Exception as e:
        print(f"[Servo ERROR] {e}")

def yaw_pitch_from_X(X, O, y_up_is_negative=True):
    v = X - O
    vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
    yaw   = np.degrees(np.arctan2(vx, vz))
    pitch = np.degrees(np.arctan2((-vy if y_up_is_negative else vy), np.hypot(vx, vz)))
    return yaw, pitch