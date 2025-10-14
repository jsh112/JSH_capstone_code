import cv2
NPZ_PATH       = r"./param/stereo_params_1024_576_.npz"
MODEL_PATH     = r"./param/best_6.pt"

CAM1_INDEX     = 2   # 왼쪽 카메라
CAM2_INDEX     = 3   # 오른쪽 카메라

SWAP_DISPLAY   = False   # 화면 표시 좌/우 스와프

WINDOW_NAME    = "Rectified L | R"
THRESH_MASK    = 0.7
ROW_TOL_Y      = 30

# 자동 진행(터치→다음 홀드) 관련
TOUCH_THRESHOLD = 10     # in-polygon 연속 프레임 임계(기본 10)
ADV_COOLDOWN    = 0.5    # 연속 넘김 방지 쿨다운(sec)

# 저장 옵션
SAVE_VIDEO     = False
OUT_FPS        = 30
OUT_PATH       = "stereo_overlay.mp4"
CSV_GRIPS_PATH = "route/grip_records.csv"

# 런타임 보정 오프셋(레이저 실측)
CAL_YAW_OFFSET   = 0.0
CAL_PITCH_OFFSET = 0.0

# ---- 레이저 원점(LEFT 기준) 오프셋 (cm) ----
LASER_OFFSET_CM_LEFT = 1.15
LASER_OFFSET_CM_UP   = 5.2
LASER_OFFSET_CM_FWD  = -0.6
Y_UP_IS_NEGATIVE     = True  # 위 방향이 -y인 좌표계면 True

# === 서보 기준(중립 90/90) & 부호/스케일 ===
BASE_YAW_DEG   = 90.0   # 서보 중립
BASE_PITCH_DEG = 90.0   # 서보 중립
YAW_SIGN       = -1.0   # 반대로 가면 -1.0
PITCH_SIGN     = +1.0   # 반대로 가면 -1.0
YAW_SCALE      = 1.0    # 필요시 감도 미세조정
PITCH_SCALE    = 1.0

# === 전역 기준(초기 레이저 기준 각) ===
YAW_LASER0 = None
PITCH_LASER0 = None

K_EXTRA_DEG = 3.0    # 최대 추가 피치 스케일(도)
H0_MM       = 1000.0 # 높이 정규화(1 m)
Z0_MM       = 4000.0 # 깊이 정규화(4 m에서 감쇠=1배)
BETA_Z      = 1.0    # 깊이 감쇠 기울기(0이면 감쇠 없음)
H_SOFT_MM  = 160.0   # 높이차가 이 정도 이하일 때 가산을 살짝 눌러줌
GAMMA_H    = 0.1     # 1.0~1.5 권장 (커지면 초반 더 세게 눌림)

# === 이미지 차분 파라미터 ===
DIFF_EVERY_N     = 10      # N프레임마다 차분(=부하 감소)
ADAPT_ALPHA      = 0.02   # 배경(베이스라인) 적응 속도(0=고정)
DILATE_KERNEL_SZ = 3      # 마스크 팽창 커널(노이즈↓/완충)

# === 강한 차분 기반 판정 파라미터(occlusion robust) ===
TH_DIFF_HARD   = 35        # 8-bit 강한 차분 임계(권장 30~45)
FRAC_HARD_MIN  = 0.30      # 강한 차분 픽셀 비율 최소(30% 이상이면 '크게 달라짐')
FRAC_DYN_MIN   = 0.55      # 동적 임계(노이즈 적응) 기준 비율
ERODE_ITERS    = 1         # 마스크 코어만 사용(경계 흔들림 억제). 0~1 권장

ROTATE_MAP = {
    2: cv2.ROTATE_90_COUNTERCLOCKWISE,  # LEFT
    3: cv2.ROTATE_90_CLOCKWISE,         # RIGHT
}

CAP_SIZE = (1280, 720)
size = CAP_SIZE