import cv2
import numpy as np
from ultralytics import YOLO
from realsense_adapter import RealSenseColorDepth

# YOLO 모델 경로
MODEL_PATH = r"C:\Users\jshkr\OneDrive\문서\JSH_CAPSTONE_CODE\windows\param\best_6.pt"

# RealSense RGB 스트림 시작
cap = RealSenseColorDepth(color=(1280, 720, 30), depth=(848, 480, 30), align_to_color=True, rotate90=False)
model = YOLO(str(MODEL_PATH))

print("[Info] RealSense + YOLO 세그멘테이션 테스트 시작")
cv2.namedWindow("YOLO Hold Detection", cv2.WINDOW_NORMAL)

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            print("[Warn] 프레임 읽기 실패")
            break

        # YOLO 예측
        results = model.predict(frame, conf=0.5, verbose=False)

        vis = frame.copy()

        for r in results:
            masks = r.masks
            if masks is None:
                continue

            for i, seg in enumerate(masks.xy):
                # 윤곽선 좌표를 정수형으로 변환
                contour = np.array(seg, dtype=np.int32).reshape(-1, 1, 2)
                color = (0, 255, 0)
                cv2.drawContours(vis, [contour], -1, color, 2)
                # 중심점 계산
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                    cv2.circle(vis, (cx, cy), 4, (0, 0, 255), -1)
                    cv2.putText(vis, f"Hold{i}", (cx - 10, cy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("YOLO Hold Detection", vis)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
