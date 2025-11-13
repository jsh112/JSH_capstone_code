import cv2
import numpy as np

def test_dual_cameras(left_idx=1, right_idx=2, width=1280, height=720):
    capL = cv2.VideoCapture(left_idx, cv2.CAP_DSHOW)
    capR = cv2.VideoCapture(right_idx, cv2.CAP_DSHOW)
    for cap in (capL, capR):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not capL.isOpened() or not capR.isOpened():
        print("❌ 카메라 오픈 실패 — 인덱스 또는 연결 확인")
        return

    print("✅ 좌/우 카메라 모두 열림 (Q 키로 종료)")

    while True:
        ok1, f1 = capL.read()
        ok2, f2 = capR.read()
        if not ok1 or not ok2:
            print("⚠️ 프레임 읽기 실패")
            break

        stacked = np.hstack([f1, f2])
        cv2.imshow("Stereo Cameras (L | R)", stacked)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capL.release(); capR.release()
    cv2.destroyAllWindows()

def list_available_cameras(max_index=10):
    available = []
    print("🔍 연결된 카메라 스캔 중...")

    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # Windows는 CAP_DSHOW, Linux면 CAP_V4L2로 변경
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                h, w = frame.shape[:2]
                print(f"✅ 카메라 인덱스 {i} 사용 가능 ({w}x{h})")
                available.append(i)
            cap.release()
        else:
            print(f"❌ 카메라 인덱스 {i} 사용 불가")

    if not available:
        print("⚠️ 사용 가능한 카메라를 찾지 못했습니다.")
    else:
        print(f"\n📸 사용 가능한 카메라 인덱스: {available}")

    return available

from pygrabber.dshow_graph import FilterGraph

def list_cameras(verbose=True):
    """
    Windows DirectShow 기반 카메라 목록을 반환합니다.
    (인덱스 번호와 장치 이름을 튜플 리스트로 리턴)

    Args:
        verbose (bool): True면 콘솔에 즉시 출력
    Returns:
        list[tuple[int, str]]: 예) [(0, 'Integrated Camera'), (1, 'Logitech C920')]
    """
    graph = FilterGraph()
    devices = graph.get_input_devices()

    camera_list = [(i, name) for i, name in enumerate(devices)]

    if verbose:
        print("🔍 연결된 카메라 장치 목록:")
        for i, name in camera_list:
            print(f"  [{i}] {name}")

    return camera_list

from ultralytics import YOLO
def hello():
    model = YOLO("./param/best_6.pt")  # 또는 커스텀 weight
    model.export(format="onnx", opset=12)

# 단독 실행 시 목록 확인
if __name__ == "__main__":
    hello()
