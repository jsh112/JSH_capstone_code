import numpy as np

def print_npz_calibration_info(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    print(f"\n📁 NPZ 파일: {npz_path}")
    print("────────────────────────────────────")

    # 어떤 키들이 들어 있는지
    print("[Keys in file]")
    for key in data.keys():
        val = data[key]
        shape = getattr(val, "shape", None)
        print(f" - {key:<6}  shape={shape}  dtype={val.dtype if hasattr(val, 'dtype') else type(val)}")

    print("────────────────────────────────────")

    # 각 주요 행렬 값 출력
    for name in ["K1", "D1", "K2", "D2", "R", "T", "P1", "P2"]:
        if name in data:
            val = data[name]
            print(f"\n{name}:")
            print(np.array(val))

    # baseline 확인
    if "T" in data:
        T = np.array(data["T"]).reshape(-1)
        baseline = np.linalg.norm(T)
        print("\n📏 Baseline distance (|T|): {:.3f} mm (or units used in calibration)".format(baseline))

    print("────────────────────────────────────")
    print("✅ 캘리브레이션 파일 로드 완료.\n")

if __name__ == "__main__":
    print_npz_calibration_info("./stereo_params_scaled_1012.npz")