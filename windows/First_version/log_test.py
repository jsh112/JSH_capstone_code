import pandas as pd
csv_path = r"/windows/First_version/log_dir/metric_occlusion_log_20251021_164330.csv"
pred = pd.read_csv(csv_path)

# (예시) 각 부위 등장 빈도
print(pred["blocked_part"].value_counts())

# (예시) 특정 홀드에서 가려진 부위 시각화
print(pred[pred["hold_id"] == 1][["frame_id", "blocked_part"]])

# (예시) 실험 시간축으로 변환
t0 = pred["timestamp"].min()
pred["elapsed_s"] = pred["timestamp"] - t0
print(pred.head())