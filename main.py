import pandas as pd
import numpy as np
from src.online_logistic import OnlineLogisticRegression
from src.metrics import SlidingWindowMetrics
from src.streaming import stream_csv

PATH = "data/stream.csv"
TARGET_COL = "target"

df = pd.read_csv(PATH)
features = [c for c in df.columns if c != TARGET_COL]
n_features = len(features)

model = OnlineLogisticRegression(n_features=n_features, lr=0.01)
metrics = SlidingWindowMetrics(window_size=100)

for i, row in enumerate(stream_csv(PATH), start=1):
    x = np.array(row[features], dtype=float)
    y = int(row[TARGET_COL])

    y_pred = model.predict(x)
    metrics.update(y, y_pred)
    model.update(x, y)

    if i % 1000 == 0:
        stats = metrics.get_metrics()
        print(f"Seen: {i} samples | Accuracy = {stats['accuracy']:.4f} | F1 = {stats['f1']:.4f}")
