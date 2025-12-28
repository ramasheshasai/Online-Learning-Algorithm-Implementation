import pandas as pd
import numpy as np
from src.online_logistic import OnlineLogisticRegression
from src.metrics import SlidingWindowMetrics
from src.streaming import stream_csv
from src.drift import DriftDetector
from src.visualizer import LiveVisualizer

PATH = "data/stream.csv"
TARGET_COL = "target"

df = pd.read_csv(PATH)
features = [c for c in df.columns if c != TARGET_COL]

model = OnlineLogisticRegression(n_features=len(features), lr=0.01)
metrics = SlidingWindowMetrics(window_size=200)
drift = DriftDetector(window=200)
viz = LiveVisualizer()

for i, row in enumerate(stream_csv(PATH), start=1):
    x = np.array(row[features], dtype=float)
    y = int(row[TARGET_COL])


    pred = model.predict(x)
    metrics.update(y, pred)
    model.update(x, y)

    if drift.update(y, pred):
        print("⚠️ Concept Drift Detected — Resetting Model")
        model = OnlineLogisticRegression(n_features=len(features), lr=0.01)

    if i % 1000 == 0:
        stats = metrics.get_metrics()
        print(f"Seen {i} | {stats}")
        viz.update(i, stats["accuracy"])
