from collections import deque
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

class SlidingWindowMetrics:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.true = deque(maxlen=window_size)
        self.pred = deque(maxlen=window_size)

    def update(self, y_true, y_pred):
        self.true.append(y_true)
        self.pred.append(y_pred)

    def get_metrics(self):
        if len(self.true) == 0:
            return {}
        acc = accuracy_score(self.true, self.pred)
        f1 = f1_score(self.true, self.pred)
        prec = precision_score(self.true, self.pred)
        rec = recall_score(self.true, self.pred)

        return {
            "accuracy": acc,
            "f1": f1,
            "precision": prec,
            "recall": rec
        }
