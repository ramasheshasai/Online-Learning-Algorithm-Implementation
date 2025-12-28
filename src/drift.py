from collections import deque
import numpy as np


class DriftDetector:
    def __init__(self, window=200):
        self.window = window
        self.errors = deque(maxlen=window)

    def update(self, y_true, y_pred):
        self.errors.append(int(y_true != y_pred))

        if len(self.errors) == self.window:
            mean = np.mean(self.errors)
            return mean > 0.3   # drift threshold
        return False
