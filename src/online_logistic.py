import numpy as np

class OnlineLogisticRegression:
    def __init__(self, n_features, lr=0.01, decay=True):
        self.lr = lr
        self.initial_lr = lr
        self.decay = decay
        self.t = 1
        self.weights = np.zeros(n_features)
        self.bias = 0

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))


    def predict_proba(self, x):
        z = np.dot(x, self.weights) + self.bias
        return self.sigmoid(z)

    def predict(self, x):
        return (self.predict_proba(x) >= 0.5).astype(int)

    def update(self, x, y):
        y_pred = self.predict_proba(x)
        error = y_pred - y

        lr = self.lr
        if self.decay:
            lr = self.initial_lr / np.sqrt(self.t)

        self.weights -= lr * error * x
        self.bias -= lr * error
        self.t += 1
