import matplotlib.pyplot as plt


class LiveVisualizer:
    def __init__(self):
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.x = []
        self.acc = []

    def update(self, step, accuracy):
        self.x.append(step)
        self.acc.append(accuracy)

        self.ax.clear()
        self.ax.plot(self.x, self.acc)
        self.ax.set_title("Real Time Accuracy")
        self.ax.set_xlabel("Samples Seen")
        self.ax.set_ylabel("Accuracy")
        plt.pause(0.01)
