import numpy as np


class IFS:
    def __init__(self):
        self.points = None
        self.transformations = None
        self.bias = None

    def start(self, tp: int, pop_size: int = 100):
        if tp == 0:
            self.init_first()
        elif tp == 1:
            self.init_second()

        self.points = [np.array([0, 0, 0])]

        for _ in range(pop_size):
            choice = np.random.choice([0, 1, 2, 3])
            self.points.append(np.dot(self.transformations[choice], self.points[-1]) + self.bias[choice])

    def init_first(self):
        tr1 = np.array([0.00,  0.00,  0.01,  0.00,  0.26,  0.00,  0.00,  0.00,  0.05]).reshape(3, 3)
        tr2 = np.array([0.20, -0.26, -0.01,  0.23,  0.22, -0.07,  0.07,  0.00,  0.24]).reshape(3, 3)
        tr3 = np.array([-0.25,  0.28,  0.01,  0.26,  0.24, -0.07,  0.07,  0.00,  0.24]).reshape(3, 3)
        tr4 = np.array([0.85,  0.04, -0.01, -0.04,  0.85,  0.09,  0.00,  0.08,  0.84]).reshape(3, 3)

        b1 = np.array([0.00,  0.00,  0.00])
        b2 = np.array([0.00,  0.80,  0.00])
        b3 = np.array([0.00,  0.22,  0.00])
        b4 = np.array([0.00,  0.80,  0.00])

        self.transformations = [tr1, tr2, tr3, tr4]
        self.bias = [b1, b2, b3, b4]

    def init_second(self):
        tr1 = np.array([0.05,  0.00, 0.00, 0.00, 0.60, 0.00, 0.00, 0.00, 0.05]).reshape(3, 3)
        tr2 = np.array([0.45, -0.22, 0.22, 0.22, 0.45, 0.22, -0.22, 0.22, -0.45]).reshape(3, 3)
        tr3 = np.array([-0.45, 0.22, -0.22, 0.22, 0.45, 0.22, 0.22, -0.22, 0.45]).reshape(3, 3)
        tr4 = np.array([0.49, -0.08, 0.08, 0.08, 0.49, 0.08, 0.08, -0.08, 0.49]).reshape(3, 3)

        b1 = np.array([0.00, 0.00, 0.00])
        b2 = np.array([0.00, 1.00, 0.00])
        b3 = np.array([0.00, 1.25, 0.00])
        b4 = np.array([0.00, 2.00, 0.00])

        self.transformations = [tr1, tr2, tr3, tr4]
        self.bias = [b1, b2, b3, b4]
