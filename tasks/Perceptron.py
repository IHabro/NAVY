import random

from utilities.Helpers import PointND, activate_function_x, ActivationFunction


class Perceptron:
    def __init__(self, training: list[PointND], testing: list[PointND], rate: float = 0.1, function: ActivationFunction = ActivationFunction.Signum, weights: list[float] = None, offset: float = 0.5):
        if len(training) == 0 or len(testing) == 0 or training[0].dimension != testing[0].dimension:
            return

        self.training = training
        self.testing = testing
        self.learning_rate = rate
        self.function = function
        self.weights = None     # w1, w2
        self.offset = offset    # b
        self.dimension = training[0].dimension

        if weights is None:
            self.generate_weights()
        else:
            self.weights = weights

    def generate_weights(self):
        self.weights = [random.uniform(-1, 1) for _ in range(self.dimension)]

    def train(self, epochs: int):
        for i in range(epochs):
            for point in self.training:
                # Error calculation -> y - y_guess
                y_right = point.classify_perceptron_point()
                y_guess = activate_function_x(point, self.function, self.weights, self.offset)
                error = y_right - y_guess

                # Weights recalculation
                for i in range(len(self.weights)):
                    self.weights[i] = self.weights[i] + error * point.coordinates[i] * self.learning_rate

                self.offset = self.offset + error * self.learning_rate

    def test(self):
        for p in self.testing:
            p.classification = activate_function_x(p, self.function, self.weights, self.offset)
