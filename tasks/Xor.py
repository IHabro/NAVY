import random
import copy

from utilities.Helpers import PointND, ActivationFunction, derivative_activate, activate_function_x


class Xor:
    def __init__(self, training: list[PointND], hidden_weights: list[list[float]] = None, output_weights: list[float] = None, hidden_offsets: list[float] = None, output_offset: float = None, rate: float = 0.1, function: ActivationFunction = ActivationFunction.Sigmoid):
        if len(training) == 0:
            return

        self.training = training

        self.learningRate = rate
        self.function = function

        self.hiddenW = hidden_weights
        self.hiddenO = hidden_offsets

        self.outputW = output_weights
        self.outputO = output_offset

        if hidden_weights is None or output_weights is None or hidden_offsets is None or output_offset is None:
            self.generate_parameters()

        self.resultH = []
        self.resultO = None

        self.error = None
        self.derivative_error = None

    def generate_parameters(self):
        self.hiddenW = [[random.uniform(0, 1) for _ in range(2)] for _ in range(2)]
        self.outputW = [random.uniform(0, 1) for _ in range(2)]
        self.hiddenO = [random.uniform(0, 1) for _ in range(2)]
        self.outputO = random.uniform(0, 1)

    def train(self, epochs: int):
        for _ in range(epochs):
            for point in self.training:
                self.calculate_hidden(point)
                self.calculate_output()
                self.calculate_error(point)
                self.recalculate_weights(point)

    def test(self) -> list[float]:
        result = []

        for p in self.training:
            result.append(self.guess_result(p))

        return result

    def guess_result(self, point) -> float:
        self.calculate_hidden(point)
        self.calculate_output()

        return copy.deepcopy(self.resultO)

    def calculate_hidden(self, point: PointND):
        self.resultH.clear()

        for i in range(len(self.hiddenW)):
            sm = self.hiddenO[i]            # sum = offset_i

            for weight_i, input_i in zip(self.hiddenW[i], point.coordinates):
                sm += weight_i * input_i    # sum += weight_i * input_i

            self.resultH.append(activate_function_x(sm, self.function))         # out_H.add(sum)

    def calculate_output(self):
        if len(self.resultH) != 2:
            raise Exception("Mismatch in size of hidden layer result expected: {}, god: {}".format(2, len(self.resultH)))

        # Same logic as in calculate_hidden, just for 1 single neuron
        sm = self.outputO

        for weight_i, hidden_i in zip(self.outputW, self.resultH):
            sm += weight_i * hidden_i

        self.resultO = activate_function_x(sm, self.function)

    def calculate_error(self, point: PointND):
        # Total error = y_guess, because we do not have more output neurons
        # error = 1/2*(y_target - y_guess)^2
        self.error = 1.0/2.0 * (point.classify_xor_point() - self.resultO)**2
        self.derivative_error = -1*(point.classify_xor_point() - self.resultO)

    def recalculate_weights(self, point: PointND):
        # 1. calculate Delta resultO and Delta resultH
        # 2. recalculate weights

        # Deltas
        deltaO = self.derivative_error * derivative_activate(self.resultO, self.function)
        deltaH = []

        for rst, weight in zip(self.resultH, self.outputW):
            tmp = deltaO * weight * derivative_activate(rst, self.function)

            deltaH.append(tmp)

        # Output weights update:
        self.outputW = [wgt + self.learningRate * inp * deltaO for wgt, inp in zip(self.outputW, self.resultH)]

        # Hidden weights update:
        lst = []
        for w, delta in zip(self.hiddenW, deltaH):
            tmp = []

            for weight, value in zip(w, point.coordinates):
                tmp.append(weight + self.learningRate * value * delta)

            lst.append(tmp)

        self.hiddenW = copy.deepcopy(lst)

        # Output offset update:
        self.outputO = self.outputO + self.learningRate * deltaO

        # Hidden offset update:
        a = []
        for off, delta in zip(self.hiddenO, deltaH):
            a.append(off + self.learningRate * delta)

        self.hiddenO = copy.deepcopy(a)
