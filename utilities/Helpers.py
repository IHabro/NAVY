from enum import Enum

import numpy as np
from tasks.L_systems import LSystems


class Task(Enum):
    Task01 = PERCEPTRON = 1
    Task02 = XOR = 2
    Task03 = HOPFIELD = 3
    Task04 = CHEESE = 4
    Task05 = BALANCING = 5
    Task06 = LSYSTEMS = 6
    Task07 = IFS = 7
    Task08 = FRACTALS = 8
    Task09 = COUNTRY = 9
    Task10 = 10
    Task11 = PENDULUM = 11
    Task12 = 12

class ActivationFunction(Enum):
    Signum = 1
    Sigmoid = 2
    ReLU = 3
    Softplus = 4


class PointND:
    def __init__(self, coordinates: list[float]):
        self.coordinates = coordinates
        self.dimension = len(coordinates)

    def classify_perceptron_point(self) -> int:
        if self.dimension == 2:
            # y = 3x + 2
            result = 3 * self.coordinates[0] + 2
            y = self.coordinates[1]

            if y > result:
                return 1
            elif y < result:
                return -1
            elif y == result:
                return 0

    def classify_xor_point(self):
        if self.dimension == 2:
            x, y = self.coordinates

            if x == y:
                return 0
            else:
                return 1


def activate_function_point(point: PointND, function: ActivationFunction, weights: list[float], offset: float) -> float:
    if function == ActivationFunction.Signum:
        return np.sign(point.coordinates[0]*weights[0] + point.coordinates[1]*weights[1] + offset)
    elif function == ActivationFunction.Sigmoid:
        return 1 / (1 + np.exp(point.coordinates[0]))
        # version with 1 / (1 + e^(-x))
        # return 1.0 / (1 + np.exp(-1 * point.coordinates[0]))
    elif function == ActivationFunction.ReLU:
        return max(0.0, point.coordinates[0])
    elif function == ActivationFunction.Softplus:
        return np.log(1 + np.exp(point.coordinates[0]))

    raise Exception("Wrong function ID")


def activate_function_x(x: float, function: ActivationFunction) -> float:
    if function == ActivationFunction.Signum:
        return np.sign(x)
    elif function == ActivationFunction.Sigmoid:
        return 1 / (1 + np.exp(-x))
        # version with 1 / (1 + e^(-x))
        # return 1.0 / (1 + np.exp(-1 * point.coordinates[0]))
    elif function == ActivationFunction.ReLU:
        return 0
    elif function == ActivationFunction.Softplus:
        return 0

    raise Exception("Wrong function ID")


def derivative_activate(x: float, function: ActivationFunction) -> float:
    if function == ActivationFunction.Signum:
        return 0
    elif function == ActivationFunction.Sigmoid:
        f = activate_function_x(x, function)
        return f * (1 - f)
    elif function == ActivationFunction.ReLU:
        return 0
    elif function == ActivationFunction.Softplus:
        return 0

    raise Exception("Wrong function ID")


def draw_l_system(turtle_obj, system: LSystems):
    path = system.drawString
    distance = system.length
    angle = system.angle

    heading = 0
    stack = []
    for cmd in path:
        if cmd == 'F':
            turtle_obj.setheading(heading)
            turtle_obj.forward(distance)
        elif cmd == '+':
            heading += angle  # Update heading for '+'
        elif cmd == '-':
            heading -= angle  # Update heading for '-'
        elif cmd == '[':
            # Save current heading
            stack.append((turtle_obj.position(), turtle_obj.heading()))
        elif cmd == ']':
            # Restore previous position and heading
            position, heading = stack.pop()
            turtle_obj.penup()
            turtle_obj.goto(position)
            turtle_obj.setheading(heading)
            turtle_obj.pendown()


class Parameters:
    pass
