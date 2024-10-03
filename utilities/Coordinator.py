import random
import numpy as np

from utilities.Helpers import Task, PointND
from utilities.CustomDisplays import CustomDisplays
from tasks.Perceptron import Perceptron
from tasks.Hopfield import Hopfield
from tasks.Qlearning import Qlearning
from tasks.Xor import Xor
from tasks.L_systems import LSystems
from tasks.Fractals import Mandelbrot, Julia
from tasks.Terrain import TerrainGeneratorGUI
from tasks.Pendulum import Pendulum


class Coordinator:
    def __init__(self, task: Task, pop_size: int, canvas_size: list[int], dimension: int):
        self.task = task
        self.popSize = pop_size
        self.canvasSize = canvas_size
        self.canvas = CustomDisplays(canvas_size)
        self.dimension = dimension

    def generate_population_norm(self) -> list[PointND]:
        return [PointND([random.randint(0, self.canvasSize[0]), random.randint(0, self.canvasSize[1])]) for _ in range(self.popSize)]

    def generate_population_xor(self) -> list[PointND]:
        # Form of data storage:
        # [x1, x2]
        return [PointND([0, 0]), PointND([0, 1]), PointND([1, 0]), PointND([1, 1])]

    def generate_patterns(self) -> list[np.ndarray]:
        pattern1 = np.array([[1, 0, 0, 0, 1], [0, 1, 0, 1, 0], [0, 0, 1, 0, 0], [0, 1, 0, 1, 0], [1, 0, 0, 0, 1]])
        pattern2 = np.array([[0, 1, 1, 1, 0], [0, 1, 0, 1, 0], [0, 1, 1, 1, 0], [0, 1, 0, 1, 0], [0, 1, 1, 1, 0]])
        pattern3 = np.array([[1, 1, 1, 1, 1], [1, 0, 0, 0, 1], [1, 0, 1, 0, 1], [1, 0, 0, 0, 1], [1, 1, 1, 1, 1]])

        return [pattern1, pattern2, pattern3]

    def generate_flawed_pattern(self) -> list[np.ndarray]:
        pattern1 = np.array([[1, 0, 0, 0, 1], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 1, 0, 1, 0], [0, 0, 0, 0, 0]])
        pattern2 = np.array([[0, 1, 0, 1, 0], [0, 0, 0, 0, 0], [0, 1, 0, 1, 0], [0, 0, 0, 0, 0], [0, 1, 0, 1, 0]])
        pattern3 = np.array([[0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [1, 0, 1, 0, 1], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0]])

        return [pattern1, pattern2, pattern3]

    def change_task(self, task: Task):
        self.task = task

    def start_task(self, epochs: int = 100, shape: int = 5, use_async: bool = True):
        if self.task == Task.Task01:
            self.start_perceptron(epochs)
        elif self.task == Task.Task02:
            self.start_xor(epochs)
        elif self.task == Task.Task03:
            self.start_hopfield(epochs, shape, use_async)
        elif self.task == Task.Task04:
            self.start_qlearning()
        elif self.task == Task.Task05:
            self.start_pole_balancing()
        elif self.task == Task.Task06:
            self.start_lsystem()
        elif self.task == Task.Task07:
            self.start_ifs()
        elif self.task == Task.Task08:
            self.start_tea()
        elif self.task == Task.Task09:
            self.start_terrain()
        elif self.task == Task.Task11:
            self.start_pendulum()

    def start_perceptron(self, epochs: int):
        task01 = Perceptron(self.generate_population_norm(), self.generate_population_norm())
        task01.train(epochs)
        task01.test()

        self.canvas.perceptron_print(task01)

    def start_xor(self, epochs: int):
        task02 = Xor(self.generate_population_xor())

        self.canvas.xor_print(task02, "before learning")
        task02.train(epochs)
        self.canvas.xor_print(task02, "after learning", True)

    def start_hopfield(self, epochs: int, shape: int, use_async: bool):
        task03 = Hopfield(use_async)
        self.canvas.hopfield_qlearning_print(task03, None)

    def start_qlearning(self):
        task04 = Qlearning()
        self.canvas.hopfield_qlearning_print(None, task04)

    def start_pole_balancing(self):
        task05 = None

    def start_lsystem(self):
        task06 = LSystems("F+F+F+F", "F+F-F-FF+F+F-F", 90, 3, 5)
        task06 = LSystems("F++F++F", "F+F--F+F", 60, 3, 15)
        task06 = LSystems("F", "FF+[+F-F-F]-[-F+F+F]", 22.5, 3, 5)
        task06 = LSystems("F", "F[+F]F[-F]F", 25.7143, 3, 10)
        self.canvas.l_systems_print(task06)

    def start_ifs(self):
        self.canvas.ifs_print()

    def start_tea(self):
        print("Mandelbrot")

        task08 = Mandelbrot()
        task08.plot()

        print("Julia")

        task08 = Julia()
        task08.plot()

    def start_terrain(self):
        start = [[0, 600], [0, 500], [0, 300]]
        end = [[1000, 900], [1000, 400], [1000, 300]]
        pltSize = [1000, 1000]
        colours = ["green", "black", "brown"]

        task09 = TerrainGeneratorGUI()

    def start_pendulum(self):
        task11 = Pendulum()
        task11.start()
