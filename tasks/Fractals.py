import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class Mandelbrot():
    def __init__(self, max_iter: int = 512, threshold: float = 2.0):
        self.maxIter = max_iter
        self.threshold = threshold

    def mandelbrot(self, c):
        z = 0

        for n in range(self.maxIter):
            if abs(z) > self.threshold:
                return n

            z = z**2 + c

        return self.maxIter

    def start(self, xmin: float = -2, xmax: float = 1, ymin: float = -1, ymax: float = 1, width: float = 800, height: float = 800) -> np.ndarray:
        x = np.linspace(xmin, xmax, width)
        y = np.linspace(ymin, ymax, height)
        pixels = np.zeros((height, width))

        for j in range(height):
            for i in range(width):
                pixels[j, i] = self.mandelbrot(x[i] + 1j * y[j])

        return pixels

    def plot(self):
        xmin, xmax, ymin, ymax, width, height = -2.0, 1.0, -1.5, 1.5, 7680, 4320

        mandelbrot_image = self.start(xmin, xmax, ymin, ymax, width, height)

        plt.imsave('_mandelbrot_set_8k.png', mandelbrot_image, cmap='plasma')

        plt.imshow(mandelbrot_image, extent=(xmin, xmax, ymin, ymax), cmap='plasma')
        plt.title("Mandelbrot Set Plasma")
        plt.xlabel("Real")
        plt.ylabel("Imaginary")
        plt.show()


class Julia():
    def __init__(self, max_iter: int = 256, threshold: float = 2.0, c: complex = complex(-0.7, 0.27015)):
        self.maxIter = max_iter
        self.threshold = threshold
        self.c = c

    def julia(self, z):
        for n in range(self.maxIter):
            if abs(z) > self.threshold:
                return n

            z = z * z + self.c

        return self.maxIter

    def start(self, xmin: float = -2, xmax: float = 1, ymin: float = -1, ymax: float = 1, width: float = 800, height: float = 800) -> np.ndarray:
        x = np.linspace(xmin, xmax, width)
        y = np.linspace(ymin, ymax, height)
        pixels = np.zeros((height, width))

        for j in range(height):
            for i in range(width):
                pixels[j, i] = self.julia(x[i] + 1j * y[j])

        return pixels

    def plot(self):
        xmin, xmax, ymin, ymax, width, height = -1.5, 1.5, -1.5, 1.5, 7680, 4320

        julia_image = self.start(xmin, xmax, ymin, ymax, width, height)

        plt.imsave('_julia_set_8k.png', julia_image, cmap='plasma')

        plt.imshow(julia_image, extent=(xmin, xmax, ymin, ymax), cmap='plasma')
        plt.title("Julia Set Plasma")
        plt.xlabel("Real")
        plt.ylabel("Imaginary")
        plt.show()
