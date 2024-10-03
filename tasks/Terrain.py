import numpy as np
import matplotlib.pyplot as plt
from tkinter import *
import matplotlib.colors as mcolors

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class Terrain:
    def __init__(self, start: list[list[float]], end: list[list[float]], iterations: int, offset: float, display: list[int]):
        self.start = start
        self.end = end
        self.iterations = iterations
        self.offset = offset
        self.canvasSize = display

    def midpoint(self, x_start, x_end, y_start, y_end):
        if abs(x_end - x_start) <= 1:
            return [(x_start, y_start), (x_end, y_end)]

        x_mid = (x_start + x_end) / 2
        y_mid = (y_start + y_end) / 2

        y_mid += np.random.uniform(-self.offset, self.offset)

        left_points = self.midpoint(x_start, x_mid, y_start, y_mid)
        right_points = self.midpoint(x_mid, x_end, y_mid, y_end)

        return left_points[:-1] + right_points

    def generate_terrain(self):
        all_terrain_points = []

        for i in range(len(self.start)):
            terrain_points = [(self.start[i][0], self.start[i][1]), (self.end[i][0], self.end[i][1])]

            for _ in range(self.iterations):
                new_points = []
                for i in range(len(terrain_points) - 1):
                    x1, y1 = terrain_points[i]
                    x2, y2 = terrain_points[i + 1]
                    new_points.extend(self.midpoint(x1, x2, y1, y2))
                terrain_points = new_points

            all_terrain_points.append(terrain_points)

        return all_terrain_points


class TerrainGeneratorGUI:
    def __init__(self):
        self.root = Tk()
        self.root.title("Terrain Generator")

        # Variables
        self.start = []
        self.end = []
        self.colours = list(mcolors.BASE_COLORS.keys())
        self.canvasSize = [500, 500]
        self.iterations = DoubleVar(value=6)
        self.offset = DoubleVar(value=5)
        self.lines = IntVar(value=3)

        # Frame for input fields
        input_frame = Frame(self.root)
        input_frame.pack(side=RIGHT, padx=10, pady=10)

        # Canvas for plotting terrain
        canvas_frame = Frame(self.root, width=self.canvasSize[0], height=self.canvasSize[1])
        canvas_frame.pack(side=LEFT, padx=10, pady=10)

        self.figure = plt.figure(figsize=(5, 5))
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=canvas_frame)
        self.canvas.get_tk_widget().pack()

        # Labels
        x_label = Label(input_frame, text="X")
        x_label.grid(row=0, column=1, padx=5, pady=5)
        y_label = Label(input_frame, text="Y")
        y_label.grid(row=0, column=2, padx=5, pady=5)

        # Canvas size
        self.canvas_x = Entry(input_frame)
        self.canvas_x.grid(row=1, column=1, padx=5, pady=5)
        canvas_x_label = Label(input_frame, text="Canvas:")
        canvas_x_label.grid(row=1, column=0, padx=5, pady=5)

        self.canvas_y = Entry(input_frame)
        self.canvas_y.grid(row=1, column=2, padx=5, pady=5)

        # Start X entry
        self.start_x_entry = Entry(input_frame)
        self.start_x_entry.grid(row=2, column=1, padx=5, pady=5)
        start_x_label = Label(input_frame, text="Start:")
        start_x_label.grid(row=2, column=0, padx=5, pady=5)

        self.start_y_entry = Entry(input_frame)
        self.start_y_entry.grid(row=2, column=2, padx=5, pady=5)

        # End X entry
        self.end_x_entry = Entry(input_frame)
        self.end_x_entry.grid(row=3, column=1, padx=5, pady=5)
        end_x_label = Label(input_frame, text="End:")
        end_x_label.grid(row=3, column=0, padx=5, pady=5)

        self.end_y_entry = Entry(input_frame)
        self.end_y_entry.grid(row=3, column=2, padx=5, pady=5)

        # Color entry
        self.color_entry = Entry(input_frame)
        self.color_entry.grid(row=4, column=1, padx=5, pady=5)
        color_label = Label(input_frame, text="Color:")
        color_label.grid(row=4, column=0, padx=5, pady=5)

        # Variance entry
        self.variance_entry = Entry(input_frame, textvariable=self.offset)
        self.variance_entry.grid(row=5, column=1, padx=5, pady=5)
        variance_label = Label(input_frame, text="Variance:")
        variance_label.grid(row=5, column=0, padx=5, pady=5)

        # Iterations entry
        self.iterations_entry = Entry(input_frame, textvariable=self.iterations)
        self.iterations_entry.grid(row=6, column=1, padx=5, pady=5)
        iterations_label = Label(input_frame, text="Iterations:")
        iterations_label.grid(row=6, column=0, padx=5, pady=5)

        # Random Lines
        self.rand_lines_entry = Entry(input_frame, textvariable=self.lines)
        self.rand_lines_entry.grid(row=7, column=1, padx=5, pady=5)
        rand_lines_label = Label(input_frame, text="Random Lines:")
        rand_lines_label.grid(row=7, column=0, padx=5, pady=5)

        # Buttons
        change_canvas_button = Button(input_frame, text="Change Canvas size", command=self.change_canvas)
        change_canvas_button.grid(row=9, column=0, columnspan=1, padx=5, pady=5, sticky="we")

        add_line_button = Button(input_frame, text="Add Line Coord", command=self.add_line_coord)
        add_line_button.grid(row=9, column=1, columnspan=1, padx=5, pady=5, sticky="we")

        random_button = Button(input_frame, text="Random", command=self.set_random_values)
        random_button.grid(row=9, column=2, columnspan=1, padx=5, pady=5, sticky="we")

        change_variance_button = Button(input_frame, text="Change Variance", command=self.change_variance)
        change_variance_button.grid(row=10, column=0, columnspan=1, padx=5, pady=5, sticky="we")

        change_iter_button = Button(input_frame, text="Change Iteration", command=self.change_iterations)
        change_iter_button.grid(row=10, column=1, columnspan=1, padx=5, pady=5, sticky="we")

        change_lines_button = Button(input_frame, text="Change lines", command=self.change_lines)
        change_lines_button.grid(row=10, column=2, columnspan=1, padx=5, pady=5, sticky="we")

        plot_button = Button(input_frame, text="Plot", command=self.plot_terrain)
        plot_button.grid(row=12, column=0, columnspan=3, padx=5, pady=5, sticky="we")

        # Status label
        self.status_label = Label(input_frame, text="Status: Ready")
        self.status_label.grid(row=15, column=0, columnspan=3, padx=5, pady=5)

        self.root.mainloop()

    def add_line_coord(self):
        start_x = float(self.start_x_entry.get())
        start_y = float(self.start_y_entry.get())
        end_x = float(self.end_x_entry.get())
        end_y = float(self.end_y_entry.get())
        color = self.color_entry.get()

        self.start.append([start_x, start_y])
        self.end.append([end_x, end_y])
        self.colours.append(color)

        self.start_x_entry.delete(0, END)
        self.start_y_entry.delete(0, END)
        self.end_x_entry.delete(0, END)
        self.end_y_entry.delete(0, END)
        self.color_entry.delete(0, END)

        row=1
        self.status_label.config(text="Status: Line added")

    def change_variance(self):
        new_variance = float(self.variance_entry.get())
        self.offset.set(new_variance)

        self.status_label.config(text="Status: Line Offset changed")

    def change_iterations(self):
        new_iterations = float(self.iterations_entry.get())
        self.iterations.set(new_iterations)

        self.status_label.config(text="Status: Iterations changed")

    def change_lines(self):
        new_lines = int(self.rand_lines_entry.get())

        if new_lines > len(self.colours):
            new_lines = len(self.colours)
            self.rand_lines_entry.config(textvariable=self.lines)

        self.lines.set(new_lines)

        self.status_label.config(text="Status: Number of random lines changed")

    def change_canvas(self):
        new_width = int(self.canvas_x.get())
        new_height = int(self.canvas_y.get())
        self.canvasSize = [new_width, new_height]
        self.canvas.get_tk_widget().config(width=new_width, height=new_height)

        self.status_label.config(text="Status: Canvas size changed")

    def set_random_values(self):
        borders = [i*100 for i in range(self.lines.get())]
        borders.reverse()
        tmp_x = self.canvasSize[0]
        tmp_y = self.canvasSize[1]

        self.start.clear()
        self.end.clear()

        for i in range(self.lines.get()):
            tmp_x = np.random.randint(borders[i], tmp_x)
            tmp_y = np.random.randint(borders[i], tmp_y)

            self.start.append([0, tmp_x])
            self.end.append([self.canvasSize[1], tmp_y])

        self.status_label.config(text="Status: Random values set")

    def plot_terrain(self):
        if not self.start or not self.end:
            self.status_label.config(text="Status: Please add coordinates first")
            return

        self.ax.clear()
        self.canvas.draw()

        terrain = Terrain(self.start, self.end, int(self.iterations.get()), self.offset.get(), [1000, 1000])

        terrain_points = terrain.generate_terrain()

        for i, terrain_points in enumerate(terrain_points):
            x_values = [point[0] for point in terrain_points]
            y_values = [point[1] for point in terrain_points]

            self.ax.plot(x_values, y_values, color=self.colours[i])  # Change color here
            self.ax.fill_between(x_values, y_values, color=self.colours[i])  # Fill below the line

        self.ax.set_title('2D Terrain Generated Using Fractal Noise')
        self.ax.set_xlabel('X-coordinate')
        self.ax.set_ylabel('Y-coordinate')

        self.canvas.draw()

        self.status_label.config(text="Status: Terrain plotted")

