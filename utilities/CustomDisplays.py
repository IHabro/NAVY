import turtle

import matplotlib.pyplot as plt
import tkinter as tk
import numpy as np
from tkinter import Canvas, Button, Label

from utilities.Helpers import draw_l_system

from tasks.Perceptron import Perceptron
from tasks.Xor import Xor
from tasks.Hopfield import Hopfield
from tasks.Qlearning import Qlearning
from tasks.L_systems import LSystems
from tasks.IFS import IFS


class CustomDisplays:
    def __init__(self, size: list[int]):
        self.size = size

    def perceptron_print(self, perceptron: Perceptron):
        plt.figure(figsize=(5, 5))

        if len(perceptron.training) == 0 or perceptron.dimension != 2:
            return

        # points
        for p in perceptron.training:
            classification = p.classify_perceptron_point()

            if classification == 1:
                plt.scatter(p.coordinates[0], p.coordinates[1], color='green')
            elif classification == 0:
                plt.scatter(p.coordinates[0], p.coordinates[1], color='red')
            elif classification == -1:
                plt.scatter(p.coordinates[0], p.coordinates[1], color='blue')

        # line
        x_vals = [i for i in range(self.size[0])]
        y_vals = [3 * x + 2 for x in x_vals]
        plt.plot(x_vals, y_vals, color='black')

        # Set plot limits and labels
        plt.xlabel('X')
        plt.ylabel('Y')

        # Display the plot
        plt.grid(True)
        plt.title('Perceptron')
        plt.show()

    def xor_print(self, xor: Xor, comment: str, tmp: bool = False):
        print("Xor statistics {}:".format(comment))
        for i in range(2):
            print("Weights and Offset for hidden node {}: {}, {}".format(i + 1, xor.hiddenW[i], xor.hiddenO[i]))
        print("Weights and Offset for output node: {}, {}\n".format(xor.outputW, xor.outputO))

        if tmp:
            lst = [0, 1, 1, 0]
            print("Guess \t\t\t\t Expected \t Equal")
            for gss, exp in zip(xor.test(), lst):
                print("{} \t {} \t\t\t {}".format(gss, exp, round(gss) == exp))

    def hopfield_qlearning_print(self, hop: Hopfield = None, mouse: Qlearning = None):
        # Example usage
        root = tk.Tk()

        # Create a sample matrix
        sample_matrix = np.zeros((5, 5), dtype=int)

        # Initialize the interactive bitmap display
        ibd = InteractiveBitmapDisplay(root, sample_matrix, 5, 5, 100, hop, mouse)

        root.mainloop()

    def l_systems_print(self, system: LSystems):
        # Initialize turtle graphics
        screen = turtle.Screen()
        screen.setup(width=1000, height=1000)
        screen.bgcolor("white")
        screen.title("L-system Visualization")
        turtle_obj = turtle.Turtle()
        turtle_obj.speed(0)
        turtle_obj.penup()
        turtle_obj.setpos(-250, -250)
        turtle_obj.pendown()

        # Draw L-system
        draw_l_system(turtle_obj, system)

        # Hide turtle and display result
        turtle_obj.hideturtle()
        screen.mainloop()

    def ifs_print(self, ifs: IFS = IFS()):
        for i in range(2):
            ifs.start(i)
            points = np.vstack(ifs.points)

            # Extract x, y, and z coordinates from points
            x = points[:, 0]
            y = points[:, 1]
            z = points[:, 2]

            # Create a 3D scatter plot
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(x, y, z)

            # Set labels and title
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            plt.title('3D Scatter Plot')

            # Show plot
            plt.show()


class InteractiveBitmapDisplay:
    def __init__(self, parent, matrix, rows, columns, pixel_size, hopfiel: Hopfield, mouse: Qlearning):
        self.parent = parent
        self.matrix = matrix
        self.pixel_size = pixel_size
        self.rows = rows
        self.columns = columns

        self.click_counts = [[0] * columns for _ in range(rows)]
        self.colors = ["white", "black", "grey", "red", "yellow"]
        self.max_clicks = len(self.colors) - 1

        self.canvas_width = self.columns * self.pixel_size
        self.canvas_height = self.rows * self.pixel_size

        self.canvas = Canvas(self.parent, width=self.canvas_width, height=self.canvas_height)
        self.canvas.grid(row=0, column=0, columnspan=2)

        self.save_button = Button(self.parent, text="Save pattern", command=self.save_pattern)
        self.save_button.grid(row=1, column=0)

        self.clear_button = Button(self.parent, text="Clear Canvas", command=self.clear_canvas)
        self.clear_button.grid(row=2, column=0, columnspan=2)

        self.restore_button = Button(self.parent, text="Restore pattern", command=self.restore_pattern)
        self.restore_button.grid(row=1, column=1)

        self.status_label = Label(self.parent, text="Status: Ready")
        self.status_label.grid(row=3, column=0, columnspan=2)

        self.matrix = [[2, 1, 0, 0, 0], [0, 1, 0, 3, 0], [0, 1, 0, 1, 0], [0, 1, 0, 1, 0], [0, 0, 0, 1, 4]]

        self.display_matrix(self.matrix)
        self.canvas.bind("<Button-1>", self.toggle_pixel)

        self.hopfield = hopfiel
        self.mouse = mouse

        # Create buttons for training and releasing mouse
        self.train_button = Button(self.parent, text="Train Mouse", command=self.train_mouse)
        self.train_button.grid(row=4, column=0)

        self.release_button = Button(self.parent, text="Release Mouse", command=self.release_mouse)
        self.release_button.grid(row=4, column=1)

    def display_matrix(self, matrix):
        self.canvas.delete("all")
        for i in range(self.rows):
            for j in range(self.columns):
                color_index = matrix[i][j]
                color = self.colors[color_index]
                x0 = j * self.pixel_size
                y0 = i * self.pixel_size
                x1 = x0 + self.pixel_size
                y1 = y0 + self.pixel_size
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="gray")

    def toggle_pixel(self, event):
        col = event.x // self.pixel_size
        row = event.y // self.pixel_size

        # Increment click count for the pixel
        self.click_counts[row][col] = (self.click_counts[row][col] + 1) % (self.max_clicks + 1)

        # Get the color based on the click count
        color_index = self.click_counts[row][col]
        color = self.colors[color_index]

        # Update matrix and display
        self.matrix[row][col] = color_index
        self.display_matrix(self.matrix)

    def clear_canvas(self):
        try:
            self.matrix = np.zeros((self.rows, self.columns), dtype=int)
            self.display_matrix(self.matrix)

            self.status_label.config(text="Status: Cleared")
        except:
            self.status_label.config(text="Status: Jsi v piči clear")

    def save_pattern(self):
        try:
            matrix = self.extract_matrix()
            matrix = np.where(matrix > 1, 1, matrix)
            self.click_counts = [[0] * self.columns for _ in range(self.rows)]

            self.hopfield.train(matrix)

            self.status_label.config(text="Status: Pattern saved")
        except:
            self.status_label.config(text="Status: Jsi v piči save")

    def restore_pattern(self):
        try:
            matrix = self.extract_matrix()
            matrix = np.where(matrix > 1, 1, matrix)
            self.click_counts = [[0] * self.columns for _ in range(self.rows)]

            new_matrix = self.hopfield.get_result(matrix)
            self.matrix = new_matrix

            self.display_matrix(self.matrix)

            self.status_label.config(text="Status: Pattern restored")
        except:
            self.status_label.config(text="Status: Jsi v piči restore")

    def extract_matrix(self):
        return np.array(self.matrix)

    def train_mouse(self):
        tmp = self.extract_matrix()
        self.mouse.train(tmp)

        self.status_label.config(text="Status: Trained!")

    def release_mouse(self):
        tmp = self.extract_matrix()
        res = self.mouse.find_path(tmp)

        print(res)

        self.status_label.config(text="Status: {}".format(res))
