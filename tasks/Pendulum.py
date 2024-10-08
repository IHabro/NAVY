import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


class Pendulum:
    def __init__(self):
        self.l1 = 3
        self.l2 = 2
        self.m1 = 10
        self.m2 = 3
        self.g = 9.81
        self.tmax = 30
        self.dt = 0.01

    # Vzano z netu, vicemene se jedna pouye o implementaci Slidu 6, tedy "Expresions for acceleration"
    def get_derivative(self, y, t, L1, L2, m1, m2):
        theta1, z1, theta2, z2 = y

        c, s = np.cos(theta1 - theta2), np.sin(theta1 - theta2)

        theta1dot = z1
        z1dot = (m2 * self.g * np.sin(theta2) * c - m2 * s * (L1 * z1 ** 2 * c + L2 * z2 ** 2) -
                 (m1 + m2) * self.g * np.sin(theta1)) / L1 / (m1 + m2 * s ** 2)
        theta2dot = z2
        z2dot = ((m1 + m2) * (L1 * z1 ** 2 * s - self.g * np.sin(theta2) + self.g * np.sin(theta1) * c) +
                 m2 * L2 * z2 ** 2 * s * c) / L2 / (m1 + m2 * s ** 2)

        return theta1dot, z1dot, theta2dot, z2dot

    def start(self):
        t = np.arange(0, self.tmax+self.dt, self.dt)

        #                               theta1, vel1,    theta2, vel2
        thetasAndVelicities = np.array([2*np.pi/6, 0, 5*np.pi/8, 0])

        y = odeint(self.get_derivative, thetasAndVelicities, t, args=(self.l1, self.l2, self.m1, self.m2))

        theta1, theta2 = y[:, 0], y[:, 2]

        # Convert to Cartesian coordinates of the two bob positions.
        x1 = self.l1 * np.sin(theta1)
        y1 = -self.l1 * np.cos(theta1)
        x2 = x1 + self.l2 * np.sin(theta2)
        y2 = y1 - self.l2 * np.cos(theta2)

        # Plotted bob circle radius
        r = 0.05
        # Plot a trail of the m2 bob's position for the last trail_secs seconds.
        trail_secs = 1
        # This corresponds to max_trail time points.
        max_trail = int(trail_secs / self.dt)

        def make_plot(i):
            # Plot and save an image of the double pendulum configuration for time
            # point i.
            # The pendulum rods.
            ax.plot([0, x1[i], x2[i]], [0, y1[i], y2[i]], lw=2, c='k')
            # Circles representing the anchor point of rod 1, and bobs 1 and 2.
            c0 = Circle((0, 0), r / 2, fc='k', zorder=10)
            c1 = Circle((x1[i], y1[i]), r, fc='b', ec='b', zorder=10)
            c2 = Circle((x2[i], y2[i]), r, fc='r', ec='r', zorder=10)
            ax.add_patch(c0)
            ax.add_patch(c1)
            ax.add_patch(c2)

            # The trail will be divided into ns segments and plotted as a fading line.
            ns = 20
            s = max_trail // ns

            for j in range(ns):
                imin = i - (ns - j) * s
                if imin < 0:
                    continue
                imax = imin + s + 1
                # The fading looks better if we square the fractional length along the
                # trail.
                alpha = (j / ns) ** 2
                ax.plot(x2[imin:imax], y2[imin:imax], c='r', solid_capstyle='butt',
                        lw=2, alpha=alpha)

            # Centre the image on the fixed anchor point, and ensure the axes are equal
            ax.set_xlim(-self.l1 - self.l2 - r, self.l1 + self.l2 + r)
            ax.set_ylim(-self.l1 - self.l2 - r, self.l1 + self.l2 + r)
            ax.set_aspect('equal', adjustable='box')
            plt.axis('off')
            plt.savefig('frames/_img{:04d}.png'.format(i // di), dpi=72)
            plt.cla()

        # Make an image every di time points, corresponding to a frame rate of fps
        # frames per second.
        # Frame rate, s-1
        fps = 10
        di = int(1 / fps / self.dt)
        fig = plt.figure(figsize=(8.3333, 6.25), dpi=72)
        ax = fig.add_subplot(111)

        for i in range(0, t.size, di):
            print(i // di, '/', t.size // di)
            make_plot(i)
