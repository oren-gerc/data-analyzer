"""
Performs data analysis - brownian motion experiment
"""

import numpy as np
from scipy.stats import linregress
import equations
import matplotlib.pyplot as plt
from Graph import Graph


def normalize(vec):
    # normalizes the x and y values to start with zero
    return vec - vec[0]


def calculate_r_squared(x, y):
    r_squared = np.square(normalize(x)) + np.square(normalize(y))
    return np.cumsum(r_squared) / np.arange(1, len(r_squared) + 1)


def min_length(arr_2d):
    min_len = 18000000
    for i in range(len(arr_2d)):
        if int(len(arr_2d[i])) < min_len:
            min_len = len(arr_2d[i])
    return min_len


def cut_array_equally(arr, num_chunks):
    partitions = np.array_split(arr, num_chunks)
    minimal_length = min_length(partitions)
    for i, particle in enumerate(partitions):
        partitions[i] = particle[:minimal_length]
    return partitions


def drop_drift(x):
    time = np.linspace(0, x.shape[0], num=x.shape[0])
    slope_x, intercept, r, p, se = linregress(time, x)
    after = x - slope_x * time
    return after


def calc_r_squared_error(x, y, x_error, y_error):
    return np.sqrt(np.square(2 * x * x_error) + np.square(2 * y * y_error))


def calc_average_r_squared(x, y, num_fictive_particles):
    x_partitions = cut_array_equally(x, num_fictive_particles)
    y_partitions = cut_array_equally(y, num_fictive_particles)
    r_squared_vectors = []
    for i in range(num_fictive_particles):
        r_squared_vectors.append(calculate_r_squared(x_partitions[i], y_partitions[i]))
    return np.average(r_squared_vectors, axis=0)


class ArtificialBrownianMotion:
    def __init__(self, T, dt, mu_x, mu_y, sigma):
        dims = 2
        N = round(T / dt) + 1
        t = np.linspace(0, T, N)
        W = np.random.standard_normal(size=(N - 1, dims))  # create random (x,y) steps
        origin = np.zeros((1, dims))
        steps = np.concatenate([origin, W]).cumsum(0) / np.sqrt(dt)
        self._x = ((mu_x - (sigma ** 2 / 2)) * t).reshape((N, 1))
        self._y = ((mu_y - (sigma ** 2 / 2)) * t).reshape((N, 1))
        drift = np.concatenate([self._x, self._y], axis=1)
        self._path = sigma * steps + drift

    def set_x_y(self, x, y):
        self._x = x
        self._y = y

    def plot_trajectory(self):
        start = self._path[:1]
        stop = self._path[-1:]
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.scatter(self._path[:, 0], self._path[:, 1], c="blue", alpha=0.25, s=0.05)
        ax.plot(self._path[:, 0], self._path[:, 1], c="blue", alpha=0.5, lw=0.25)
        ax.plot(start[:, 0], start[:, 1], c='red', marker='+')
        ax.plot(stop[:, 0], stop[:, 1], c='black', marker='o')
        plt.title("Movement in 2D: X vs Y")
        plt.tight_layout(pad=0)
        plt.show()

    def drop_drift(self):
        self._x = drop_drift(self._x)
        self._y = drop_drift(self._y)

    def plot_r_squared_vs_time(self, num_particles):
        average_r_2 = calc_average_r_squared(self._x, self._y, num_particles)
        time = np.linspace(0, len(average_r_2), num=len(average_r_2))
        g = Graph(time, average_r_2)
        g.plot_with_fit(equations.linear_no_intercept)


if __name__ == '__main__':
    b = ArtificialBrownianMotion(1000, 0.1, 0, 0, 0.5)
    b.plot_trajectory()
    b.plot_r_squared_vs_time(50)
