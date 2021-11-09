"""
Runs analysis according to specific demands
"""

from DataHandler import DataHandler
from Graph import Graph
from CurveFit import CurveFit
import equations
import numpy as np


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


def calc_average_r_squared(x, y, num_fictive_particles):
    x_partitions = cut_array_equally(x, num_fictive_particles)
    y_partitions = cut_array_equally(y, num_fictive_particles)
    r_squared_vectors = [calculate_r_squared(x_partitions[i], y_partitions[i])
                         for i in range(num_fictive_particles)]
    return np.average(r_squared_vectors, axis=0)


def analyze_effect_of_temperature():
    path = "C:\\Users\\user\\Desktop\\lab\\data-analyzer\\experiment_data\\week3\\measurements.xlsx"
    data_handler = DataHandler(path)
    radii = [17, 20, 12, 13, 14, 15, 17, 12]
    temperatures = [17.3, 21.4, 25.5, 29.8, 34.8, 42.4, 45]
    values = []
    for temperature in temperatures:
        # get data and normalize it
        x, y = data_handler.get_columns(['x{}'.format(temperature), 'y{}'.format(temperature)])
        x, y = normalize(x), normalize(y)

        # calculate average r^2
        average_r_squared = calc_average_r_squared(x, y, 20)

        # save r^2 back to excel
        data_handler.add_column_to_excel('r{}'.format(temperature), average_r_squared)

        # get fit params
        D = CurveFit(x, y, equations.linear).get_fit_params()
        values.append(D[0])

    print(values)
    graph = Graph(np.array(temperatures), np.array(values))
    # calculate errors for values and temperatures
    # set lables for axes


if __name__ == '__main__':
    analyze_effect_of_temperature()
