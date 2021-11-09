"""
Runs analysis according to specific demands
"""

from DataHandler import DataHandler
from Graph import Graph
from CurveFit import CurveFit
import equations
import numpy as np


def normalize_values(vec):
    # normalizes the x and y values to start with zero
    return vec - vec[0]


def calc_average_r_squared(x, y, partitions):
    # return np.cumsum(self._r_squared) / np.arange(1, len(self._r_squared) - 1)
    x_partitions = np.array_split(x, partitions)
    y_partitions = np.array_split(y, partitions)
    assert len(x) == len(y)
    r_2 = np.zeros(int(len(x) / partitions))
    for j in range(partitions):
        for i in range(int(len(x) / partitions)):
            r_2[i] += (np.square(x_partitions[j][i] - x_partitions[j][0]) + np.square(
                y_partitions[j][i] - y_partitions[j][0]))
    r_2 = r_2 / partitions
    return r_2


def analyze_effect_of_temperature():
    path = "C:\\Users\\user\\Desktop\\lab\\data-analyzer\\experiment_data\\week3\\measurements.xlsx"
    data_handler = DataHandler(path)
    radii = [17, 20, 12, 13, 14, 15, 17, 12]
    temperatures = [17.3, 21.4, 25.5, 29.8, 34.8, 42.4, 45]
    values = []
    for temperature in temperatures:
        # get data and normalize it
        x, y = data_handler.get_columns(['x{}'.format(temperature), 'y{}'.format(temperature)])
        x, y = normalize_values(x), normalize_values(y)

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
