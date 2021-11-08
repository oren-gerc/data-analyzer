"""
Runs analysis according to specific demands
"""

import DataHandler
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


def analyze_week3():
    path = "C:\\Users\\ORENGER\\Desktop\\uni\\physics-data-analyzer\\experiment_data\\week3\\measurements.xlsx"
    data_handler = DataHandler.DataHandler(path)
    radii = [17, 20, 12, 13, 14, 15, 17, 12]
    temperatures = [1, 2, 3, 4, 5, 6, 7, 8]
    for temperature in temperatures:
        # get data and normalize it
        x, y = data_handler.get_columns(['x{}'.format(temperature), 'y{}'.format(temperature)])
        x, y = normalize_values(x), normalize_values(y)

        # calculate average r^2
        average_r_squared = calc_average_r_squared(x, y, 20)

        # save r^2 back to excel
        data_handler.add_column_to_excel('r{}'.format(temperature), average_r_squared)

        # fit, save params


if __name__ == '__main__':
    analyze_week3()
