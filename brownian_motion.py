"""
Performs data analysis - brownian motion experiment
"""
import utils
import numpy as np
from scipy.stats import linregress
import pandas as pd
import equations


def normalize(vec):
    # normalizes the x and y values to start with zero
    return vec - vec[0]


def create_brownian_motion(T, dt, mu_x, mu_y, sigma, dims=2):
    N = round(T / dt) + 1
    t = np.linspace(0, T, N)
    W = np.random.standard_normal(size=(N - 1, dims))  # create random (x,y) steps
    origin = np.zeros((1, dims))
    steps = np.concatenate([origin, W]).cumsum(0) / np.sqrt(dt)
    drift_x = ((mu_x - (sigma ** 2 / 2)) * t).reshape((N, 1))
    drift_y = ((mu_y - (sigma ** 2 / 2)) * t).reshape((N, 1))
    drift = np.concatenate([drift_x, drift_y], axis=1)
    drifted_path = sigma * steps + drift
    return drifted_path


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
    return (x - slope_x * time)


def calc_average_r_squared(x, y, num_fictive_particles):
    x_partitions = cut_array_equally(x, num_fictive_particles)
    y_partitions = cut_array_equally(y, num_fictive_particles)
    r_squared_vectors = []
    for i in range(num_fictive_particles):
        a = drop_drift(x_partitions[i])
        b = drop_drift(y_partitions[i])
        r_squared_vectors.append(calculate_r_squared(a, b))
    return np.average(r_squared_vectors, axis=0)


def calc_r_squared_error(x, y, x_error, y_error):
    return np.sqrt(np.square(2 * x * x_error) + np.square(2 * y * y_error))


def analyze_week1():
    # for each particle:
    # read its data, normalize it against drift, plot its r^2 vs time, fit and discover D, plot D vs R of particle
    drifted = [1, 2, 3]
    for particle in range(1, 6):
        excel_path = r'C:\Users\user\Desktop\lab\data-analyzer\experiment_data\week1\particle{}.xlsx'.format(
            particle)
        frames_per_second = 30

        # read and normalize data
        x, y, x_error, y_error = utils.read_data(excel_path)
        x, y = utils.normalize_values(x), utils.normalize_values(y)

        # plot trajectory of the particle
        utils.plot(x, y, "x", "y", "2D trajectory of the particle #{}".format(particle))

        # # plot trajectory of the particle without drift (simulated)
        # utils.plot(x, y, "x", "y",
        #            "2D trajectory of the particle #{}, without drift".format(particle))

        # plot r^2 vs time, with linear/parabolic fit, depending on drift
        # if particle in goodies:
        equation = equations.linear_no_intercept
        # else:
        #     equation = equations.parabolic_no_intercept

        # numerically check for drift in both axes, than normalize values
        average_r_2 = calc_average_r_squared(x, y, 25)
        error_r = calc_r_squared_error(x, y, x_error, y_error)
        time = np.linspace(0, average_r_2.shape[0], num=average_r_2.shape[0]) / frames_per_second
        utils.plot_curve_with_fit(equation, time, average_r_2, particle)
        error_t = np.zeros(shape=average_r_2.shape)
        utils.plot_curve_with_fit_and_errors(equation, time, error_t, average_r_2, error_r, particle)


def analyze_week2():
    # read one excel into x's and y's
    # decide which are drifted which aren't and marks them in order to fit them
    # fit all paths, save D param from fit
    # plot D against viscosity
    rikuzim = np.array([5, 30, 40, 50, 60, 85, 95])
    coeffs = []
    frames_per_second = 4.4
    excel_path = r'C:\Users\user\Desktop\lab\physics-data-analyzer\experiment_data\week2.xlsx'

    # read and normalize data
    data = pd.read_excel(excel_path)
    pixel_meter_ratio = 1. / 22.86

    for concentration in rikuzim:
        # read data, normalize it
        x_col_name = 'x{}'.format(concentration)
        y_col_name = 'y{}'.format(concentration)
        x, y = (data[x_col_name].to_numpy()) * pixel_meter_ratio, (data[y_col_name].to_numpy()) * pixel_meter_ratio
        x, y = utils.normalize_values(x), utils.normalize_values(y)
        x, y = x[~np.isnan(x)], y[~np.isnan(y)]

        # plot movement of particle
        # utils.plot(x, y, "x", "y", "2D trajectory of the particle #{}".format(concentration))

        average_r_2 = calc_average_r_squared(x, y, 20)
        time = np.linspace(0, average_r_2.shape[0], num=average_r_2.shape[0]) / frames_per_second

        # utils.plot_curve_with_fit(equations.linear_no_intercept, time, average_r_2, concentration)
        coeffs.append(utils.curve_fit(equations.linear_no_intercept, time, average_r_2)[0][0])

    # add errors, and make the range better so it will appear smoother
    utils.plot_curve_with_fit(equations.one_over_x, rikuzim, np.array(coeffs), 1)


def plot_brownian_motion():
    path = create_brownian_motion(T=10000, dt=0.1, mu_x=-0.1, mu_y=-0.1, sigma=0.1, dims=2)
    utils.plot_motion(path)
    x, y = path[:, 0], path[:, 1]

    average_r_2 = calc_average_r_squared(x, y, 100)
    time = np.linspace(0, len(average_r_2), num=len(average_r_2)) / 30
    utils.plot_curve_with_fit(equations.parabolic_no_intercept, time, average_r_2, 7)


if __name__ == '__main__':
    analyze_week1()
    # analyze_week2()
