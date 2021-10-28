"""
Performs data analysis - brownian motion experiment
"""
import utils
import equations
import numpy as np
from scipy.stats import linregress
import pandas as pd


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


def drop_drift(x, y, time):
    slope_x, intercept, r, p, se = linregress(time, x)
    slope_y, intercept, r, p, se = linregress(time, y)
    utils.plot(time, x - slope_x * time, "", "", "x vs time")
    utils.plot(time, y - slope_y * time, "", "", "y vs time")
    return (x - slope_x * time), (y - slope_y * time)
    # return x, y


def calc_average_r_squared(x, y):
    r_squared = np.square(x) + np.square(y)
    averaged = np.cumsum(r_squared) / np.arange(1, len(r_squared) + 1)
    return averaged


def calc_average_r_squared2(x, y, partitions):
    x_partitions = np.array_split(x, partitions)
    y_partitions = np.array_split(y, partitions)
    assert len(x) == len(y)
    r_2 = []
    for i in range(partitions):
        r_2.append(np.mean(np.square(x_partitions[i]) + np.square(y_partitions[i])))
    return r_2


def calc_r_squared_error(x, y, x_error, y_error):
    return np.sqrt(np.square(2 * x * x_error) + np.square(2 * y * y_error))


def analyze_week1():
    # for each particle:
    # read its data, normalize it against drift, plot its r^2 vs time, fit and discover D, plot D vs R of particle
    goodies = [5]
    baddies = [1, 2, 3, 4]
    for particle in range(1, 6):
        excel_path = r'C:\Users\user\Desktop\lab\physics-data-analyzer\experiment_data\particle{}.xlsx'.format(
            particle)
        frames_per_second = 30

        # read and normalize data
        x, y, x_error, y_error = utils.read_data(excel_path)
        x, y = utils.normalize_values(x), utils.normalize_values(y)

        # plot trajectory of the particle
        utils.plot(x, y, "x", "y", "2D trajectory of the particle #{}".format(particle))
        time = np.linspace(0, x.shape[0], num=x.shape[0]) / frames_per_second

        # # numerically check for drift in both axes, than normalize values
        # x, y = drop_drift(x, y, time)
        #
        # # plot trajectory of the particle without drift (simulated)
        # utils.plot(x, y, "x", "y",
        #            "2D trajectory of the particle #{}, without drift".format(particle))

        # plot r^2 vs time, with linear/parabolic fit, depending on drift
        if particle in goodies:
            equation = equations.linear_no_intercept
        else:
            equation = equations.parabolic_no_intercept

        average_r_2 = calc_average_r_squared(x, y)
        error_r = calc_r_squared_error(x, y, x_error, y_error)
        utils.plot_curve_with_fit(equation, time, average_r_2, particle)
        error_t = np.zeros(shape=average_r_2.shape)
        utils.plot_curve_with_fit_and_errors(equation, time, error_t, average_r_2, error_r, particle)

        # average_r_2 = calc_average_r_squared2(x, y, 200)
        # time = np.linspace(0, len(average_r_2), num=len(average_r_2)) / 30
        # utils.plot_curve_with_fit(equations.linear, time, average_r_2, particle)


def analyze_week1_again():
    # use 30% concentrations to plot r^2 vs time
    # than D vs radius
    pass


def analyze_week2():
    # read one excel into x's and y's
    # decide which are drifted which aren't and marks them in order to fit them
    # fit all paths, save D param from fit
    # plot D against viscosity
    results = {0: 0, 5: 0, 10: 0, 30: 0, 40: 0, 50: 0, 60: 0, 85: 0, 95: 0}
    frames_per_second = 4.4
    excel_path = r'C:\Users\user\Desktop\lab\physics-data-analyzer\experiment_data\week2.xlsx'

    # read and normalize data
    data = pd.read_excel(excel_path)
    pixel_meter_ratio = 1. / 22.86

    for concentration in results.keys():
        # read data, normalize it
        x_col_name = 'x{}'.format(concentration)
        y_col_name = 'y{}'.format(concentration)
        x, y = (data[x_col_name].to_numpy()) * pixel_meter_ratio, (data[y_col_name].to_numpy()) * pixel_meter_ratio
        x, y = utils.normalize_values(x), utils.normalize_values(y)
        time = np.linspace(0, x.shape[0], num=x.shape[0]) / frames_per_second
        # plot r^2 vs time, with linear/parabolic fit, depending on drift
        average_r_2 = calc_average_r_squared(x, y)
        results[concentration] = utils.curve_fit(equations.parabolic_no_intercept, time, average_r_2)[1]
    utils.plot_with_fit(results.keys(), results.values(),
                        *(utils.curve_fit(equations.linear, results.keys(), results.values())),
                        "D", "R", "D vs. R")


if __name__ == '__main__':
    # analyze_week1()
    # analyze_week1_again()
    analyze_week2()

    # path = create_brownian_motion(T=10000, dt=0.1, mu_x=-0.1, mu_y=-0.1, sigma=0.1, dims=2)
    # utils.plot_motion(path)
    # x, y = path[:, 0], path[:, 1]
    #
    # average_r_2 = calc_average_r_squared2(x, y, 100)
    # time = np.linspace(0, len(average_r_2), num=len(average_r_2)) / 30
    # utils.plot_curve_with_fit(equations.parabolic_no_intercept, time, average_r_2, 7)
