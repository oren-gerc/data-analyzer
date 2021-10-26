"""
Performs data analysis - brownian motion experiment
"""
import matplotlib.pyplot as plt
import utils
import equations
import numpy as np
from scipy.stats import linregress


def drop_drift(x, y, time):
    slope_x, intercept, r, p, se = linregress(time, x)
    slope_y, intercept, r, p, se = linregress(time, y)
    utils.plot(time, x - slope_x * time, "", "", "x vs time")
    utils.plot(time, y - slope_y * time, "", "", "y vs time")
    # return (x - slope_x * time), (y - slope_y * time)
    return x, y


def calc_average_r_squared(r_squared):
    averaged = np.cumsum(r_squared) / np.arange(1, len(r_squared) + 1)
    return averaged


def analyze_week1():
    # for each particle:
    # read its data, normalize it against drift, plot its r^2 vs time, fit and discover D, plot D vs R of particle
    particles = [1, 2, 3, 4, 5]
    for particle in particles:
        excel_path = r'C:\Users\ORENGER\Desktop\uni\physics-data-analyzer\experiment_data\particle{}.xlsx'.format(
            particle)

        # read and normalize data
        trajectory, errors = utils.read_data(excel_path)
        trajectory = utils.normalize_values(trajectory)

        # plot trajectory of the particle
        utils.plot(trajectory[:, 0], trajectory[:, 1], "x", "y",
                   "2D trajectory of the particle #{}".format(particle))
        time = np.linspace(0, trajectory.shape[0], num=trajectory.shape[0])

        # numerically check for drift in both axes, than normalize values
        trajectory[:, 0], trajectory[:, 1] = drop_drift(trajectory[:, 0], trajectory[:, 1], time)

        # plot trajectory of the particle without drift (simulated)
        utils.plot(trajectory[:, 0], trajectory[:, 1], "x", "y",
                   "2D trajectory of the particle #{}, without drift".format(particle))

        # calculate r^2, plot against time
        r_2 = np.square(trajectory[:, 0]) + np.square(trajectory[:, 1])
        average_r_2 = calc_average_r_squared(r_2)
        error_t = np.zeros(shape=average_r_2.shape)
        error_x, error_y = errors[:, 0], errors[:, 1]
        error_r = utils.calc_r_squared_error(trajectory[:, 0], trajectory[:, 1], error_x, error_y)
        utils.plot_curve_with_fit(equations.linear, time, average_r_2, particle)
        utils.plot_curve_with_fit_and_errors(equations.linear, time, error_t, average_r_2, error_r, particle)


if __name__ == '__main__':
    analyze_week1()

    # path = utils.create_brownian_motion(T=10000, dt=0.1, mu_x=0, mu_y=2, sigma=0.1, dims=2)
    # utils.plot_motion(path)
    # utils.plot_average_r_squared_vs_time(path)
