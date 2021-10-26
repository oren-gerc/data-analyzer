"""
Performs data analysis - brownian motion experiment
"""

import utils
import equations
import numpy as np


def analyze_week1():
    # for each particle:
    # read its data, normalize it against drift, plot its r^2 vs time, fit and discover D, plot D vs R of particle
    particles = [5, 4, 3, 2, 1]
    for particle in particles:
        excel_path = r'C:/Users/user/Desktop/lab/Brownian_Motion/particle{}.xlsx'.format(particle)

        # read and normalize data
        trajectory, errors = utils.read_data(excel_path)
        trajectory = utils.normalize_values(trajectory)

        # plot trajectory of the particle
        utils.plot(trajectory[:, 0], trajectory[:, 1], "x", "y",
                   "2D trajectory of the particle #{}".format(particle))

        # calculate r^2, plot against time
        time = np.linspace(0, trajectory.shape[0], num=trajectory.shape[0])
        r_2 = utils.calc_average_r_squared(trajectory)
        error_t = np.zeros(shape=r_2.shape)
        error_r = utils.calc_r_squared_error(trajectory[:, 0], trajectory[:, 1], errors[:, 0], errors[:, 1])
        utils.plot_curve_with_fit(equations.linear, time, r_2)
        utils.plot_curve_with_fit_and_errors(equations.linear, time, error_t, r_2, error_r)


if __name__ == '__main__':
    analyze_week1()

    # path = create_brownian_motion(T=10000, dt=0.1, mu_x=0, mu_y=0, sigma=0.1, dims=2)
    # plot_brownian_motion(path)
    # plot_average_r_squared_vs_time(path)
