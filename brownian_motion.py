"""
Performs data analysis - brownian motion experiment
"""
import utils
import equations
import numpy as np
from scipy.stats import linregress


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
    # return (x - slope_x * time), (y - slope_y * time)
    return x, y


def calc_average_r_squared(x, y):
    r_squared = np.square(x) + np.square(y)
    averaged = np.cumsum(r_squared) / np.arange(1, len(r_squared) + 1)
    return averaged


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

        # numerically check for drift in both axes, than normalize values
        # trajectory[:, 0], trajectory[:, 1] = drop_drift(trajectory[:, 0], trajectory[:, 1], time)

        # plot trajectory of the particle without drift (simulated)
        # utils.plot(trajectory[:, 0], trajectory[:, 1], "x", "y",
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


if __name__ == '__main__':
    analyze_week1()

    # path = utils.create_brownian_motion(T=10000, dt=0.1, mu_x=0, mu_y=2, sigma=0.1, dims=2)
    # utils.plot_motion(path)
    # utils.plot_average_r_squared_vs_time(path)
