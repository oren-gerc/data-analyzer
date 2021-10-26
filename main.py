import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from scipy.optimize import curve_fit
import utils


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


def calc_average_r_squared(brownian_path):
    r_squared = np.square(brownian_path[:, 0]) + np.square(brownian_path[:, 1])
    averaged = np.cumsum(r_squared) / np.arange(1, len(r_squared) + 1)
    return averaged


def plot_average_r_squared_vs_time(brownian_path):
    t = np.linspace(0, path.shape[0], num=path.shape[0])
    utils.plot(t, calc_average_r_squared(brownian_path), "time", "<r^2>", "<r^2> vs. time")


def plot_brownian_motion(brownian_path):
    start = brownian_path[:1]
    stop = brownian_path[-1:]
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(brownian_path[:, 0], brownian_path[:, 1], c="blue", alpha=0.25, s=0.05)
    ax.plot(brownian_path[:, 0], brownian_path[:, 1], c="blue", alpha=0.5, lw=0.25)
    ax.plot(start[:, 0], start[:, 1], c='red', marker='+')
    ax.plot(stop[:, 0], stop[:, 1], c='black', marker='o')
    plt.title("Movement in 2D: X vs Y")
    plt.tight_layout(pad=0)
    plt.show()


def build_particle_path_from_excel():
    data = pd.read_excel(r'C:/Users/user/Desktop/lab/Brownian_Motion/data.xlsx')
    pixel_meter_ratio = 1. / 22.86
    x, y = (data['x'].to_numpy()) * pixel_meter_ratio, (data['y'].to_numpy()) * pixel_meter_ratio
    x_error, y_error = data['x error'].to_numpy(), data['y error'].to_numpy()
    b_path = np.concatenate([x.reshape((x.shape[0], 1)), y.reshape((y.shape[0], 1))], axis=1)
    column_errors = np.concatenate([x_error.reshape((x.shape[0], 1)), y_error.reshape((y.shape[0], 1))], axis=1)
    return b_path, column_errors


def linear(t, a, b):
    return b + a * t


def plot_curve_with_fit_and_errors(test, x, error_x, y, error_y):
    params, y_fit = utils.curve_fit(test, x, y)
    utils.plot_with_fit_and_errors(x, error_x, y, error_y, y_fit, "Time [s]", "<r^2> [m^2]",
                                   "Average Squared Distance vs. Time")


def plot_curve_with_fit(test, x, y):
    params, y_fit = utils.curve_fit(test, x, y)
    utils.plot_with_fit(x, y, y_fit, "Time [s]", "<r^2> [m^2]", "Average Squared Distance vs. Time")


def normalize_values(brownian_path):
    # normalizes the x and y values to start with zero
    brownian_path[:, 0] = brownian_path[:, 0] - brownian_path[0, 0]
    brownian_path[:, 1] = brownian_path[:, 1] - brownian_path[0, 1]
    return brownian_path


def calc_r_squared_error(x, y, x_error, y_error):
    return np.sqrt(np.square(2 * x * x_error) + np.square(2 * y * y_error))


if __name__ == '__main__':
    excel_filename = ""

    # path = create_brownian_motion(T=10000, dt=0.1, mu_x=0, mu_y=0, sigma=0.1, dims=2)
    # plot_brownian_motion(path)
    # plot_average_r_squared_vs_time(path)

    path, errors = build_particle_path_from_excel()
    path = normalize_values(path)
    plot_brownian_motion(path)

    time = np.linspace(0, path.shape[0], num=path.shape[0])
    r_2 = calc_average_r_squared(path)
    error_t = np.zeros(shape=r_2.shape)
    error_r = calc_r_squared_error(path[:, 0], path[:, 1], errors[:, 0], errors[:, 1])
    plot_curve_with_fit(linear, time, r_2)
    plot_curve_with_fit_and_errors(linear, time, error_t, r_2, error_r)
