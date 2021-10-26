"""
Holds general functions
"""

import plotly.io as pio
import plotly.graph_objects as go
import scipy.optimize as opt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

pio.templates["custom"] = go.layout.Template(layout=go.Layout(margin=dict(l=20, r=20, t=40, b=0)))
pio.templates.default = "simple_white+custom"


def read_data(csv_path):
    data = pd.read_excel(csv_path)
    pixel_meter_ratio = 1. / 22.86
    x, y = (data['x'].to_numpy()) * pixel_meter_ratio, (data['y'].to_numpy()) * pixel_meter_ratio
    x_error, y_error = data['x error'].to_numpy(), data['y error'].to_numpy()
    # b_path = np.concatenate([x.reshape((x.shape[0], 1)), y.reshape((y.shape[0], 1))], axis=1)
    # column_errors = np.concatenate([x_error.reshape((x.shape[0], 1)), y_error.reshape((y.shape[0], 1))], axis=1)
    return x, y, x_error, y_error


def plot(x, y, x_label, y_label, graph_title):
    fig = go.Figure(
        [go.Scatter(x=x, y=y, name="Graph", showlegend=True,
                    marker=dict(color="black", opacity=.7),
                    line=dict(color="black", width=1))],
        layout=go.Layout(title=fr"{graph_title}",
                         xaxis={"title": x_label},
                         yaxis={"title": y_label},
                         height=400))
    fig.show()


def plot_motion(path):
    start = path[:1]
    stop = path[-1:]
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(path[:, 0], path[:, 1], c="blue", alpha=0.25, s=0.05)
    ax.plot(path[:, 0], path[:, 1], c="blue", alpha=0.5, lw=0.25)
    ax.plot(start[:, 0], start[:, 1], c='red', marker='+')
    ax.plot(stop[:, 0], stop[:, 1], c='black', marker='o')
    plt.title("Movement in 2D: X vs Y")
    plt.tight_layout(pad=0)
    plt.show()


def plot_with_fit(x, y, y_fit, fit_params, x_label, y_label, graph_title):
    fig = go.Figure(
        [go.Scatter(x=x, y=y,
                    name="Experiment", showlegend=True,
                    marker=dict(color="black", opacity=.7),
                    line=dict(color="black", width=1)),
         go.Scatter(x=x, y=y_fit, name="Fit", showlegend=True,
                    marker=dict(color="red", opacity=.7),
                    line=dict(color="red", dash="dash", width=1))],
        layout=go.Layout(title=fr"{graph_title}",
                         xaxis={"title": x_label},
                         yaxis={"title": y_label},
                         height=400))
    fig.show()
    if not os.path.exists("graphs"):
        os.mkdir("graphs")
    fig.write_image("graphs/{}.png".format(graph_title))
    print(graph_title, "fit parameters: ", fit_params)


def plot_with_fit_and_errors(x, x_error, y, y_error, y_fit, fit_params, x_label, y_label, graph_title):
    fig = go.Figure(
        [go.Scatter(x=x, y=y,
                    error_x=dict(type='data', array=x_error, thickness=0.06, visible=True),
                    error_y=dict(type='data', array=y_error, thickness=0.06, visible=True),
                    name="Experiment", showlegend=True,
                    marker=dict(color="black", opacity=.7),
                    line=dict(color="black", width=1)),
         go.Scatter(x=x, y=y_fit, name="Fit", showlegend=True,
                    marker=dict(color="red", opacity=.7),
                    line=dict(color="red", dash="dash", width=1))],
        layout=go.Layout(title=fr"{graph_title}",
                         xaxis={"title": x_label},
                         yaxis={"title": y_label},
                         height=400))
    fig.show()
    if not os.path.exists("graphs"):
        os.mkdir("graphs")
    fig.write_image("graphs/{}.png".format(graph_title))


def curve_fit(func, x, y):
    # returns the parameters and the y values for the fit curve
    parameters, params_cov = opt.curve_fit(func, x, y)
    return parameters, func(x, *parameters)


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


def plot_curve_with_fit_and_errors(test, x, error_x, y, error_y, num):
    params, y_fit = curve_fit(test, x, y)
    plot_with_fit_and_errors(x, error_x, y, error_y, y_fit, params, "Time [s]", "<r^2> [m^2]",
                             "Average Squared Distance vs. Time, particle #{}, with errors".format(num))


def plot_curve_with_fit(test, x, y, num):
    params, y_fit = curve_fit(test, x, y)
    plot_with_fit(x, y, y_fit, params, "Time [s]", "<r^2> [m^2]",
                  "Average Squared Distance vs. Time, particle #{}".format(num))


def normalize_values(vec):
    # normalizes the x and y values to start with zero
    return vec - vec[0]
