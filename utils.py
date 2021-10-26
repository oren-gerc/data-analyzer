"""
Holds general functions
"""

import plotly.io as pio
import plotly.graph_objects as go
import scipy.optimize as opt
import matplotlib.pyplot as plt

pio.templates["custom"] = go.layout.Template(layout=go.Layout(margin=dict(l=20, r=20, t=40, b=0)))
pio.templates.default = "simple_white+custom"


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


def plot_with_fit(x, y, y_fit, x_label, y_label, graph_title):
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


def plot_with_fit_and_errors(x, x_error, y, y_error, y_fit, x_label, y_label, graph_title):
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


def curve_fit(func, x, y):
    # returns the parameters and the y values for the fit curve
    parameters, params_cov = opt.curve_fit(func, x, y)
    return parameters, func(x, *parameters)
