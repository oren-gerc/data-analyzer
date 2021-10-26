import numpy as np
import pandas as pd
from datetime import datetime
import plotly.io as pio
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import scipy.optimize as opt

pio.templates["custom"] = go.layout.Template(layout=go.Layout(margin=dict(l=20, r=20, t=40, b=0)))
pio.templates.default = "simple_white+custom"


def plot(x, y, x_label, y_label, graph_title):
    """
    plots a graph with given vectors and names
    :param name:
    :param x: v1
    :param y: v2
    :param x_label:
    :param y_label:
    :param graph_title:
    :return: None
    """
    fig = go.Figure(
        [go.Scatter(x=x, y=y, name="Graph", showlegend=True,
                    marker=dict(color="black", opacity=.7),
                    line=dict(color="black", dash="dash", width=1))],
        layout=go.Layout(title=fr"{graph_title}",
                         xaxis={"title": x_label},
                         yaxis={"title": y_label},
                         height=400))
    fig.show()


def plot_with_fit(x, y, y_fit, x_label, y_label, graph_title):
    """
    plots a graph with given vectors and names
    :param y_fit:
    :param y_error:
    :param y:
    :param x_error:
    :param x: v1
    :param y: v2
    :param x_label:
    :param y_label:
    :param graph_title:
    :return: None
    """
    fig = go.Figure(
        [go.Scatter(x=x, y=y,
                    name="Experiment", showlegend=True,
                    marker=dict(color="black", opacity=.7),
                    line=dict(color="black", dash="dash", width=1)),
         go.Scatter(x=x, y=y_fit, name="Fit", showlegend=True,
                    marker=dict(color="red", opacity=.7),
                    line=dict(color="red", dash="dash", width=1))],
        layout=go.Layout(title=fr"{graph_title}",
                         xaxis={"title": x_label},
                         yaxis={"title": y_label},
                         height=400))
    fig.show()


def plot_with_fit_and_errors(x, x_error, y, y_error, y_fit, x_label, y_label, graph_title):
    """
    plots a graph with given vectors and names
    :param y_fit:
    :param y_error:
    :param y:
    :param x_error:
    :param x: v1
    :param y: v2
    :param x_label:
    :param y_label:
    :param graph_title:
    :return: None
    """
    fig = go.Figure(
        [go.Scatter(x=x, y=y,
                    error_x=dict(type='data', array=x_error, visible=True),
                    error_y=dict(type='data', array=y_error, visible=True),
                    name="Experiment", showlegend=True,
                    marker=dict(color="black", opacity=.7),
                    line=dict(color="black", dash="dash", width=1)),
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


