"""
Receives data to present
Holds functions that plot with or without fit and errors
Saves figures to local environment automatically
"""

import plotly.io as pio
import plotly.graph_objects as go
import os
from CurveFit import CurveFit
import numpy as np

pio.templates["custom"] = go.layout.Template(layout=go.Layout(margin=dict(l=20, r=20, t=40, b=0)))
pio.templates.default = "simple_white+custom"


class Graph:
    def __init__(self, x, y):
        self._x = x
        self._y = y
        self._x_err = 0
        self._y_err = 0
        self._x_label = ""
        self._y_label = ""
        self._graph_title = ""

    def set_labels(self, graph_title, x_label, y_label):
        self._graph_title = graph_title
        self._x_label = x_label
        self._y_label = y_label

    def set_errors(self, x_err, y_err):
        self._x_err = x_err
        self._y_err = y_err

    def save_fig(self, fig):
        if not os.path.exists("graphs"):
            os.mkdir("graphs")
        fig.write_image("graphs/{}.png".format(self._graph_title))

    def simple_plot(self):
        fig = go.Figure(
            [go.Scatter(x=self._x, y=self._y, name="Graph", showlegend=True,
                        marker=dict(color="black", opacity=.7),
                        line=dict(color="black", width=1))],
            layout=go.Layout(title=fr"{self._graph_title}",
                             xaxis={"title": self._x_label},
                             yaxis={"title": self._y_label},
                             height=400))
        fig.show()
        self.save_fig(fig)

    def plot_with_fit(self, fit_function):
        x_fit, y_fit = CurveFit(self._x, self._y, fit_function).get_fit()

        fig = go.Figure(
            [go.Scatter(x=self._x, y=self._y,
                        name="Experiment", showlegend=True,
                        marker=dict(color="black", opacity=.7),
                        line=dict(color="black", width=1)),
             go.Scatter(x=x_fit, y=y_fit, name="Fit", showlegend=True,
                        marker=dict(color="red", opacity=.7),
                        line=dict(color="red", dash="dash", width=1))],
            layout=go.Layout(title=fr"{self._graph_title}",
                             xaxis={"title": self._x_label},
                             yaxis={"title": self._y_label},
                             height=400))
        fig.show()
        self.save_fig(fig)

    def plot_with_fit_and_errors(self, fit_function):
        x_fit, y_fit = CurveFit(self._x, self._y, fit_function).get_fit()

        fig = go.Figure(
            [go.Scatter(x=self._x, y=self._y,
                        error_x=dict(type='data', array=self._x_err, thickness=0.06, visible=True),
                        error_y=dict(type='data', array=self._y_err, thickness=0.06, visible=True),
                        name="Experiment", showlegend=True,
                        marker=dict(color="black", opacity=.7),
                        line=dict(color="black", width=1)),
             go.Scatter(x=x_fit, y=y_fit, name="Fit", showlegend=True,
                        marker=dict(color="red", opacity=.7),
                        line=dict(color="red", dash="dash", width=1))],
            layout=go.Layout(title=fr"{self._graph_title}",
                             xaxis={"title": self._x_label},
                             yaxis={"title": self._y_label},
                             height=400))
        fig.show()
        self.save_fig(fig)
