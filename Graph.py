"""
Receives data to present
Holds functions that plot with or without fit and errors
Saves figures to local environment automatically
"""

import plotly.io as pio
import plotly.graph_objects as go
import os
import curve_fit
import numpy as np

pio.templates["custom"] = go.layout.Template(layout=go.Layout(margin=dict(l=20, r=20, t=40, b=0)))
pio.templates.default = "simple_white+custom"


class Graph:
    def __init__(self, x, y, fit_func, x_err, y_err):
        self._x = x
        self._y = y
        self._fit_func = fit_func
        self._x_err = x_err
        self._y_err = y_err

        # calculate fit
        self._fit_parameters = curve_fit.fit(self._fit_func, self._x, self._y)
        self._x_fit = np.linspace(np.amin(self._x), np.amax(self._x), num=1000)
        self._y_fit = self._fit_func(self._x_fit, *self._fit_parameters)

        # init labels
        self._x_label = ""
        self._y_label = ""
        self._graph_title = ""

    def set_labels(self, graph_title, x_label, y_label):
        self._graph_title = graph_title
        self._x_label = x_label
        self._y_label = y_label

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

    def plot_with_fit(self):
        fig = go.Figure(
            [go.Scatter(x=self._x, y=self._y,
                        name="Experiment", showlegend=True,
                        marker=dict(color="black", opacity=.7),
                        line=dict(color="black", width=1)),
             go.Scatter(x=self._x, y=self._y_fit, name="Fit", showlegend=True,
                        marker=dict(color="red", opacity=.7),
                        line=dict(color="red", dash="dash", width=1))],
            layout=go.Layout(title=fr"{self._graph_title}",
                             xaxis={"title": self._x_label},
                             yaxis={"title": self._y_label},
                             height=400))
        fig.show()
        self.save_fig(fig)

    def plot_with_fit_and_errors(self):
        fig = go.Figure(
            [go.Scatter(x=self._x, y=self._y,
                        error_x=dict(type='data', array=self._x_err, thickness=0.06, visible=True),
                        error_y=dict(type='data', array=self._y_err, thickness=0.06, visible=True),
                        name="Experiment", showlegend=True,
                        marker=dict(color="black", opacity=.7),
                        line=dict(color="black", width=1)),
             go.Scatter(x=self._x_fit, y=self._y_fit, name="Fit", showlegend=True,
                        marker=dict(color="red", opacity=.7),
                        line=dict(color="red", dash="dash", width=1))],
            layout=go.Layout(title=fr"{self._graph_title}",
                             xaxis={"title": self._x_label},
                             yaxis={"title": self._y_label},
                             height=400))
        fig.show()
        self.save_fig(fig)
