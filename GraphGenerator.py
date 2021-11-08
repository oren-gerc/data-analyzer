"""
Receives data to present: x,y,fit_function,errors
Holds functions that plot with or without fit and errors
"""

import plotly.io as pio
import plotly.graph_objects as go
import os
import curve_fit

pio.templates["custom"] = go.layout.Template(layout=go.Layout(margin=dict(l=20, r=20, t=40, b=0)))
pio.templates.default = "simple_white+custom"


class GraphGenerator:
    def __init__(self, x, y, fit_func, x_err, y_err):
        self._x = x
        self._y = y
        self._fit_func = fit_func
        self._x_err = x_err
        self._y_err = y_err
        # calculate fit parameters

    def simple_plot(self, graph_title):
        fig = go.Figure(
            [go.Scatter(x=self._x, y=self._y, name="Graph", showlegend=True,
                        marker=dict(color="black", opacity=.7),
                        line=dict(color="black", width=1))],
            layout=go.Layout(title=fr"{graph_title}",
                             xaxis={"title": self._x_label},
                             yaxis={"title": self._y_label},
                             height=400))
        fig.show()

    def plot_with_fit(self, graph_title):
        # choose wide-span x axis for fit, and calculate y from fit params
        fig = go.Figure(
            [go.Scatter(x=self._x, y=self._y,
                        name="Experiment", showlegend=True,
                        marker=dict(color="black", opacity=.7),
                        line=dict(color="black", width=1)),
             go.Scatter(x=self._x, y=self._y_fit, name="Fit", showlegend=True,
                        marker=dict(color="red", opacity=.7),
                        line=dict(color="red", dash="dash", width=1))],
            layout=go.Layout(title=fr"{graph_title}",
                             xaxis={"title": self._x_label},
                             yaxis={"title": self._y_label},
                             height=400))
        fig.show()
        if not os.path.exists("graphs"):
            os.mkdir("graphs")
        fig.write_image("graphs/{}.png".format(graph_title))


