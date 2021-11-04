"""
Holds equations used for fitting
"""
import numpy as np


def linear_no_intercept(x, a):
    return a * x


def linear(x, a, b):
    return a * x + b


def parabolic_no_intercept(x, a, b):
    return a * (x ** 2) + b * x


def one_over_x(x, c, b, d, e):
    return c / x + b * np.exp(-d) + e
