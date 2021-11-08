"""
Holds equations used for fitting
"""

import scipy.optimize as opt


def fit(func, x, y):
    return opt.curve_fit(func, x, y)[0]


def linear_no_intercept(x, a):
    return a * x


def linear(x, a, b):
    return a * x + b


def parabolic_no_intercept(x, a, b):
    return a * (x ** 2) + b * x


def one_over_x(x, a, b):
    return a / x + b
