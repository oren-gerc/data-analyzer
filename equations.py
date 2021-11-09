"""
Holds equations used for fitting
"""


def linear_no_intercept(x, a):
    return a * x


def linear(x, a, b):
    return a * x + b


def parabolic_no_intercept(x, a, b):
    return a * (x ** 2) + b * x


def one_over_x(x, a, b):
    return a / x + b
