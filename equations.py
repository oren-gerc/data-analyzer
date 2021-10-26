"""
Holds equations used for fitting
"""


def linear_no_intercept(t, a):
    return a * t


def linear(t, a, b):
    return a * t + b


def parabolic_no_intercept(t, a, b):
    return a * (t ** 2) + b * t


equations = {"linear": linear}
