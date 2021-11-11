"""
Runs analysis according to specific demands
"""

from DataHandler import DataHandler
from Graph import Graph, plot_many
from CurveFit import CurveFit
import equations
import numpy as np
from scipy.stats import linregress


def normalize(vec):
    # normalizes the x and y values to start with zero
    return vec - vec[0]


def calculate_r_squared(x, y):
    r_squared = np.square(normalize(x)) + np.square(normalize(y))
    return np.cumsum(r_squared) / np.arange(1, len(r_squared) + 1)


def min_length(arr_2d):
    min_len = 18000000
    for i in range(len(arr_2d)):
        if int(len(arr_2d[i])) < min_len:
            min_len = len(arr_2d[i])
    return min_len


def cut_array_equally(arr, num_chunks):
    partitions = np.array_split(arr, num_chunks)
    minimal_length = min_length(partitions)
    for i, particle in enumerate(partitions):
        partitions[i] = particle[:minimal_length]
    return partitions


def drop_drift(x):
    time = np.linspace(0, x.shape[0], num=x.shape[0])
    slope_x, intercept, r, p, se = linregress(time, x)
    after = x - slope_x * time
    return after


def calc_r_squared_error(x, y, x_error, y_error):
    return np.sqrt(np.square(2 * x * x_error) + np.square(2 * y * y_error))


def calc_average_r_squared(x, y, num_fictive_particles):
    x_partitions = cut_array_equally(x, num_fictive_particles)
    y_partitions = cut_array_equally(y, num_fictive_particles)
    r_squared_vectors = []
    for i in range(num_fictive_particles):
        r_squared_vectors.append(calculate_r_squared(x_partitions[i], y_partitions[i]))
    return np.average(r_squared_vectors, axis=0)


def calc_visc(temps):
    # the following data is taken from 'Glycerin and It's Solutions' online book, for glycerin 24%
    # visc = np.array([4.29, 2.95, 2.13, 1.61, 1.265, 1.0195, 0.8435, 0.7255])
    # temp = np.array([0, 10, 20, 30, 40, 50, 60, 70])
    # g = Graph(temp, visc)
    # g.plot_with_fit(equations.exp)
    # params = g.get_fit_parameters()
    params = 1.654, -0.6651
    values = params[0] * np.exp(params[1] * temps)
    return params, values


def analyze_effect_of_temperature():
    path = "C:\\Users\\user\\Desktop\\lab\\data-analyzer\\experiment_data\\week3\\m2.xlsx"
    data_handler = DataHandler(path)
    radii = data_handler.get_columns(['radii'])[0]
    temperatures = data_handler.get_columns(['temps'])[0]

    slopes = []
    r_s = []
    for temperature in temperatures:
        # get data and normalize it
        x, y = data_handler.get_columns(['x{}'.format(temperature), 'y{}'.format(temperature)])
        x, y = drop_drift(x), drop_drift(y)

        # calculate average r^2
        average_r_squared = calc_average_r_squared(x, y, 30)
        r_s.append(average_r_squared)

        # save r^2 back to excel
        data_handler.add_column_to_excel('r{}'.format(temperature), average_r_squared)

        # graph it to see r^2
        frames_per_second = 4.7
        graph = Graph(np.arange(1, len(average_r_squared) + 1) / frames_per_second, average_r_squared)
        graph.set_labels("r squared of particle vs time, temperature={}".format(temperature), "time", "r^2")
        graph.plot_with_fit(equations.linear_no_intercept)
        D = graph.get_fit_parameters()
        slopes.append(D[0])

    plot_many(r_s, temperatures)
    temperatures = np.array(temperatures)
    viscosities = calc_visc(temperatures)[1]
    # slopes = np.array(slopes) * np.array(radii) * viscosities
    slopes = np.array(slopes) * np.array(radii) * np.exp((0.53 * (np.arange(1, 1 + len(slopes)))))
    data_handler.add_column_to_excel("slopes*radii", slopes)
    data_handler.add_column_to_excel("visc", viscosities)
    graph = Graph(temperatures, slopes)

    temp_error = np.ones(shape=temperatures.shape)
    # a, b = calc_visc(temperatures)[0]
    value_error_constant = 4.47607
    graph.set_errors(temp_error, value_error_constant * temperatures)
    graph.set_labels("Normalized Slope of Average Squared Distance vs. Temperature", "Temperature [Celsius]",
                     "Normalized Slope [m^-18/s]")
    graph.plot_with_fit_and_errors(equations.exp)

    visc = np.array([4.29, 2.95, 2.13, 1.61, 1.265, 1.0195, 0.8435, 0.7255])
    viscosities = visc[:len(slopes)]
    slopes = np.array(slopes) * np.array(radii) * viscosities
    graph = Graph(temperatures, slopes)
    graph.set_errors(temp_error, value_error_constant * temperatures)
    graph.set_labels("Normalized Slope of Average Squared Distance vs. Temperature", "Temperature [Celsius]",
                     "Normalized Slope [m^-18/s]")
    graph.plot_with_fit_and_errors(equations.linear)


def analyze_effect_of_viscosity():
    path = "C:\\Users\\user\\Desktop\\lab\\data-analyzer\\experiment_data\\week2\\week2.xlsx"
    data_handler = DataHandler(path)
    radii = [18, 16, 16, 16]
    concentrations = [5, 10, 30, 40, 50, 60, 85, 95]
    values = []
    for concentration in concentrations:
        # get data and normalize it
        x, y = data_handler.get_columns(['x{}'.format(concentration), 'y{}'.format(concentration)])
        x, y = drop_drift(normalize(x)), drop_drift(normalize(y))

        # calculate average r^2
        average_r_squared = calc_average_r_squared(x, y, 20)

        # save r^2 back to excel
        data_handler.add_column_to_excel('r{}'.format(concentration), average_r_squared)

        # graph it to see r^2
        frames_per_second = 4.7
        graph = Graph(np.arange(1, len(average_r_squared) + 1) / frames_per_second, average_r_squared)
        graph.set_labels("r squared of particle vs time, temperature={}".format(concentration), "time", "r^2")
        # graph.plot_with_fit(equations.linear)

        # get fit params
        D = CurveFit(np.arange(1, len(average_r_squared) + 1), average_r_squared,
                     equations.linear_no_intercept).get_fit_params()
        print(D)
        values.append(D[0])

    values = np.array(values) / np.array(radii)
    data_handler.add_column_to_excel("D", values)
    graph = Graph(np.array(concentrations), values)
    graph.plot_with_fit(equations.one_over_x_no_intercept)
    # calculate errors for values and temperatures


def analyze_diffusion_coefficient_vs_time():
    excel_path = r'C:\Users\user\Desktop\lab\data-analyzer\experiment_data\week1\particle5.xlsx'
    frames_per_second = 4.7

    # read and normalize data
    d = DataHandler(excel_path)
    x, y, x_error, y_error = d.get_columns(['x', 'y', 'x error', 'y error'])
    x, y = normalize(x), normalize(y)

    # plot trajectory of the particle
    g = Graph(x, y)
    g.set_labels("2D trajectory of the particle #5", r"x [pixels]", r"y [pixels]")
    g.simple_plot()

    average_r_2 = calc_average_r_squared(x, y, 25)
    error_r = calc_r_squared_error(x, y, x_error, y_error) * 3
    time = np.linspace(0, average_r_2.shape[0], num=average_r_2.shape[0]) / frames_per_second

    g = Graph(time, average_r_2)
    g.set_labels("Average Squared Distance vs. Time", "Time [s]", "Average Squared Distance [pixels^2]")
    g.set_errors(np.zeros(time.shape), error_r)
    g.plot_with_fit_and_errors(equations.linear_no_intercept)
    D = 1 / 4 * g.get_fit_parameters()[0]
    print("Diffusion Coefficient = ", D)


def analyze_diffusion():
    
    # for each video file:
    # get it's data as frame_number images
    # from each image, get average absolute radius of outer circle (vector size = num of frames)

    # plot all vectors on the same plot
    # calculate D out of fit
    # plot D as a function of concentration/viscosity
    pass


if __name__ == '__main__':
    # analyze_diffusion_coeff_vs_time()
    # analyze_effect_of_viscosity()
    # analyze_effect_of_temperature()
    analyze_diffusion()
