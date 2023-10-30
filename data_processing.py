import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def linear_fit(x, a, b):
    return a * x + b


def square_fit(x, a, b, c):
    return a * x * x + b * x + c


def extract_data_new(file_name):
    """extract coefficients and errors from file"""
    data = np.loadtxt(file_name)
    a = data[:, 2]
    a_err = data[:, 3]
    return a, a_err


def W_calculation_upup(chi_odd, chi_odd_err, chi_even=None, chi_even_err=None, U=None, even=True):
    """W same spin polarization"""
    if even:
        W = - U * chi_odd / ((1 + U * chi_even) ** 2 - (U * chi_odd) ** 2)
        W_err = U / ((1 + U * chi_even) ** 2 - (U * chi_odd) ** 2) * chi_odd_err + \
                2 * (U * chi_odd) ** 2 * U / ((1 + U * chi_even) ** 2 - (U * chi_odd) ** 2) ** 2 * chi_odd_err + \
                2 * U ** 2 * chi_odd * (1 + U * chi_even) / ((1 + U * chi_even) ** 2 -
                                                             (U * chi_odd) ** 2) ** 2 * chi_even_err
    else:
        W = - U * chi_odd / (1 - (U * chi_odd) ** 2)
        W_err = U / (1 - (U * chi_odd) ** 2) * chi_odd_err + \
                2 * U * (U * chi_odd) ** 2 / (1 - (U * chi_odd) ** 2) ** 2 * chi_odd_err
    return W, W_err


def W_calculation_up_down(chi_odd, chi_odd_err, chi_even=None, chi_even_err=None, U=None, even=True):
    """W different spin polarization"""
    if even:
        # W = (1 - U * chi_even) / ((1 - U * chi_even) ** 2 - (U * chi_odd) ** 2)
        W = (1 + U * chi_even) / ((1 + U * chi_even) ** 2 - (U * chi_odd) ** 2)
        W_err = 2 * U ** 2 * chi_odd * (1 + U * chi_even) / (
                    (1 + U * chi_even) ** 2 - (U * chi_odd) ** 2) ** 2 * chi_odd_err + \
                2 * U * (1 + U * chi_even) ** 2 / ((1 + U * chi_even) ** 2 - (U * chi_odd) ** 2) ** 2 * chi_even_err - \
                U / ((1 + U * chi_even) ** 2 - (U * chi_odd) ** 2) * chi_even_err
    else:
        W = 1 / (1 - (U * chi_odd) ** 2)
        W_err = 2 * U ** 2 * chi_odd / (1 - (U * chi_odd) ** 2) ** 2 * chi_odd_err
    return W, W_err


def generate_chi(orders: list, basic_path, U, o=None):
    """create chi of nth order summing coeff from 0th to nth orders"""
    a, a_err, chi, chi_err = dict(), dict(), dict(), dict()
    for i, ii in enumerate(orders):
        for j in range(i + 1):
            jj = orders[j]
            o = o or jj
            a[jj], a_err[jj] = extract_data_new(basic_path + f'/o{o}/a{jj}.dat')
            if j == 0:
                chi[ii] = np.zeros(len(a[jj]))
                chi_err[ii] = np.zeros(len(a_err[jj]))
            chi[ii] += a[jj] * pow(U, jj)
            chi_err[ii] += a_err[jj] * pow(U, jj)
    return chi, chi_err


def generate_omega_sum(main_sum_input, error_sum_input, range_length):
    """sum for W over omega matsubara frequencies"""
    main_sum_output = np.zeros(range_length)
    error_sum_output = np.zeros(range_length)
    for start_value in range(range_length):
        for k in main_sum_input[start_value::range_length]:
            main_sum_output[start_value] += k
        for k in error_sum_input[start_value::range_length]:
            error_sum_output[start_value] += k
    return main_sum_output, error_sum_output


def generate_summed_w_c(input_data, range_length, w, build_plot=False):
    """sum for W over omega matsubara frequencies with different number of frequencies summed"""
    w_c = np.arange(-w, w + 1)
    output_distrib = np.zeros(len(w_c))
    output_summed_w = np.zeros((range_length, w))
    for n_of_q_points in range(range_length):
        for i in range(w):
            for k, kk in enumerate(input_data[n_of_q_points + i * range_length:- 1 - i * range_length:range_length]):
                output_distrib[k] = kk
                output_summed_w[n_of_q_points][i] += kk
            if build_plot:
                plt.plot(w_c, output_distrib, marker='.')
            output_distrib = np.zeros(len(w_c))
    if build_plot:
        plt.show()
    return output_summed_w


def generate_fitting_params(input_data, range_length, w, build_plot=False):
    """fit the W summed in omega points with linear fit"""
    output_fiting_param = np.zeros(range_length)
    output_err_fiting_param = np.zeros(range_length)
    w_c = np.arange(w, 0, -1)
    w_c = 1 / w_c
    for n_of_q_points in range(range_length):
        popt, pcov = curve_fit(linear_fit, w_c[:3], input_data[n_of_q_points][:3])
        if build_plot:
            plt.plot(w_c, linear_fit(w_c, *popt), label='fit')
            plt.plot(w_c, input_data[n_of_q_points], marker='.')
        output_fiting_param[n_of_q_points] = popt[1]
        output_err_fiting_param[n_of_q_points] = np.sqrt(np.diag(pcov))[1]
    # + abs(popt[1] - W_w_summed[n_of_q_points][0])
    if build_plot:
        plt.show()
    return output_fiting_param, output_err_fiting_param


def generate_fitting_params_curved(input_data, range_length, w, build_plot=False):
    """fit the W summed in omega points with square fit"""
    output_fiting_param = np.zeros(range_length)
    output_err_fiting_param = np.zeros(range_length)
    w_c = np.arange(w, 0, -1)
    w_c = 1 / w_c
    for n_of_q_points in range(range_length):
        popt, pcov = curve_fit(square_fit, w_c[:3], input_data[n_of_q_points][:3])
        if build_plot:
            plt.plot(w_c, square_fit(w_c, *popt), label='fit')
            plt.plot(w_c, input_data[n_of_q_points], marker='.')
        output_fiting_param[n_of_q_points] = popt[2]
        output_err_fiting_param[n_of_q_points] = np.sqrt(np.diag(pcov))[2]
    # + abs(popt[1] - W_w_summed[n_of_q_points][0])
    if build_plot:
        plt.show()
    return output_fiting_param, output_err_fiting_param


def cut(input_for_cut):
    """cut in gamma points"""
    matrix_size = int(np.sqrt(len(input_for_cut)))
    for_cut = np.reshape(input_for_cut, (matrix_size, matrix_size))
    output_for_cut = np.array([])
    output_for_cut = np.append(output_for_cut, for_cut[0, :-1])
    output_for_cut = np.append(output_for_cut, for_cut[:-1, -1])
    output_for_cut = np.append(output_for_cut, np.flip(np.diagonal(for_cut)))
    return output_for_cut


def fourier_transform(data_q, data_q_err, cut=3):
    """
    fourier transform data from momentum dependence to position
    create from data (0, pi) date of (0, 2pi) reflecting data
    """
    N = int(np.sqrt(len(data_q)))
    data_q = np.reshape(data_q, (N, N))
    data_q = np.vstack((np.hstack((data_q, np.fliplr(data_q))), np.flipud(np.hstack((data_q, np.fliplr(data_q))))))
    data_q = np.delete(data_q, -N, 0)
    data_q = np.delete(data_q, -N, 1)
    data_q_err = np.reshape(data_q_err, (N, N))
    data_q_err = np.vstack(
        (np.hstack((data_q_err, np.fliplr(data_q_err))), np.flipud(np.hstack((data_q_err, np.fliplr(data_q_err))))))
    data_q_err = np.delete(data_q_err, -N, 0)
    data_q_err = np.delete(data_q_err, -N, 1)

    N = N * 2 - 1
    shift = 0.01
    x_for_q = np.arange(shift, 2 * np.pi + shift + 0.02, 2 * np.pi / (N - 1))
    qy, qx = np.meshgrid(x_for_q, x_for_q)
    data_r = np.zeros((N, N))
    data_r_err = np.zeros((N, N))

    for x in range(N):
        for y in range(N):
            data_r[x][y] = np.sum(data_q * np.real(np.exp(1j * (qx * x + qy * y)))) / (N ** 2)
            data_r_err[x][y] = np.sum(data_q_err * np.real(np.exp(1j * (qx * x + qy * y)))) / (N ** 2)
    if cut == 1:
        N_cut = int((N + 1) / 2)
        return data_r[0, :N_cut], data_r_err[0, :N_cut]  # horizontal
    if cut == 2:
        N_cut = int((N + 1) / 2)
        return np.diagonal(data_r)[:N_cut], np.diagonal(data_r_err)[:N_cut]  # diagonal
    if cut == 3:
        return data_r[0, 0], data_r_err[0, 0], data_r[0, 1], data_r_err[0, 1], data_r[1, 1], data_r_err[1, 1]
