from sys import exit

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


from data_processing import (W_calculation_up_down, W_calculation_upup, fourier_transform, generate_chi,
                             generate_fitting_params, generate_summed_w_c)

axis_font = {'fontname':'Arial', 'size':'15'}
BASIC_PATH = '/Users/mariagazizova/work/test_sum_omega/'
odd_orders = [0, 2, 3]
even_orders = [2, 3]
W_upup = dict()
W_upup_err = dict()

beta = [1, 2, 3, 4, 5, 6, 7, 8.33]
grid_size = 169 # 12*12 grid
omega = 15
uu_orders_trancation = [0, 2, 3, 4]
ud_orders_trancation = [1, 2, 3, 4]

def extract_W_r_dependence(U_list, beta, spin, full=False):
    W_00 = np.zeros(len(U_list))
    W_00_err = np.zeros(len(U_list))
    W_01 = np.zeros(len(U_list))
    W_01_err = np.zeros(len(U_list))
    W_11 = np.zeros(len(U_list))
    W_11_err = np.zeros(len(U_list))
    for u, uu in enumerate(U_list):
        int_beta = int(beta)
        odd_u, odd_err_u = generate_chi(odd_orders, BASIC_PATH + f'omega_q/beta{int_beta}/odd', uu, 3)
        even_u, even_err_u = generate_chi(even_orders, BASIC_PATH + f'omega_q/beta{int_beta}/even', uu, 3)
        if spin:
            W_u, W_u_err = W_calculation_upup(odd_u[3], odd_err_u[3], even_u[3], even_err_u[3], uu)
        else:
            W_u, W_u_err = W_calculation_up_down(odd_u[3], odd_err_u[3], even_u[3], even_err_u[3], uu)
            W_u = W_u - 1
        if full:
            W_u *= uu
        W_u_w_c_summed = generate_summed_w_c(W_u, grid_size, 15)
        W_u_w_c_summed_err = generate_summed_w_c(W_u_err, grid_size, 15)
        W_u_fp, W_u_err_fp = generate_fitting_params(W_u_w_c_summed, grid_size, 15)
        W_u_fp, W_u_err_fp = W_u_fp / beta, (W_u_err_fp + W_u_w_c_summed_err[:, 0]) / beta
        if not spin:
            if full:
                W_u_fp += uu
                W_u_err_fp *= uu
            else:
                W_u_fp += 1
        # W_u_ft, W_u_ft_err = fourier_transform(W_u_fp, W_u_err_fp, cut=1)
        # plt.errorbar(np.arange(len(W_u_ft)), W_u_ft, yerr=abs(W_u_ft_err), markersize=5, capsize=2, capthick=2, fmt='.-', label=uu)
        W_00[u], W_00_err[u], W_01[u], W_01_err[u], W_11[u], W_11_err[u] = fourier_transform(W_u_fp, W_u_err_fp)
    # plt.legend()
    # plt.show()
    return W_00, W_00_err, W_01, W_01_err, W_11, W_11_err


U_list = np.arange(0, 4.2, 0.2)
W_00_1212_uu, W_00_err_1212_uu, W_01_1212_uu, W_01_err_1212_uu, W_11_1212_uu, W_11_err_1212_uu = \
    [0 for i in range(len(beta))], [0 for i in range(len(beta))], [0 for i in range(len(beta))], \
    [0 for i in range(len(beta))], [0 for i in range(len(beta))], [0 for i in range(len(beta))]
W_00_1212_ud, W_00_err_1212_ud, W_01_1212_ud, W_01_err_1212_ud, W_11_1212_ud, W_11_err_1212_ud = \
    [0 for i in range(len(beta))], [0 for i in range(len(beta))], [0 for i in range(len(beta))], \
    [0 for i in range(len(beta))], [0 for i in range(len(beta))], [0 for i in range(len(beta))]
for b, bb in enumerate(beta):
    W_00_1212_uu[b], W_00_err_1212_uu[b], W_01_1212_uu[b], W_01_err_1212_uu[b], W_11_1212_uu[b], W_11_err_1212_uu[b] =\
        extract_W_r_dependence(U_list, bb, True)
    W_00_1212_ud[b], W_00_err_1212_ud[b], W_01_1212_ud[b], W_01_err_1212_ud[b], W_11_1212_ud[b], W_11_err_1212_ud[b] =\
        extract_W_r_dependence(U_list, bb, False)

W_00_full_uu, W_00_err_full_uu, W_01_full_uu, W_01_err_full_uu, W_11_full_uu, W_11_err_full_uu = \
    [0 for i in range(len(beta))], [0 for i in range(len(beta))], [0 for i in range(len(beta))], \
    [0 for i in range(len(beta))], [0 for i in range(len(beta))], [0 for i in range(len(beta))]
W_00_full_ud, W_00_err_full_ud, W_01_full_ud, W_01_err_full_ud, W_11_full_ud, W_11_err_full_ud = \
    [0 for i in range(len(beta))], [0 for i in range(len(beta))], [0 for i in range(len(beta))], \
    [0 for i in range(len(beta))], [0 for i in range(len(beta))], [0 for i in range(len(beta))]
for b, bb in enumerate(beta):
    W_00_full_uu[b], W_00_err_full_uu[b], W_01_full_uu[b], W_01_err_full_uu[b], W_11_full_uu[b], W_11_err_full_uu[b] =\
        extract_W_r_dependence(U_list, bb, True, full=True)
    W_00_full_ud[b], W_00_err_full_ud[b], W_01_full_ud[b], W_01_err_full_ud[b], W_11_full_ud[b], W_11_err_full_ud[b] =\
        extract_W_r_dependence(U_list, bb, False, full=True)




# data = np.column_stack((U_list, W_00_1212_uu[5], W_00_1212_ud[5]))
# np.savetxt("data_beta6.txt", data, fmt='%.9e')
#
# data = np.column_stack((U_list, W_00_1212_uu[6], W_00_1212_ud[6]))
# np.savetxt("data_beta7.txt", data, fmt='%.9e')
#
# data = np.column_stack((U_list, W_00_1212_uu[7], W_00_1212_ud[7]))
# np.savetxt("data_beta8.txt", data, fmt='%.9e')

# data_uu, data_ud = [], []
data_uu_01, data_ud_01 = [], []
# data_uu_11, data_ud_11 = [], []
for i in range(len(beta)):
    # data_uu.append(W_00_1212_uu[i])
    # data_ud.append(W_00_1212_ud[i])
    data_uu_01.append(W_01_1212_uu[i])
    data_ud_01.append(W_01_1212_ud[i])
    # data_uu_11.append(W_11_1212_uu[i])
    # data_ud_11.append(W_11_1212_ud[i])
# data_uu, data_ud = np.array(data_uu), np.array(data_ud)
data_uu_01, data_ud_01 = np.array(data_uu_01), np.array(data_ud_01)
# np.savetxt("data_W_uu.txt", data_uu, fmt='%.9e')
# np.savetxt("data_W_ud.txt", data_ud, fmt='%.9e')
# np.savetxt("data_W_uu_01.txt", data_uu_01, fmt='%.9e')
np.savetxt("data_W_ud_01.txt", data_ud_01, fmt='%.9e')
# np.savetxt("data_W_uu_11.txt", data_uu_11, fmt='%.9e')
# np.savetxt("data_W_ud_11.txt", data_ud_11, fmt='%.9e')


# exit(0)


u_ind = 10
u_ind_1 = 15
print(U_list[15])
W_00_uu_beta, W_00_ud_beta, W_00_uu_beta_err, W_00_ud_beta_err = np.zeros(len(beta)), np.zeros(len(beta)), np.zeros(len(beta)), np.zeros(len(beta))
W_01_uu_beta, W_01_ud_beta, W_01_uu_beta_err, W_01_ud_beta_err = np.zeros(len(beta)), np.zeros(len(beta)), np.zeros(len(beta)), np.zeros(len(beta))
W_11_uu_beta, W_11_ud_beta, W_11_uu_beta_err, W_11_ud_beta_err = np.zeros(len(beta)), np.zeros(len(beta)), np.zeros(len(beta)), np.zeros(len(beta))
W_00_uu_beta_1, W_00_ud_beta_1, W_00_uu_beta_err_1, W_00_ud_beta_err_1 = np.zeros(len(beta)), np.zeros(len(beta)), np.zeros(len(beta)), np.zeros(len(beta))
W_01_uu_beta_1, W_01_ud_beta_1, W_01_uu_beta_err_1, W_01_ud_beta_err_1 = np.zeros(len(beta)), np.zeros(len(beta)), np.zeros(len(beta)), np.zeros(len(beta))
W_11_uu_beta_1, W_11_ud_beta_1, W_11_uu_beta_err_1, W_11_ud_beta_err_1 = np.zeros(len(beta)), np.zeros(len(beta)), np.zeros(len(beta)), np.zeros(len(beta))
for i in range(len(W_00_1212_uu)):
    W_00_uu_beta[i], W_00_uu_beta_err[i] = W_00_1212_uu[i][u_ind], W_00_err_1212_uu[i][u_ind]
    W_00_ud_beta[i], W_00_ud_beta_err[i] = W_00_1212_ud[i][u_ind], W_00_err_1212_ud[i][u_ind]
    W_01_uu_beta[i], W_01_uu_beta_err[i] = W_01_1212_uu[i][u_ind], W_01_err_1212_uu[i][u_ind]
    W_01_ud_beta[i], W_01_ud_beta_err[i] = W_01_1212_ud[i][u_ind], W_01_err_1212_ud[i][u_ind]
    W_11_uu_beta[i], W_11_uu_beta_err[i] = W_11_1212_uu[i][u_ind], W_11_err_1212_uu[i][u_ind]
    W_11_ud_beta[i], W_11_ud_beta_err[i] = W_11_1212_ud[i][u_ind], W_11_err_1212_ud[i][u_ind]
for i in range(len(W_00_1212_uu)):
    W_00_uu_beta_1[i], W_00_uu_beta_err_1[i] = W_00_1212_uu[i][u_ind_1], W_00_err_1212_uu[i][u_ind_1]
    W_00_ud_beta_1[i], W_00_ud_beta_err_1[i] = W_00_1212_ud[i][u_ind_1], W_00_err_1212_ud[i][u_ind_1]
    W_01_uu_beta_1[i], W_01_uu_beta_err_1[i] = W_01_1212_uu[i][u_ind_1], W_01_err_1212_uu[i][u_ind_1]
    W_01_ud_beta_1[i], W_01_ud_beta_err_1[i] = W_01_1212_ud[i][u_ind_1], W_01_err_1212_ud[i][u_ind_1]
    W_11_uu_beta_1[i], W_11_uu_beta_err_1[i] = W_11_1212_uu[i][u_ind_1], W_11_err_1212_uu[i][u_ind_1]
    W_11_ud_beta_1[i], W_11_ud_beta_err_1[i] = W_11_1212_ud[i][u_ind_1], W_11_err_1212_ud[i][u_ind_1]


data_W_00_beta = np.column_stack((beta, W_00_uu_beta, W_00_uu_beta_err, W_00_uu_beta_1, W_00_uu_beta_err_1,
                                  W_00_ud_beta, W_00_ud_beta_err, W_00_ud_beta_1, W_00_ud_beta_err_1))
data_W_01_beta = np.column_stack((beta, W_01_uu_beta, W_01_uu_beta_err, W_01_uu_beta_1, W_01_uu_beta_err_1,
                                  W_01_ud_beta, W_01_ud_beta_err, W_01_ud_beta_1, W_01_ud_beta_err_1))
data_W_11_beta = np.column_stack((beta, W_11_uu_beta, W_11_uu_beta_err, W_11_uu_beta_1, W_11_uu_beta_err_1,
                                  W_11_ud_beta, W_11_ud_beta_err, W_11_ud_beta_1, W_11_ud_beta_err_1))
np.savetxt("data_W_00_beta.txt", data_W_00_beta, fmt='%.9e')
np.savetxt("data_W_01_beta.txt", data_W_01_beta, fmt='%.9e')
np.savetxt("data_W_11_beta.txt", data_W_11_beta, fmt='%.9e')

# exit(0)



def extract_W_r_dependence_trunc(U_list, beta, spin):
    W_00 = np.zeros(len(U_list))
    W_00_err= np.zeros(len(U_list))
    W_01 = np.zeros(len(U_list))
    W_01_err = np.zeros(len(U_list))
    W_11 = np.zeros(len(U_list))
    W_11_err = np.zeros(len(U_list))
    for u, uu in enumerate(U_list):
        int_beta = int(beta)
        if spin:
            W_u, W_err_u = generate_chi(uu_orders_trancation, BASIC_PATH + f'sum_omega_trancation/beta{int_beta}/uu', uu, 4)
        else:
            W_u, W_err_u = generate_chi(ud_orders_trancation, BASIC_PATH + f'sum_omega_trancation/beta{int_beta}/ud', uu, 4)
        for keys in W_u:
            W_u[keys] = W_u[keys] * uu * (-1)
            W_err_u[keys] = W_err_u[keys] * uu * (-1)
        if not spin:
            W_u[4] = W_u[4] + 1
        W_00[u], W_00_err[u], W_01[u], W_01_err[u], W_11[u], W_11_err[u] = fourier_transform(W_u[4], W_err_u[4])
    return W_00, W_00_err, W_01, W_01_err, W_11, W_11_err

beta1 = [1, 2, 3, 4, 5, 6, 7, 8.33]
W_00_uu, W_00_err_uu, W_01_uu, W_01_err_uu, W_11_uu, W_11_err_uu = \
    [0 for i in range(len(beta1))], [0 for i in range(len(beta1))], [0 for i in range(len(beta1))], \
    [0 for i in range(len(beta1))], [0 for i in range(len(beta1))], [0 for i in range(len(beta1))]
W_00_ud, W_00_err_ud, W_01_ud, W_01_err_ud, W_11_ud, W_11_err_ud = \
    [0 for i in range(len(beta1))], [0 for i in range(len(beta1))], [0 for i in range(len(beta1))], \
    [0 for i in range(len(beta1))], [0 for i in range(len(beta1))], [0 for i in range(len(beta1))]
for b, bb in enumerate(beta1):
    W_00_uu[b], W_00_err_uu[b], W_01_uu[b], W_01_err_uu[b], W_11_uu[b], W_11_err_uu[b] =\
        extract_W_r_dependence_trunc(U_list, bb, True)
    W_00_ud[b], W_00_err_ud[b], W_01_ud[b], W_01_err_ud[b], W_11_ud[b], W_11_err_ud[b] =\
        extract_W_r_dependence_trunc(U_list, bb, False)


W_00_uu_beta_trunc, W_00_ud_beta_trunc, W_00_uu_beta_err_trunc, W_00_ud_beta_err_trunc = np.zeros(len(beta)), np.zeros(len(beta)), np.zeros(len(beta)), np.zeros(len(beta))
W_01_uu_beta_trunc, W_01_ud_beta_trunc, W_01_uu_beta_err_trunc, W_01_ud_beta_err_trunc = np.zeros(len(beta)), np.zeros(len(beta)), np.zeros(len(beta)), np.zeros(len(beta))
W_11_uu_beta_trunc, W_11_ud_beta_trunc, W_11_uu_beta_err_trunc, W_11_ud_beta_err_trunc = np.zeros(len(beta)), np.zeros(len(beta)), np.zeros(len(beta)), np.zeros(len(beta))
W_00_uu_beta_1_trunc, W_00_ud_beta_1_trunc, W_00_uu_beta_err_1_trunc, W_00_ud_beta_err_1_trunc = np.zeros(len(beta)), np.zeros(len(beta)), np.zeros(len(beta)), np.zeros(len(beta))
W_01_uu_beta_1_trunc, W_01_ud_beta_1_trunc, W_01_uu_beta_err_1_trunc, W_01_ud_beta_err_1_trunc = np.zeros(len(beta)), np.zeros(len(beta)), np.zeros(len(beta)), np.zeros(len(beta))
W_11_uu_beta_1_trunc, W_11_ud_beta_1_trunc, W_11_uu_beta_err_1_trunc, W_11_ud_beta_err_1_trunc = np.zeros(len(beta)), np.zeros(len(beta)), np.zeros(len(beta)), np.zeros(len(beta))
for i in range(len(W_00_uu)):
    W_00_uu_beta_trunc[i], W_00_uu_beta_err_trunc[i] = W_00_uu[i][u_ind], W_00_err_uu[i][u_ind]
    W_00_ud_beta_trunc[i], W_00_ud_beta_err_trunc[i] = W_00_ud[i][u_ind], W_00_err_ud[i][u_ind]
    W_01_uu_beta_trunc[i], W_01_uu_beta_err_trunc[i] = W_01_uu[i][u_ind], W_01_err_uu[i][u_ind]
    W_01_ud_beta_trunc[i], W_01_ud_beta_err_trunc[i] = W_01_ud[i][u_ind], W_01_err_ud[i][u_ind]
    W_11_uu_beta_trunc[i], W_11_uu_beta_err_trunc[i] = W_11_uu[i][u_ind], W_11_err_uu[i][u_ind]
    W_11_ud_beta_trunc[i], W_11_ud_beta_err_trunc[i] = W_11_ud[i][u_ind], W_11_err_ud[i][u_ind]
for i in range(len(W_00_uu)):
    W_00_uu_beta_1_trunc[i], W_00_uu_beta_err_1_trunc[i] = W_00_uu[i][u_ind_1], W_00_err_uu[i][u_ind_1]
    W_00_ud_beta_1_trunc[i], W_00_ud_beta_err_1_trunc[i] = W_00_ud[i][u_ind_1], W_00_err_ud[i][u_ind_1]
    W_01_uu_beta_1_trunc[i], W_01_uu_beta_err_1_trunc[i] = W_01_uu[i][u_ind_1], W_01_err_uu[i][u_ind_1]
    W_01_ud_beta_1_trunc[i], W_01_ud_beta_err_1_trunc[i] = W_01_ud[i][u_ind_1], W_01_err_ud[i][u_ind_1]
    W_11_uu_beta_1_trunc[i], W_11_uu_beta_err_1_trunc[i] = W_11_uu[i][u_ind_1], W_11_err_uu[i][u_ind_1]
    W_11_ud_beta_1_trunc[i], W_11_ud_beta_err_1_trunc[i] = W_11_ud[i][u_ind_1], W_11_err_ud[i][u_ind_1]



# data_uu_01, data_ud_01 = [], []
# for i in range(len(beta)):
#     data_uu_01.append(W_01_uu[i])
#     data_ud_01.append(W_01_ud[i])
# data_uu_01, data_ud_01 = np.array(data_uu_01), np.array(data_ud_01)
# np.savetxt("data_W_trunc_uu_01.txt", data_uu_01, fmt='%.9e')
# np.savetxt("data_W_trunc_ud_01.txt", data_ud_01, fmt='%.9e')

plt.rc('font', size=16)
plt.rc('axes', labelsize=16)
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.figure(figsize=(10, 7))
# plt.figure(figsize=(9, 7))
plt.subplot(3, 2, 1)
plt.text(0.2, -0.35, '(a)', **axis_font)
plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=4))
# plt.ylim(0.2, 1.3)
# plt.axhline(1, linestyle='dashed', color='r')
plt.ylabel(r'$W_{\uparrow \uparrow}(r,\tau = 0)/U$', **axis_font)
plt.xticks([])
# plt.errorbar(U_list, W_00_1212_uu[0], yerr=abs(W_00_err_1212_uu[0]), markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$\beta = 1$')
# plt.errorbar(U_list, W_00_1212_uu[1], yerr=abs(W_00_err_1212_uu[1]), markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$\beta = 2$')
# plt.errorbar(U_list, W_00_1212_uu[2], yerr=abs(W_00_err_1212_uu[2]), markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$\beta = 4$')
plt.errorbar(U_list, W_00_1212_uu[4], yerr=abs(W_00_err_1212_uu[4]), markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$W_{r=0}$')
# plt.errorbar(U_list, W_00_uu[0], yerr=abs(W_00_err_uu[0]), markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$trunc, \beta = 1$')
# plt.errorbar(U_list, W_00_uu[1], yerr=abs(W_00_err_uu[1]), markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$trunc, \beta = 2$')
# plt.errorbar(U_list, W_00_uu[2], yerr=abs(W_00_err_uu[2]), markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$trunc, \beta = 5$')
plt.errorbar(U_list, W_00_uu[4], yerr=abs(W_00_err_uu[4]), markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$W_{r=0}$')
plt.legend(prop={'family': 'Arial', 'size': 12}, ncol=3, fontsize=14, loc='best', fancybox=True, framealpha=0)
plt.subplot(3, 2, 2)
plt.text(0.2, 0.07, '(d)', **axis_font)
plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=4))
plt.xticks([])
plt.ylim(-0.1, 0.2)
# plt.ylabel(r'$W_{\uparrow \uparrow}(r,\tau = 0 )/U$')
# plt.errorbar(U_list, W_01_1212_uu[0], yerr=abs(W_01_err_1212_uu[0]), markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$\beta = 1$')
# plt.errorbar(U_list, W_01_1212_uu[1], yerr=abs(W_01_err_1212_uu[1]), markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$ \beta = 2$')
# plt.errorbar(U_list, W_01_1212_uu[2], yerr=abs(W_01_err_1212_uu[2]), markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$\beta = 4$')
plt.errorbar(U_list, W_01_1212_uu[4], yerr=abs(W_01_err_1212_uu[4]), markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$W_{r=(0, 1)}$')

# plt.errorbar(U_list, W_11_1212_uu[0], yerr=abs(W_11_err_1212_uu[0]),  markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$ \beta = 1$')
# plt.errorbar(U_list, W_11_1212_uu[1], yerr=abs(W_11_err_1212_uu[1]),  markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$\beta = 2$')
# plt.errorbar(U_list, W_11_1212_uu[2], yerr=abs(W_11_err_1212_uu[2]),  markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$\beta = 4$')
plt.errorbar(U_list, W_11_1212_uu[4], yerr=abs(W_11_err_1212_uu[4]),  markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$W_{r=(1, 1)}$')
# plt.errorbar(U_list, W_01_uu, yerr=abs(W_01_err_uu), markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$W_{r = (0,1)}^{(3)}$')
# plt.errorbar(U_list, W_11_uu, yerr=abs(W_11_err_uu),  markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$W_{r = (1,1)}^{(3)}$')

# plt.errorbar(U_list, W_01_uu[0], yerr=abs(W_01_err_uu[0]), markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$trunc, \beta = 1$')
# plt.errorbar(U_list, W_11_uu[0], yerr=abs(W_11_err_uu[0]),  markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$trunc, \beta = 1$')
# plt.errorbar(U_list, W_01_uu[1], yerr=abs(W_01_err_uu[1]), markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$trunc, \beta = 2$')
# plt.errorbar(U_list, W_11_uu[1], yerr=abs(W_11_err_uu[1]),  markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$trunc, \beta = 2$')
# plt.errorbar(U_list, W_01_uu[2], yerr=abs(W_01_err_uu[2]), markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$trunc, \beta = 4$')
# plt.errorbar(U_list, W_11_uu[2], yerr=abs(W_11_err_uu[2]),  markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$trunc, \beta = 4$')
plt.errorbar(U_list, W_01_uu[4], yerr=abs(W_01_err_uu[4]), markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$trunc, \beta = 5$')
plt.errorbar(U_list, W_11_uu[4], yerr=abs(W_11_err_uu[4]),  markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$trunc, \beta = 5$')
plt.legend(prop={'family': 'Arial', 'size': 12}, ncol=3, fontsize=14, loc='best', fancybox=True, framealpha=0)
plt.subplot(3, 2, 3)
plt.text(0.2, 1.3, '(b)', **axis_font)
plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=4))
plt.ylabel(r'$W_{\uparrow \downarrow}(r, \tau = 0) / U$', **axis_font)
plt.xticks([])
# plt.xlabel('U')
# plt.errorbar(U_list, W_00_ud[0], yerr=abs(W_00_err_ud[0]), markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$trunc, \beta = 1$')
# plt.errorbar(U_list, W_00_ud[1], yerr=abs(W_00_err_ud[1]), markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$trunc, \beta = 2$')
# plt.errorbar(U_list, W_00_ud[2], yerr=abs(W_00_err_ud[2]), markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$trunc, \beta = 5$')
plt.errorbar(U_list, W_00_ud[4], yerr=abs(W_00_err_ud[4]), markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$trunc, \beta = 5$')
# plt.errorbar(U_list, W_00_1212_ud[0], yerr=abs(W_00_err_1212_ud[0]), markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$\beta = 1$')
# plt.errorbar(U_list, W_00_1212_ud[1], yerr=abs(W_00_err_1212_ud[1]), markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$\beta = 2$')
# plt.errorbar(U_list, W_00_1212_ud[2], yerr=abs(W_00_err_1212_ud[2]), markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$\beta = 4$')
plt.errorbar(U_list, W_00_1212_ud[4], yerr=abs(W_00_err_1212_ud[4]), markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$W_{r=0}$')
# plt.errorbar(U_list, W_00_1212_uu[4] + W_00_1212_ud[4], yerr=abs(W_00_err_1212_ud[4] + W_00_err_1212_uu[4]), markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$W_{r=0}$')

plt.legend(prop={'family': 'Arial', 'size': 12}, ncol=3, fontsize=14, loc='best', fancybox=True, framealpha=0)
plt.subplot(3, 2, 4)
# plt.ylim(-0.1, 0.17)
plt.ylim(-0.1, 0.2)
plt.text(0.2, 0.07, '(e)', **axis_font)
plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=4))
# plt.ylabel(r'$W_{\uparrow \downarrow}(r, \tau = 0) / U$')
plt.xticks([])
# plt.xlabel('U', **axis_font)
# plt.errorbar(U_list, W_01_ud[0], yerr=abs(W_01_err_ud[0]), markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$trunc \beta = 1$')
# plt.errorbar(U_list, W_11_ud[0], yerr=abs(W_11_err_ud[0]),  markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$trunc \beta = 1$')
# plt.errorbar(U_list, W_01_ud[1], yerr=abs(W_01_err_ud[1]), markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$trunc \beta = 2$')
# plt.errorbar(U_list, W_11_ud[1], yerr=abs(W_11_err_ud[1]),  markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$trunc \beta = 2$')
# plt.errorbar(U_list, W_01_ud[2], yerr=abs(W_01_err_ud[2]), markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$trunc \beta = 5$')
# plt.errorbar(U_list, W_11_ud[2], yerr=abs(W_11_err_ud[2]),  markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$trunc \beta = 5$')

plt.errorbar(U_list, W_01_ud[4], yerr=abs(W_01_err_ud[4]), markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$trunc \beta = 5$')
plt.errorbar(U_list, W_11_ud[4], yerr=abs(W_11_err_ud[4]),  markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$trunc \beta = 5$')

# plt.errorbar(U_list, W_01_1212_ud[0], yerr=abs(W_01_err_1212_ud[0]), markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$\beta = 1$')
# plt.errorbar(U_list, W_01_1212_ud[1], yerr=abs(W_01_err_1212_ud[1]), markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$\beta = 2$')
# plt.errorbar(U_list, W_01_1212_ud[2], yerr=abs(W_01_err_1212_ud[2]), markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$\beta = 4$')
plt.errorbar(U_list, W_01_1212_ud[4], yerr=abs(W_01_err_1212_ud[4]), markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$W_{r=(0, 1)}$')
# plt.errorbar(U_list, W_11_1212_ud[0], yerr=abs(W_11_err_1212_ud[0]),  markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$\beta = 1$')
# plt.errorbar(U_list, W_11_1212_ud[1], yerr=abs(W_11_err_1212_ud[1]),  markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$\beta = 2$')
# plt.errorbar(U_list, W_11_1212_ud[2], yerr=abs(W_11_err_1212_ud[2]),  markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$\beta = 4$')
plt.errorbar(U_list, W_11_1212_ud[4], yerr=abs(W_11_err_1212_ud[4]),  markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$W_{r=(1, 1)}$')
plt.legend(prop={'family': 'Arial', 'size': 12}, ncol=3, fontsize=14, loc='best', fancybox=True, framealpha=0)

plt.subplot(3, 2, 5)
plt.text(0.2, 3.3, '(c)', **axis_font)
plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=4))
plt.ylabel(r'$W_{\uparrow \downarrow}(r, \tau = 0)$', **axis_font)
# plt.xticks([])
plt.xlabel('U', **axis_font)
x = np.linspace(0, 4.2, 100)
y = x
plt.plot(x, y, '--', color='black')
# plt.errorbar(U_list, W_00_ud[0], yerr=abs(W_00_err_ud[0]), markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$trunc, \beta = 1$')
# plt.errorbar(U_list, W_00_ud[1], yerr=abs(W_00_err_ud[1]), markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$trunc, \beta = 2$')
# plt.errorbar(U_list, W_00_ud[2], yerr=abs(W_00_err_ud[2]), markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$trunc, \beta = 5$')
# plt.errorbar(U_list, W_00_ud[3], yerr=abs(W_00_err_ud[3]), markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$trunc, \beta = 5$')
# plt.errorbar(U_list, W_00_1212_ud[0], yerr=abs(W_00_err_1212_ud[0]), markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$\beta = 1$')
# plt.errorbar(U_list, W_00_1212_ud[1], yerr=abs(W_00_err_1212_ud[1]), markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$\beta = 2$')
# plt.errorbar(U_list, W_00_1212_ud[2], yerr=abs(W_00_err_1212_ud[2]), markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$\beta = 4$')
plt.errorbar(U_list, W_00_full_ud[4], yerr=abs(W_00_err_full_ud[4]), markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$W_{r=0}$')
plt.legend(prop={'family': 'Arial', 'size': 12}, ncol=3, fontsize=14, loc='best', fancybox=True, framealpha=0)

plt.subplot(3, 2, 6)
# plt.ylim(-0.12, 0.15)
# plt.ylim(-0.28, 0.3)
plt.ylim(-0.1, 0.2)
plt.text(0.2, 0.065, '(f)', **axis_font)
plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=4))
# plt.ylabel(r'$W_{\uparrow \downarrow}(r, \tau = 0) / U$')
plt.xlabel('U', **axis_font)
# plt.errorbar(U_list, W_01_ud[0], yerr=abs(W_01_err_ud[0]), markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$trunc \beta = 1$')
# plt.errorbar(U_list, W_11_ud[0], yerr=abs(W_11_err_ud[0]),  markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$trunc \beta = 1$')
# plt.errorbar(U_list, W_01_ud[1], yerr=abs(W_01_err_ud[1]), markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$trunc \beta = 2$')
# plt.errorbar(U_list, W_11_ud[1], yerr=abs(W_11_err_ud[1]),  markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$trunc \beta = 2$')
# plt.errorbar(U_list, W_01_ud[2], yerr=abs(W_01_err_ud[2]), markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$trunc \beta = 5$')
# plt.errorbar(U_list, W_11_ud[2], yerr=abs(W_11_err_ud[2]),  markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$trunc \beta = 5$')

# plt.errorbar(U_list, W_01_ud[4], yerr=abs(W_01_err_ud[4]), markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$trunc \beta = 5$')
# plt.errorbar(U_list, W_11_ud[4], yerr=abs(W_11_err_ud[4]),  markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$trunc \beta = 5$')

# plt.errorbar(U_list, W_01_1212_ud[0], yerr=abs(W_01_err_1212_ud[0]), markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$\beta = 1$')
# plt.errorbar(U_list, W_01_1212_ud[1], yerr=abs(W_01_err_1212_ud[1]), markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$\beta = 2$')
# plt.errorbar(U_list, W_01_1212_ud[2], yerr=abs(W_01_err_1212_ud[2]), markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$\beta = 4$')
plt.errorbar(U_list, W_01_full_ud[4], yerr=abs(W_01_err_full_ud[4]), markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$W_{r=(0, 1)}$')
# plt.errorbar(U_list, W_11_1212_ud[0], yerr=abs(W_11_err_1212_ud[0]),  markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$\beta = 1$')
# plt.errorbar(U_list, W_11_1212_ud[1], yerr=abs(W_11_err_1212_ud[1]),  markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$\beta = 2$')
# plt.errorbar(U_list, W_11_1212_ud[2], yerr=abs(W_11_err_1212_ud[2]),  markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$\beta = 4$')
plt.errorbar(U_list, W_11_full_ud[4], yerr=abs(W_11_err_full_ud[4]),  markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$W_{r=(1, 1)}$')
plt.legend(prop={'family': 'Arial', 'size': 12}, ncol=3, fontsize=14, loc='best', fancybox=True, framealpha=0)
plt.subplots_adjust(left=0.15,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.2,
                    hspace=0)
plt.savefig(f'/Users/mariagazizova/work/graph_new_after_sign_problem/sum_W_ft_odd_even_1.png', dpi=300, bbox_inches='tight')
plt.show()


exit(0)


plt.figure(figsize=(14, 6))
plt.subplot(1, 3, 1)
plt.text( r'$r = 0$', **axis_font)
plt.errorbar(beta, W_00_uu_beta, yerr= abs(W_00_uu_beta_err), markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$W_{\uparrow \uparrow}{r=0}, U=2$')
plt.errorbar(beta, W_00_ud_beta, yerr= abs(W_00_ud_beta_err), markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$W_{\uparrow \downarrow}{r=0}, U=2$')
plt.errorbar(beta, W_00_uu_beta_1, yerr= abs(W_00_uu_beta_err_1), markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$W_{\uparrow \uparrow}{r=0}, U=3$')
plt.errorbar(beta, W_00_ud_beta_1, yerr= abs(W_00_ud_beta_err_1), markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$W_{\uparrow \downarrow}{r=0}, U=3$')

# plt.errorbar(beta, W_00_uu_beta_trunc, yerr= abs(W_00_uu_beta_err_trunc), markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$W_{\uparrow \uparrow}{r=0}, U=2$')
# plt.errorbar(beta, W_00_ud_beta_trunc, yerr= abs(W_00_ud_beta_err_trunc), markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$W_{\uparrow \downarrow}{r=0}, U=2$')
# plt.errorbar(beta, W_00_uu_beta_1_trunc, yerr= abs(W_00_uu_beta_err_1_trunc), markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$W_{\uparrow \uparrow}{r=0}, U=3$')
# plt.errorbar(beta, W_00_ud_beta_1_trunc, yerr= abs(W_00_ud_beta_err_1_trunc), markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$W_{\uparrow \downarrow}{r=0}, U=3$')
plt.ylabel(r'$W(r, \tau = 0) / U$')
plt.legend(ncol=1, fontsize=12, loc='best', fancybox=True, framealpha=0)
plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=4))
plt.xlabel(r'$\beta/t$')
plt.subplot(1, 3, 2)
plt.ylim(-0.01, 0.14)
plt.text( r'$r = (0, 1)$', **axis_font)
plt.errorbar(beta, W_01_uu_beta, yerr=abs(W_01_uu_beta_err), markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$W_{\uparrow \uparrow}{r=0, 1}, U=2$')
plt.errorbar(beta, W_01_ud_beta, yerr= abs(W_01_ud_beta_err), markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$W_{\uparrow \downarrow}{r=0, 1}, U=2$')
plt.errorbar(beta, W_01_uu_beta_1, yerr= abs(W_01_uu_beta_err_1), markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$W_{\uparrow \uparrow}{r=0, 1}, U=3$')
plt.errorbar(beta, W_01_ud_beta_1, yerr= abs(W_01_ud_beta_err_1), markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$W_{\uparrow \downarrow}{r=0, 1}, U=3$')

# plt.errorbar(beta, W_01_uu_beta_trunc, yerr= abs(W_01_uu_beta_err_trunc), markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$W_{\uparrow \uparrow}{r=0, 1}, U=2$')
# plt.errorbar(beta, W_01_ud_beta_trunc, yerr= abs(W_01_ud_beta_err_trunc), markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$W_{\uparrow \downarrow}{r=0, 1}, U=2$')
# plt.errorbar(beta, W_01_uu_beta_1_trunc, yerr= abs(W_01_uu_beta_err_1_trunc), markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$W_{\uparrow \uparrow}{r=0, 1}, U=3$')
# plt.errorbar(beta, W_01_ud_beta_1_trunc, yerr= abs(W_01_ud_beta_err_1_trunc), markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$W_{\uparrow \downarrow}{r=0, 1}, U=3$')
plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=4))
plt.legend(ncol=1, fontsize=12, loc='best', fancybox=True, framealpha=0)
plt.xlabel(r'$\beta/t$')
plt.subplot(1, 3, 3)
plt.ylim(0, 0.02)
plt.text( r'$r = (1, 1)$', **axis_font)
plt.errorbar(beta, W_11_uu_beta, yerr= abs(W_11_uu_beta_err), markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$W_{\uparrow \uparrow}{r=1, 1}, U=2$')
plt.errorbar(beta, W_11_ud_beta, yerr=abs(W_11_ud_beta_err), markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$W_{\uparrow \downarrow}{r=1, 1}, U=2$')
plt.errorbar(beta, W_11_uu_beta_1, yerr= abs(W_11_uu_beta_err_1), markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$W_{\uparrow \uparrow}{r=1, 1}, U=3$')
plt.errorbar(beta, W_11_ud_beta_1, yerr=abs(W_11_ud_beta_err_1), markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$W_{\uparrow \downarrow}{r=1, 1}, U=3$')

# plt.errorbar(beta, W_11_uu_beta_trunc, yerr = abs(W_11_uu_beta_err_trunc), markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$W_{\uparrow \uparrow}{r=1, 1}, U=2$')
# plt.errorbar(beta, W_11_ud_beta_trunc, yerr = abs(W_11_ud_beta_err_trunc), markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$W_{\uparrow \downarrow}{r=1, 1}, U=2$')
# plt.errorbar(beta, W_11_uu_beta_1_trunc, yerr = abs(W_11_uu_beta_err_1_trunc), markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$W_{\uparrow \uparrow}{r=1, 1}, U=3$')
# plt.errorbar(beta, W_11_ud_beta_1_trunc, yerr = abs(W_11_ud_beta_err_1_trunc), markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$W_{\uparrow \downarrow}{r=1, 1}, U=3$')
plt.xlabel(r'$\beta/t$')
plt.legend(ncol=1, fontsize=12, loc='best', fancybox=True, framealpha=0)
plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=4))
plt.subplots_adjust(left=0.15,
                    bottom=0.15,
                    right=0.95,
                    top=0.9,
                    wspace=0.3,
                    hspace=0)
plt.savefig(f'/Users/mariagazizova/work/graph_new_after_sign_problem/sum_W_ft_beta_1.png', dpi=300, bbox_inches='tight')
plt.show()

#
# plt.figure(figsize=(14, 6))
# plt.subplot(1, 3, 1)
# plt.errorbar(beta, W_00_uu_beta_trunc, yerr= abs(W_00_uu_beta_err_trunc), markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$W_{\uparrow \uparrow}{r=0}, U=2$')
# plt.errorbar(beta, W_00_ud_beta_trunc, yerr= abs(W_00_ud_beta_err_trunc), markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$W_{\uparrow \downarrow}{r=0}, U=2$')
# plt.errorbar(beta, W_00_uu_beta_1_trunc, yerr= abs(W_00_uu_beta_err_1_trunc), markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$W_{\uparrow \uparrow}{r=0}, U=3$')
# plt.errorbar(beta, W_00_ud_beta_1_trunc, yerr= abs(W_00_ud_beta_err_1_trunc), markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$W_{\uparrow \downarrow}{r=0}, U=3$')
# plt.ylabel(r'$W(r, \tau = 0) / U$')
# plt.legend(ncol=1, fontsize=12, loc='best', fancybox=True, framealpha=0)
# plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=4))
# plt.xlabel(r'$\beta/t$')
# plt.subplot(1, 3, 2)
# plt.ylim(-0.01, 0.14)
# plt.errorbar(beta, W_01_uu_beta_trunc, yerr= abs(W_01_uu_beta_err_trunc), markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$W_{\uparrow \uparrow}{r=0, 1}, U=2$')
# plt.errorbar(beta, W_01_ud_beta_trunc, yerr= abs(W_01_ud_beta_err_trunc), markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$W_{\uparrow \downarrow}{r=0, 1}, U=2$')
# plt.errorbar(beta, W_01_uu_beta_1_trunc, yerr= abs(W_01_uu_beta_err_1_trunc), markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$W_{\uparrow \uparrow}{r=0, 1}, U=3$')
# plt.errorbar(beta, W_01_ud_beta_1_trunc, yerr= abs(W_01_ud_beta_err_1_trunc), markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$W_{\uparrow \downarrow}{r=0, 1}, U=3$')
# plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=4))
# plt.legend(ncol=1, fontsize=12, loc='best', fancybox=True, framealpha=0)
# plt.xlabel(r'$\beta/t$')
# plt.subplot(1, 3, 3)
# plt.ylim(0, 0.02)
# plt.errorbar(beta, W_11_uu_beta_trunc, yerr= abs(W_11_uu_beta_err_trunc), markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$W_{\uparrow \uparrow}{r=1, 1}, U=2$')
# plt.errorbar(beta, W_11_ud_beta_trunc, yerr=abs(W_11_ud_beta_err_trunc), markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$W_{\uparrow \downarrow}{r=1, 1}, U=2$')
# plt.errorbar(beta, W_11_uu_beta_1_trunc, yerr= abs(W_11_uu_beta_err_1_trunc), markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$W_{\uparrow \uparrow}{r=1, 1}, U=3$')
# plt.errorbar(beta, W_11_ud_beta_1_trunc, yerr=abs(W_11_ud_beta_err_1_trunc), markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$W_{\uparrow \downarrow}{r=1, 1}, U=3$')
# plt.xlabel(r'$\beta/t$')
# plt.legend(ncol=1, fontsize=12, loc='best', fancybox=True, framealpha=0)
# plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=4))
# plt.subplots_adjust(left=0.15,
#                     bottom=0.15,
#                     right=0.95,
#                     top=0.9,
#                     wspace=0.3,
#                     hspace=0)
# plt.savefig(f'/Users/mariagazizova/work/graph_new_after_sign_problem/sum_W_ft_beta_1.png', dpi=300, bbox_inches='tight')
# plt.show()
