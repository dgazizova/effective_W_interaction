from sys import exit

import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from scipy.optimize import curve_fit

from data_processing import (W_calculation_up_down, W_calculation_upup, cut, extract_data_new, fourier_transform,
                             generate_chi, generate_fitting_params, generate_fitting_params_curved, generate_omega_sum,
                             generate_summed_w_c, linear_fit)

axis_font = {'fontname':'Arial', 'size':'15'}

BASIC_PATH = '/Users/mariagazizova/work/test_sum_omega/'
odd_orders = [0, 2, 3]
even_orders = [0, 2, 3]
W_upup = dict()
W_upup_err = dict()
U = float(input('U:'))

beta = [1, 2, 3, 4, 5, 6, 7, 8.33]
# beta = [5]
# 12*12 grid
grid_size = 169
omega = 15


def extract_chi_for_difbeta(beta, odd):
    int_beta = int(beta)
    if odd:
        chi_12_12, chi_12_12_err = generate_chi(odd_orders, BASIC_PATH + f'omega_q/beta{int_beta}/odd', U, 3)
        orders = odd_orders
    else:
        chi_12_12, chi_12_12_err = generate_chi(even_orders, BASIC_PATH + f'omega_q/beta{int_beta}/even', U, 3)
        orders = even_orders
    chi_3 = dict()
    chi_3_err = dict()
    for i in orders:
        chi_3[i], chi_3_err[i] = generate_omega_sum(chi_12_12[i], chi_12_12_err[i], grid_size)
        chi_3[i] = cut(chi_3[i])
        chi_3_err[i] = cut(chi_3_err[i])
    return chi_3, chi_3_err


def extract_W_for_difbeta(beta, spin):
    int_beta = int(beta)
    chi_odd_12_12, chi_odd_12_12_err = generate_chi(odd_orders, BASIC_PATH + f'omega_q/beta{int_beta}/odd', U, 3)
    chi_even_12_12, chi_even_12_12_err = generate_chi(even_orders, BASIC_PATH + f'omega_q/beta{int_beta}/even', U, 3)
    W_12_12 = dict()
    W_12_12_err = dict()
    if spin:
        W_12_12[3], W_12_12_err[3] = W_calculation_upup(chi_odd_12_12[3], chi_odd_12_12_err[3],
                                                           chi_even_12_12[3], chi_even_12_12_err[3], U=U)
    else:
        W_12_12[3], W_12_12_err[3] = W_calculation_up_down(chi_odd_12_12[3], chi_odd_12_12_err[3],
                                                                         chi_even_12_12[3], chi_even_12_12_err[3], U=U)
        W_12_12[3] = W_12_12[3] - 1
    #sum using fittings

    W_12_12_w_c_summed = generate_summed_w_c(W_12_12[3], grid_size, omega)
    W_12_12_w_c_summed_err = generate_summed_w_c(W_12_12_err[3], grid_size, omega)

    W_12_12_fitting_par, W_12_12_fitting_par_err = generate_fitting_params(W_12_12_w_c_summed, grid_size, omega)
    # print(len(W_12_12_fitting_par_err))
    W_12_12_fitting_par_err = W_12_12_fitting_par_err + W_12_12_w_c_summed_err[:, 0]
    W_12_12_fitting_par_cut = cut(W_12_12_fitting_par) / beta
    W_12_12_fitting_par_err_cut = cut(W_12_12_fitting_par_err) / beta

    # W_12_12_w_c_summed, W_12_12_w_c_summed_err = generate_omega_sum(W_12_12[3], W_12_12_err[3], grid_size)
    # W_12_12_w_c_summed = cut(W_12_12_w_c_summed) / beta
    # W_12_12_w_c_summed_err = cut(W_12_12_w_c_summed_err) / beta

    return W_12_12_fitting_par_cut, W_12_12_fitting_par_err_cut
    # return W_12_12_w_c_summed, W_12_12_w_c_summed_err


# #exampe for frequency
#     chi_odd_12_12, chi_odd_12_12_err = generate_chi_odd(odd_orders, BASIC_PATH + f'omega_q/beta5/odd', U, 3)
#     chi_even_12_12, chi_even_12_12_err = generate_chi_odd(even_orders, BASIC_PATH + f'omega_q/beta5/even', U, 3)
#     W_12_12 = dict()
#     W_12_12_err = dict()
#     W_12_12[3], W_12_12_err[3] = W_calculation_up_down(chi_odd_12_12[3], chi_odd_12_12_err[3],
#                                                        chi_even_12_12[3], chi_even_12_12_err[3], U=U)
#     W_12_12[3] = W_12_12[3] + 1
#     W_12_12_summed = generate_omega_sum(W_12_12[3], W_12_12_err[3], grid_size, build_plot=True)




chi_odd, chi_odd_err, chi_even, chi_even_err = [0 for i in range(len(beta))], [0 for i in range(len(beta))],\
                                               [0 for i in range(len(beta))], [0 for i in range(len(beta))]
W_upup, W_upup_err, W_updown, W_updown_err = [0 for i in range(len(beta))], [0 for i in range(len(beta))], \
                                             [0 for i in range(len(beta))], [0 for i in range(len(beta))]
for b, bb in enumerate(beta):
    chi_odd[b], chi_odd_err[b] = extract_chi_for_difbeta(bb, odd=True)
    chi_even[b], chi_even_err[b] = extract_chi_for_difbeta(bb, odd=False)
    W_upup[b], W_upup_err[b] = extract_W_for_difbeta(bb, spin=True)
    W_updown[b], W_updown_err[b] = extract_W_for_difbeta(bb, spin=False)
    W_updown[b] = W_updown[b] + 1


#trancation for uu, not using resummation scheme
uu_orders_trancation = [0, 2, 3, 4]
ud_orders_trancation = [1, 2, 3, 4]

def extract_trunc_for_beta(beta, spin):
    int_beta = int(beta)
    if spin:
        W_4, W_err_4 = generate_chi(uu_orders_trancation, BASIC_PATH + f'sum_omega_trancation/beta{int_beta}/uu', U=U, o=4)
    else:
        W_4, W_err_4 = generate_chi(ud_orders_trancation, BASIC_PATH + f'sum_omega_trancation/beta{int_beta}/ud', U=U, o=4)
    for keys in W_4:
        W_4[keys] = W_4[keys] * U
        W_err_4[keys] = W_err_4[keys] * U
    W_cut_4 = (-1) * cut(W_4[4])
    W_cut_err_4 = (-1) * cut(W_err_4[4])
    return W_cut_4, W_cut_err_4

beta1 = [1, 2, 3, 4, 5, 6, 7, 8.33]
W_uu_4, W_uu_err_4 = [0 for i in range(len(beta1))], [0 for i in range(len(beta1))]
W_ud_4, W_ud_err_4 = [0 for i in range(len(beta1))], [0 for i in range(len(beta1))]
for b, bb in enumerate(beta1):
    W_uu_4[b], W_uu_err_4[b] = extract_trunc_for_beta(bb, spin=True)
    W_ud_4[b], W_ud_err_4[b] = extract_trunc_for_beta(bb, spin=False)
    W_ud_4[b] = W_ud_4[b] + 1


gamma_point = [0, 9, 18, 27]
plt.subplot(2, 1, 1)
plt.title(f'U={U}')
plt.ylim(0, 0.6)
plt.xticks(gamma_point, ['(0,0)', '(0, $\pi$)', '($\pi$,$\pi$)', '(0,0)'])
for b, bb in enumerate(beta):
    plt.errorbar(np.arange(0, 28-0.3, 27/36), chi_odd[b][3] / bb, yerr=abs(chi_odd_err[b][3] / bb), markersize=5, capsize=2, capthick=2, fmt='.-', label=fr'$\sum_{-15}^{15} \chi^{(3)}, \beta={bb}$')
plt.grid()
plt.legend()
plt.subplot(2, 1, 2)
plt.xticks(gamma_point, ['(0,0)', '(0, $\pi$)', '($\pi$,$\pi$)', '(0,0)'])
for b, bb in enumerate(beta):
    plt.errorbar(np.arange(0, 28-0.3, 27/36), chi_even[b][3] / bb, yerr=abs(chi_even_err[b][3] / bb), markersize=5, capsize=2, capthick=2,
             fmt='.-', label=fr'$\sum_{-15}^{15} \chi^{(3)}(12*12)$, \beta = {bb}')
plt.xticks(gamma_point, ['(0,0)', '(0, $\pi$)', '($\pi$,$\pi$)', '(0,0)'])
plt.grid()
plt.legend()
plt.show()



# # rpa
a_bubble, a_err = extract_data_new(BASIC_PATH + 'omega_q/beta5/odd/o3/a0.dat')
W_rpa = 1 / (1 + U * a_bubble)
W_rpa = W_rpa - 1

# this should be same as trunc (remember sum(x * x) not same as sum(x) * sum(x) )
# term_bubble_U_1 = U * a_bubble
# W_rpa = -term_bubble_U_1

W_rpa = generate_summed_w_c(W_rpa, grid_size, omega)
W_rpa, W_rpa_err = generate_fitting_params(W_rpa, grid_size, omega)
W_rpa = cut(W_rpa) / 5 + 1
W_rpa_err = cut(W_rpa_err) / 5


#rpa trunc
a_bubble_tr, a_err_tr = extract_data_new(BASIC_PATH + 'sum_omega_trancation/beta5/uu/o4/a0.dat')
term_bubble_U = U * a_bubble_tr
W_rpa_tr = 1 - term_bubble_U
W_rpa_tr_err = U * a_err_tr
W_rpa_tr = cut(W_rpa_tr)
W_rpa_tr_err = cut(W_rpa_tr_err)








plt.rc('font', size=14)
plt.subplot(2, 1, 1)
# plt.title(f'U={U}')
plt.xlim(0, 27)
plt.ylim(-1.4, -0)
# plt.ylabel(r'$W(Q,\tau = 0) / U$')
plt.ylabel(r'$W_{\uparrow \uparrow}(Q, \tau = 0) / U$', **axis_font)
#beta 5
beta_ind = 6
beta_ind_1 = 3
# plt.errorbar(np.arange(0, 28-0.3, 27/36), W_upup[beta_ind], yerr=abs(W_upup_err[beta_ind]),
#              markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$W_{\uparrow \uparrow}^{(3, 3)} (Q, \tau = 0) / U,  RS$')

plt.errorbar(np.arange(0, 28-0.3, 27/36), W_upup[0], yerr=abs(W_upup_err[0]),
             markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$\beta t = 1$')
plt.errorbar(np.arange(0, 28-0.3, 27/36), W_upup[1], yerr=abs(W_upup_err[1]),
             markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$\beta t = 2$')
# plt.errorbar(np.arange(0, 28-0.3, 27/36), W_upup[3], yerr=abs(W_upup_err[3]),
#              markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$\beta = 4$')
plt.errorbar(np.arange(0, 28-0.3, 27/36), W_upup[4], yerr=abs(W_upup_err[4]),
             markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$\beta t = 5$')
plt.errorbar(np.arange(0, 28-0.3, 27/36), W_upup[7], yerr=abs(W_upup_err[7]),
             markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$\beta t = 8.33$')

# plt.errorbar(np.arange(0, 28-0.3, 27/36), W_uu_4[beta_ind_1], yerr=abs(W_uu_err_4[beta_ind_1]), markersize=5, capsize=2, capthick=2, fmt='.-',
#              label=r'$W_{\uparrow \uparrow}^{(4)} (Q, \tau = 0)/ U, trunc$')
plt.xticks([])
plt.legend(loc=3, fancybox=False, framealpha=0.1, fontsize=13)
plt.subplot(2, 1, 2)
plt.xlim(0, 27)
plt.ylim(1, 1.4)
plt.ylabel(r'$W_{\uparrow \downarrow}(Q, \tau = 0) / U $', **axis_font)
# plt.errorbar(np.arange(0, 28-0.3, 27/36), W_updown[beta_ind], yerr=abs(W_updown_err[beta_ind]),
#              markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$W_{\uparrow \downarrow}^{(3, 3)} (Q, \tau = 0)/ U, RS$')

plt.errorbar(np.arange(0, 28-0.3, 27/36), W_updown[0] , yerr=abs(W_updown_err[0]),
             markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$\beta t = 1$')
plt.errorbar(np.arange(0, 28-0.3, 27/36), W_updown[1] , yerr=abs(W_updown_err[1]),
             markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$\beta t= 2$')
# plt.errorbar(np.arange(0, 28-0.3, 27/36), W_updown[3] , yerr=abs(W_updown_err[3]),
#              markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$\beta = 4$')
plt.errorbar(np.arange(0, 28-0.3, 27/36), W_updown[4], yerr=abs(W_updown_err[4]),
             markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$\beta t = 5$')
plt.errorbar(np.arange(0, 28-0.3, 27/36), W_updown[7], yerr=abs(W_updown_err[7]),
             markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$\beta t = 8.33$')

# plt.errorbar(np.arange(0, 28-0.3, 27/36), W_ud_4[beta_ind_1], yerr=abs(W_ud_err_4[beta_ind_1]), markersize=5, capsize=2, capthick=2, fmt='.-',
#              label=r'$W_{\uparrow \downarrow}^{(4)}(Q, \tau = 0)/ U, trunc$')
plt.xticks(gamma_point, ['(0,0)', '(0, $\pi$)', '($\pi$,$\pi$)', '(0,0)'])
plt.xlabel('Q',**axis_font)
# plt.grid()
plt.subplots_adjust(left=0.15,
                    bottom=0.15,
                    right=0.9,
                    top=0.9,
                    wspace=0.2,
                    hspace=0)
# plt.legend()
plt.legend(prop={'family': 'Arial', 'size': 12}, loc=2, fancybox=False, framealpha=0.1, fontsize=13)
plt.savefig(f'/Users/mariagazizova/work/graph_new_after_sign_problem/sum_W_U{U}_diffbeta.png', dpi=300)
plt.show()


plt.rc('font', size=14)
plt.subplot(2, 1, 1)
# plt.title(f'U={U}')
plt.xlim(0, 27)
plt.ylim(-1.2, -0)
# plt.ylabel(r'$W(Q,\tau = 0) / U$')
plt.ylabel(r'$W_{\uparrow \uparrow}(Q, \tau = 0) / U$', **axis_font)
#beta 5
beta_ind = 4
beta_ind_1 = 4
plt.errorbar(np.arange(0, 28-0.3, 27/36), W_upup[beta_ind], yerr=abs(W_upup_err[beta_ind]),
             markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$W_{\uparrow \uparrow}^{(3, 3)} (Q, \tau = 0) / U,  RS$')

plt.errorbar(np.arange(0, 28-0.3, 27/36), W_uu_4[beta_ind_1], yerr=abs(W_uu_err_4[beta_ind_1]), markersize=5, capsize=2, capthick=2, fmt='.-',
             label=r'$W_{\uparrow \uparrow}^{(4)} (Q, \tau = 0)/ U, trunc$')
plt.xticks([])
plt.legend(loc=3, fancybox=False, framealpha=0.1, fontsize=13)
plt.subplot(2, 1, 2)
plt.xlim(0, 27)
plt.ylim(1, 1.45)
plt.ylabel(r'$W_{\uparrow \downarrow}(Q, \tau = 0) / U $', **axis_font)
plt.errorbar(np.arange(0, 28-0.3, 27/36), W_updown[beta_ind], yerr=abs(W_updown_err[beta_ind]),
             markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$W_{\uparrow \downarrow}^{(3, 3)} (Q, \tau = 0)/ U, RS$')

plt.errorbar(np.arange(0, 28-0.3, 27/36), W_ud_4[beta_ind_1], yerr=abs(W_ud_err_4[beta_ind_1]), markersize=5, capsize=2, capthick=2, fmt='.-',
             label=r'$W_{\uparrow \downarrow}^{(4)}(Q, \tau = 0)/ U, trunc$')
plt.xticks(gamma_point, ['(0,0)', '(0, $\pi$)', '($\pi$,$\pi$)', '(0,0)'])
plt.xlabel('Q', **axis_font)
# plt.grid()
plt.subplots_adjust(left=0.15,
                    bottom=0.15,
                    right=0.9,
                    top=0.9,
                    wspace=0.2,
                    hspace=0)
# plt.legend()
plt.legend(prop={'family': 'Arial', 'size': 12}, loc=2, fancybox=False, framealpha=0.1, fontsize=13)
plt.savefig(f'/Users/mariagazizova/work/graph_new_after_sign_problem/sum_W_U{U}_beta5_RS_trunc.png', dpi=300)
plt.show()




# for graph in latex
plt.rc('font', size=13)
# plt.figure(figsize=(8, 6))
# plt.title(f'U={U}')
plt.xlim(0, 27)
plt.ylim(-1.5, 1.5)
plt.ylabel(r'$W(Q,\tau = 0) / U$', **axis_font)
#beta 4
beta_ind = 5
plt.errorbar(np.arange(0, 28-0.3, 27/36), W_upup[beta_ind], yerr=abs(W_upup_err[beta_ind]),
             markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$W_{\uparrow \uparrow}^{(3, 3)}$')
# plt.errorbar(np.arange(0, 28-0.3, 27/36), W_uu_4[beta_ind], yerr=abs(W_uu_err_4[beta_ind]), markersize=5, capsize=2, capthick=2, fmt='.-',
#              label=r'$W_{\uparrow \uparrow}^{(4)}$')
plt.errorbar(np.arange(0, 28-0.3, 27/36), W_updown[beta_ind] , yerr=abs(W_updown_err[beta_ind]),
             markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$W_{\uparrow \downarrow}^{(3, 3)}$')
# plt.errorbar(np.arange(0, 28-0.3, 27/36), W_ud_4[beta_ind] , yerr=abs(W_ud_err_4[beta_ind]), markersize=5, capsize=2, capthick=2, fmt='.-',
#              label=r'$W_{\uparrow \downarrow}^{(4)}$')
# plt.errorbar(np.arange(0, 28-0.3, 27/36), W_ud_4[beta_ind] + W_uu_4[beta_ind], yerr=abs(W_ud_err_4[beta_ind] + W_uu_err_4[beta_ind]), markersize=5, capsize=2, capthick=2, fmt='.-',
#              label=r'$W^{(4)}$')


plt.errorbar(np.arange(0, 28-0.3, 27/36), W_upup[3] + W_updown[3], yerr=abs(W_upup_err[3] + W_updown_err[3]), markersize=5, capsize=2, capthick=2, fmt='.-',
             label=r'$W^{(3,3)}$')
# plt.errorbar(np.arange(0, 28-0.3, d27/36), W_upup[3] + W_updown[3] - 1, yerr=abs(W_upup_err[3]),
#              markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$W_{\uparrow \uparrow}^{(3, 3)}, RS$')
plt.errorbar(np.arange(0, 28-0.3, 27/36), W_rpa, yerr=abs(W_rpa_err), markersize=5, capsize=2, capthick=2, fmt='.-',
             label=r'$W_{RPA}$')

# plt.errorbar(np.arange(0, 28-0.3, 27/36), W_rpa_tr, yerr=abs(W_rpa_tr_err), markersize=5, capsize=2, capthick=2, fmt='.-',
#              label=r'$W_{RPA_trunc}$')

plt.xticks(gamma_point, ['(0,0)', '(0, $\pi$)', '($\pi$,$\pi$)', '(0,0)'])
plt.xlabel('Q', **axis_font)
# plt.grid()
plt.subplots_adjust(left=0.15,
                    bottom=0.15,
                    right=0.9,
                    top=0.9,
                    wspace=0.2,
                    hspace=0)
plt.legend(prop={'family': 'Arial', 'size': 12}, loc=3, fancybox=False, framealpha=0.1, fontsize=13)
U = int(U)
plt.savefig(f'/Users/mariagazizova/work/graph_new_after_sign_problem/sum_W_U_RPA.png', dpi=300)
plt.show()





