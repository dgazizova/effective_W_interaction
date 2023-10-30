import numpy as np
import matplotlib.pyplot as plt
from data_processing import (extract_data_new, W_calculation_upup, W_calculation_up_down, generate_chi, generate_fitting_params, generate_omega_sum,
                             generate_summed_w_c, cut, fourier_transform, linear_fit, generate_fitting_params_curved)


BASIC_PATH = '/Users/mariagazizova/work/beta3'
# BASIC_PATH = '/Users/mariagazizova/work'

U = 1
gamma_point = [0, 32, 64, 96]
odd_orders = [0, 2, 3, 4]
even_orders = [2, 3, 4]
chi_odd = dict()
chi_odd_err = dict()
chi_even = dict()
chi_even_err = dict()
W = dict()
W_err = dict()
W_updown = dict()
W_updown_err = dict()

# chi_odd, chi_odd_err = generate_chi_odd(odd_orders, BASIC_PATH +'/odd_1', U, 4)
# chi_even, chi_even_err = generate_chi_odd(even_orders, BASIC_PATH +'/even_1', U, 4)

chi_odd, chi_odd_err = generate_chi(odd_orders, BASIC_PATH + '/odd', U, 4)
chi_even, chi_even_err = generate_chi(even_orders, BASIC_PATH + '/even', U, 4)

data = chi_odd[0]
np.savetxt("bubble.txt", data, fmt='%.9e')

for i in odd_orders:
    W[i], W_err[i] = W_calculation_upup(chi_odd[i], chi_odd_err[i], chi_even[4], chi_even_err[4], U, even=False)
    W_updown[i], W_updown_err[i] = W_calculation_up_down(chi_odd[i], chi_odd_err[i], chi_even[4], chi_even_err[4], U, even=False)
for i in odd_orders:
    for j in even_orders:
        W[(i, j)], W_err[(i, j)] = W_calculation_upup(chi_odd[i], chi_odd_err[i], chi_even[j], chi_even_err[j], U)
        W_updown[(i, j)], W_updown_err[(i, j)] = W_calculation_up_down(chi_odd[i], chi_odd_err[i], chi_even[j], chi_even_err[j], U)

# plot Chi
plt.subplot(2, 1, 1)
plt.title(f'U={U}')
plt.ylim(0, 0.8)
plt.xlim(0, 96)
plt.ylabel(r'$\Pi_o(Q, \Omega = 0)$')
x = range(len(chi_odd[0]))
for i in odd_orders:
    plt.errorbar(x, chi_odd[i], yerr=chi_odd_err[i], markersize=5, capsize=2, capthick=2, fmt='.', label=f'n = {i}')
plt.xticks(gamma_point, ['(0,0)', '(0, $\pi$)', '($\pi$,$\pi$)', '(0,0)'])
plt.grid()
plt.legend(loc=2)
plt.subplot(2, 1, 2)
plt.xlim(0, 96)
plt.ylabel('$\Pi_e$')
x = range(len(chi_odd[0]))
for i in even_orders:
    plt.errorbar(x, chi_even[i], yerr=chi_even_err[i], markersize=5, capsize=2, capthick=2, fmt='.', label=f'm = {i}')
plt.xticks(gamma_point, ['(0,0)', '(0, $\pi$)', '($\pi$,$\pi$)', '(0,0)'])
# plt.grid()
plt.legend(loc=2)
plt.tight_layout()
plt.show()



# plot Chi
# plt.ylim(0, 0.8)
plt.rc('font', size=14)
plt.subplot(2, 1, 1)
# plt.title(f'U={U}')

plt.xlim(0, 96)
plt.ylabel(r'$\Pi_o(Q, \Omega = 0)$')
x = range(len(chi_odd[0]))
plt.errorbar(x, chi_odd[0]*2, yerr=chi_odd_err[0], markersize=5, capsize=2, capthick=2, fmt='.-', label=r'$\Pi_o^{(0)}(Q, \Omega = 0)$')
plt.xticks(gamma_point, ['(0,0)', '(0, $\pi$)', '($\pi$,$\pi$)', '(0,0)'])
# plt.grid()
plt.legend(loc=2)
plt.subplot(2, 1, 2)
plt.xlim(0, 96)
plt.ylabel(r'$\Pi_e(Q, \Omega = 0)$')
x = range(len(chi_odd[0]))
plt.errorbar(x, chi_even[i], yerr=chi_even_err[i], markersize=5, capsize=2, color='orange' , capthick=2, fmt='.-', label=r'$\Pi_e^{(4)}(Q, \Omega = 0)$')
plt.xticks(gamma_point, ['(0,0)', '(0, $\pi$)', '($\pi$,$\pi$)', '(0,0)'])
# plt.grid()
plt.legend(loc=3)
plt.tight_layout()
# plt.savefig('/Users/mariagazizova/Downloads/Pi.png', dpi=300)
plt.show()

# # fix crazy points
# W[4, 3][2] = (W[4, 3][1] + W[4, 3][3]) / 2
# W[4, 4][2] = (W[4, 4][1] + W[4, 4][3]) / 2


# plot W
my_xticks = ['(0,0)', '(0, $\pi$)', '($\pi$,$\pi$)', '(0,0)']
# y_down = -10
# y_up = 10
y_down = -1
y_up = 5
plt.rc('font', size=14)
plt.subplot(2, 2, 1)
plt.ylabel(r'$W_{\uparrow \uparrow}^{(n,m)}(Q, \Omega = 0)/U$')
# plt.ylim(y_down, y_up)
# plt.ylim(-10, 10)
plt.xlim(0, 96)
for i in odd_orders:
    plt.errorbar(x, W[i], yerr=abs(W_err[i]), markersize=5, capsize=2, capthick=2, fmt='.-', label=f'W({i},-)')
# plt.errorbar(x, W[4], yerr=W_err[4], markersize=5, capsize=2, capthick=2, fmt='.-', label=f'W({4},-)')
# plt.errorbar(x, W[0], yerr= W_err[0], markersize=5, capsize=2, capthick=2, fmt='.-', label=f'W({0},-)')
plt.xticks([])
plt.legend(loc=2)
plt.subplot(2, 2, 2)
# plt.ylim(y_down, y_up)
# plt.ylim(-10, 10)
plt.xlim(0, 96)
for i in odd_orders:
    plt.errorbar(x, W[(i, 2)], yerr=abs(W_err[(i, 2)]), markersize=5, capsize=2, capthick=2, fmt='.-', label=f'W({i},2)')
# plt.errorbar(x, W[(4, 2)], yerr=W_err[(4, 2)], markersize=5, capsize=2, capthick=2, fmt='.-', label=f'W({4},2)')
plt.legend(loc=2)
plt.xticks([])
plt.yticks([])
plt.subplot(2, 2, 3)
plt.ylabel(r'$W_{\uparrow \uparrow}^{(n,m)}(Q, \Omega = 0)/U$')
# plt.ylim(0, 1.49)
# plt.ylim(y_down, y_up)
# plt.ylim(-1, 4)
plt.xlim(0, 96)
for i in odd_orders:
    plt.errorbar(x, W[(i, 3)], yerr=abs(W_err[(i, 3)]), markersize=5, capsize=2, capthick=2, fmt='.-', label=f'W({i},3)')
# plt.errorbar(x, W[(4, 3)], yerr=W_err[(4, 3)], markersize=5, capsize=2, capthick=2, fmt='.-', label=f'W({4},3)')
plt.legend(loc=2)
plt.xticks(gamma_point, my_xticks)
plt.subplot(2, 2, 4)
# plt.ylim(0, 1.5)
# plt.ylim(y_down, y_up)
# plt.ylim(-1, 4)
plt.xlim(0, 96)
for i in odd_orders:
    plt.errorbar(x, W[(i, 4)],  yerr=abs(W_err[(i, 4)]), markersize=5, capsize=2, capthick=2, fmt='.-', label=f'W({i},4)')
# plt.errorbar(x, W[(4, 4)], yerr=W_err[(4, 4)], markersize=5, capsize=2, capthick=2, fmt='.-', label=f'W({4},4)')
plt.legend(loc=2)
plt.xticks(gamma_point, my_xticks)
plt.yticks([])
plt.tight_layout()
plt.subplots_adjust(left=0.1,
          bottom=0.1,
          right=0.9,
          top=0.9,
          wspace=0,
          hspace=0)
plt.show()



y_down = 0
y_up = 5
plt.rc('font', size=14)
plt.subplot(2, 2, 1)
plt.ylabel(r'$W_{\uparrow \downarrow}^{(n,m)}(Q, \Omega = 0)/U$')
# plt.ylim(y_down, y_up)
# plt.ylim(-10, 10)
plt.xlim(0, 96)
for i in odd_orders:
    plt.errorbar(x, W_updown[i], yerr=abs(W_updown_err[i]), markersize=5, capsize=2, capthick=2, fmt='.-', label=f'W({i},-)')
plt.xticks([])
plt.legend(loc=2)
plt.subplot(2, 2, 2)
# plt.ylim(y_down, y_up)
# plt.ylim(-10, 10)
plt.xlim(0, 96)
for i in odd_orders:
    plt.errorbar(x, W_updown[(i, 2)], yerr=abs(W_updown_err[(i, 2)]), markersize=5, capsize=2, capthick=2, fmt='.-', label=f'W({i},2)')
plt.legend(loc=2)
plt.xticks([])
plt.yticks([])
plt.subplot(2, 2, 3)
plt.ylabel(r'$W_{\uparrow \downarrow}^{(n,m)}(Q, \Omega = 0)/U$')
# plt.ylim(0, 1.49)
# plt.ylim(y_down, y_up)
# plt.ylim(-1, 4)
plt.xlim(0, 96)
for i in odd_orders:
    plt.errorbar(x, W_updown[(i, 3)], yerr=abs(W_updown_err[(i, 3)]), markersize=5, capsize=2, capthick=2, fmt='.-', label=f'W({i},3)')
plt.legend(loc=2)
plt.xticks(gamma_point, my_xticks)
plt.subplot(2, 2, 4)
# plt.ylim(0, 1.5)
# plt.ylim(y_down, y_up)
# plt.ylim(-1, 4)
plt.xlim(0, 96)
for i in odd_orders:
    plt.errorbar(x, W_updown[(i, 4)],  yerr=abs(W_updown_err[(i, 4)]), markersize=5, capsize=2, capthick=2, fmt='.-', label=f'W({i},4)')
plt.legend(loc=2)
plt.xticks(gamma_point, my_xticks)
plt.yticks([])
plt.tight_layout()
plt.subplots_adjust(left=0.1,
          bottom=0.1,
          right=0.9,
          top=0.9,
          wspace=0,
          hspace=0)
plt.show()


plt.rc('font', size=14)
plt.rc('xtick',labelsize=14)
plt.rc('ytick',labelsize=12)
plt.subplot(2, 1, 1)
plt.ylabel(r'$W_{\uparrow \uparrow}(Q, \Omega = 0)/U$')
plt.errorbar(x, W[(4, 4)], yerr=abs(W_err[(4, 4)]), markersize=5, capsize=2, capthick=2, fmt='.-',
             label=r'$W_{\uparrow \uparrow}^{(4, 4)}(Q, \Omega = 0)/U$')
plt.errorbar(x, W[0], yerr=abs(W_err[0]), markersize=5, capsize=2, capthick=2, fmt='.-',
             label=r'$W_{\uparrow \uparrow}^{(0)}(Q, \Omega = 0)/U$')
plt.errorbar(x, - chi_odd[0] * U, yerr=abs(chi_odd_err[0]) * U, markersize=5, capsize=2, capthick=2, fmt='.-',
             label=r'$W_{\uparrow \downarrow}^{(0)}(Q, \Omega = 0)/U$')
plt.errorbar(x, - chi_odd[0] * U - chi_odd[0] * U * chi_odd[0] * U * chi_odd[0]
             - chi_odd[0]*U * chi_odd[0] * U *chi_odd[0] * U*chi_odd[0] * U*chi_odd[0]
             - chi_odd[0]*U *chi_odd[0] * U *chi_odd[0] * U*chi_odd[0] * U*chi_odd[0] * U*chi_odd[0] * U*chi_odd[0]
             - chi_odd[0]*U *chi_odd[0] * U *chi_odd[0] * U*chi_odd[0] * U*chi_odd[0] * U*chi_odd[0] * U*chi_odd[0] * U*chi_odd[0] * U*chi_odd[0], yerr=abs(chi_odd_err[0]) * U, markersize=5, capsize=2, capthick=2, fmt='.-',
             label=r'$W_{\uparrow \downarrow}^{(0)}(Q, \Omega = 0)/U$')
plt.errorbar(x, - chi_odd[0] * U / (1 - (U * chi_odd[0])**2), yerr=abs(chi_odd_err[0]) * U, markersize=5, capsize=2, capthick=2, fmt='.-',
             label=r'$W_{\uparrow \downarrow}^{(0)}(Q, \Omega = 0)/U$')
plt.errorbar(x, 1/(1 + chi_odd[0] * U), yerr=abs(chi_odd_err[0]) * U, markersize=5, capsize=2, capthick=2, fmt='.-',
             label=r'$W_{\uparrow \downarrow}^{(0)}(Q, \Omega = 0)/U$')
plt.errorbar(x, W[4] + W_updown[4], yerr=abs(W_err[0]), markersize=5, capsize=2, capthick=2, fmt='.-',
             label=r'$W_{\uparrow \uparrow}^{(0)}(Q, \Omega = 0)/U$')
plt.xticks([])
# plt.xticks(gamma_point, my_xticks)
plt.legend(loc=3, fancybox=False, framealpha=0.1, fontsize=13)
plt.subplot(2, 1, 2)
plt.ylabel(r'$W_{\uparrow \downarrow}(Q, \Omega = 0)/U$')
plt.errorbar(x, W_updown[(4, 4)], yerr=abs(W_updown_err[(4, 4)]), markersize=5, capsize=2, capthick=2, fmt='.-',
             label=r'$W_{\uparrow \downarrow}^{(4, 4)}(Q, \Omega = 0)/U$', color='orange')
plt.errorbar(x, W_updown[0], yerr=abs(W_updown_err[0]), markersize=5, capsize=2, capthick=2, fmt='.-',
             label=r'$W_{\uparrow \downarrow}^{(0)}(Q, \Omega = 0)/U$', color='orange')

plt.legend(loc=2, fancybox=False, framealpha=0.1, fontsize=13)
plt.xticks(gamma_point, my_xticks)
plt.tight_layout()
plt.subplots_adjust(left=0.2,
          bottom=0.1,
          right=0.9,
          top=0.9,
          wspace=0,
          hspace=0)
# plt.savefig('/Users/mariagazizova/Downloads/W.png', dpi=300)
plt.savefig('/Users/mariagazizova/work/graph_new_after_sign_problem/W_static.png', dpi=300, bbox_inches='tight')
plt.show()