import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib import ticker
axis_font = {'fontname':'Arial', 'size':'18', 'weight':'bold'}


from matplotlib.ticker import LogFormatter 
formatter = LogFormatter(10)
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)

# a ='data_W_uu.txt'
# b ='data_W_ud.txt'
a = 'data_W_uu_01.txt'
b = 'data_W_ud_01.txt'
# a = 'data_W_trunc_uu_01.txt'
# b = 'data_W_trunc_ud_01.txt'
# a = 'data_W_full_uu_01.txt'
# b = 'data_W_full_ud_01.txt'
# a = 'data_W_uu_11.txt'
# b = 'data_W_ud_11.txt'


data_uu = np.loadtxt(a)
data_ud = np.loadtxt(b)


beta = np.asarray([1, 2, 3, 4, 5, 6, 7, 8.33])
T = 1 / beta
U_list = np.arange(0, 4.2, 0.2)

UlistM, TM = np.meshgrid(U_list, T)
UlistM, betaM = np.meshgrid(U_list, beta)


print(betaM)
print(TM)
print(UlistM)

data_ud = np.reshape(data_ud, (len(beta), len(U_list)))
data_uu = np.reshape(data_uu, (len(beta), len(U_list)))

print(data_uu[:, 0])
print(data_uu[0, :])


up_ud = data_ud.max()/1
up_uu = data_uu.max()/3

fig, (ax1) = plt.subplots(nrows=1, ncols=1)
###first plot
im1 = ax1.pcolormesh(UlistM, betaM, data_ud, cmap='bwr', vmax=up_ud, vmin=-up_ud)
# im1 = ax1.pcolormesh(UlistM, TM, data_ud, cmap='bwr', vmax=up_ud, vmin=-up_ud)
# im1 = ax1.contourf(UlistM, betaM, data_ud, 50, cmap='bwr', vmax=up, vmin=-up)
cb1 = fig.colorbar(im1, ax=ax1, orientation="vertical", pad=0.05, )
tick_locator = ticker.MaxNLocator(nbins=5)
# cb1.formatter.set_powerlimits((0, 0))
cb1.locator = tick_locator
cb1.update_ticks()
#########second plot

# im2 = ax2.pcolormesh(UlistM, TM, data_uu, cmap='bwr', vmax=up_uu, vmin=-up_uu)
# # ax2.set_xlabel(r'$\mathcal{\mathbf{q}}=[q_{x},q_{y}]$', **axis_font)
# cb2 = fig.colorbar(im2, ax=ax2, orientation="horizontal", pad=0.137)
# tick_locator = ticker.MaxNLocator(nbins=4)
# cb2.formatter.set_powerlimits((0, 0))
# cb2.locator = tick_locator
# cb2.update_ticks()
####################labels
# ax1.set_title(b, axis_font)
# ax2.set_title(a, axis_font)
ax1.set_xlabel(r'$U/t$', **axis_font)
ax1.text(0.7, 8, r'$W_{\uparrow\downarrow}^{(3,3)}/U$', **axis_font)
# ax1.text(0.7, 8, r'$W_{\uparrow\downarrow}^{(4)}/U$', **axis_font)
ax1.text(0.7, 7, r'$r = (0, 1)$', **axis_font)
ax1.set_ylabel(r'$\beta t$', **axis_font)
# ax2.set_xlabel(r'$U/t$', **axis_font)
# ax2.set_ylabel(r'$\beta$', **axis_font)
################################
# plt.savefig('myfil.pdf', dpi=2400, bbox_inches='tight')
plt.savefig(f'/Users/mariagazizova/work/graph_new_after_sign_problem/contour_r_01.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()