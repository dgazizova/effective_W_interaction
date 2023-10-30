import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from matplotlib.ticker import LogFormatter

axis_font = {'fontname': 'Arial', 'size': '18', 'weight': 'bold'}

formatter = LogFormatter(10)


def build_contour_plot_beta_U(filename: str):
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)

    data_ud = np.loadtxt(filename)

    beta = np.asarray([1, 2, 3, 4, 5, 6, 7, 8.33])
    T = 1 / beta
    U_list = np.arange(0, 4.2, 0.2)

    UlistM, TM = np.meshgrid(U_list, T)
    UlistM, betaM = np.meshgrid(U_list, beta)

    data_ud = np.reshape(data_ud, (len(beta), len(U_list)))

    up_ud = data_ud.max() / 1

    fig, (ax1) = plt.subplots(nrows=1, ncols=1)
    im1 = ax1.pcolormesh(UlistM, betaM, data_ud, cmap='bwr', vmax=up_ud, vmin=-up_ud)
    cb1 = fig.colorbar(im1, ax=ax1, orientation="vertical", pad=0.05, )
    tick_locator = ticker.MaxNLocator(nbins=5)
    cb1.locator = tick_locator
    cb1.update_ticks()
    ax1.set_xlabel(r'$U/t$', **axis_font)
    ax1.text(0.7, 8, r'$W_{\uparrow\downarrow}^{(3,3)}/U$', **axis_font)
    ax1.text(0.7, 7, r'$r = (0, 1)$', **axis_font)
    ax1.set_ylabel(r'$\beta t$', **axis_font)
    plt.show()
    plt.close()


build_contour_plot_beta_U('data_W_ud_01.txt')
