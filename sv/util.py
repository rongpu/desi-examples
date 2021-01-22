import numpy as np
# import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def compare_redshifts(z_x, z_y, zmin=0, zmax=1.2, markersize=1, alpha=0.2, show=True, 
    figsize=(8.18, 10.4), outlier_threshold=0.0033, dz_range=0.1,
    xlabel='$z_{\mathrm{1}}$', ylabel='$z_{\mathrm{best}}$', ylabel2='$\Delta z/(1+z_{\mathrm{truth}})$',
    red_outliers=True):
    '''
    Plot z_y vs z_x in the upper panel and (z_y-z_x)/(1+z_x) in the lower panel.
    '''
    nmad = 1.48 * np.median(np.abs((z_y - z_x)/(1 + z_x)))
    mask_outlier = np.abs((z_y - z_x)/(1 + z_x)) > outlier_threshold
    nout = np.sum(mask_outlier)
    eta = nout/len(z_x)
    print('Normalized MAD: {:.6f}'.format(nmad))
    print('{:.2f} outliers: {:.6f}%'.format(outlier_threshold, eta*100))

    plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    # x = np.linspace(zmin, zmax, 10)
    # outlier_upper = x + outlier_threshold*(1+x)
    # outlier_lower = x - outlier_threshold*(1+x)
    # ax0.plot(x, outlier_upper, 'k--', lw=1)
    # ax0.plot(x, outlier_lower, 'k--', lw=1)
    if not red_outliers:
        ax0.plot(z_x, z_y, 'b.', markersize=markersize, alpha=alpha)
    else:
        ax0.plot(z_x[~mask_outlier], z_y[~mask_outlier], 'b.', markersize=markersize, alpha=alpha)
        ax0.plot(z_x[mask_outlier], z_y[mask_outlier], 'r.', markersize=markersize, alpha=alpha)
    ax0.set_title('$\sigma_\mathrm{NMAD} \ = $%2.5f;  '%nmad+'$\Delta z>{:g}(1+z)$ outliers = {:2.1f}% ({})'.format(outlier_threshold, eta*100, nout), fontsize=15)
    ax0.set_xlim([zmin, zmax])
    ax0.set_ylim([zmin, zmax])
    ax0.set_xlabel(xlabel, fontsize=23)
    ax0.set_ylabel(ylabel, fontsize=23)
    ax0.yaxis.grid(alpha = 0.8, ls='--')
    ax0.xaxis.grid(alpha = 0.8)
    ax0.tick_params(labelsize=15)
    ax1.plot(z_x[~mask_outlier], ((z_y-z_x)/(1+z_x))[~mask_outlier], 'b.', markersize=markersize, alpha=alpha, label='')
    ax1.plot(z_x[mask_outlier], ((z_y-z_x)/(1+z_x))[mask_outlier], 'r.', markersize=markersize, alpha=alpha, label='')

    ax1.set_ylabel(ylabel2, fontsize=23)
    ax1.axis([zmin, zmax, -dz_range, dz_range])
    ax1.tick_params(axis='x', labelbottom=False)
    ax1.grid()
    plt.tight_layout()
    if show:
        plt.show()
    else:
        return ax0, ax1
