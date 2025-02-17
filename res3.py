import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import numpy as np
import awkward as ak
import proj

ptedges = np.array([0, 150, 300, 500, 1000, np.inf])

r_edges = np.linspace(0, 1, 21)
phi_edges = np.linspace(0, np.pi/2, 15)

area = 0.5*(np.square(r_edges[1:,None]) - np.square(r_edges[:-1,None]))*(phi_edges[None,1:] - phi_edges[None,:-1])

def compute_profiles(data, btag, ptbin, RLbin, iToy):
    val = data[0, iToy, btag, ptbin, RLbin, 1:-1, 1:-1]

    nr, nphi = val.shape

    flux = val/area

    angular_avg = np.mean(flux, axis=1)
    modulation = flux/angular_avg[:,None]
    modulation = modulation/np.mean(modulation)

    return angular_avg, modulation

def compute_profile_errs(data, btag, ptbin, RLbin):
    ntoys = data.shape[1]

    nominal_avg, nominal_modulation = compute_profiles(data, btag, ptbin, RLbin, 0)

    diff_avg = []
    diff_modulation = []

    for itoy in range(1, ntoys):
        a, b = compute_profiles(data, btag, ptbin, RLbin, itoy)
        diff_avg.append(a-nominal_avg)
        diff_modulation.append(b-nominal_modulation)

    diff_avg = np.array(diff_avg)
    diff_modulation = np.array(diff_modulation)

    mean_diff_avg = np.mean(diff_avg, axis=0)
    mean_diff_modulation = np.mean(diff_modulation, axis=0)

    std_diff_avg = np.std(diff_avg, axis=0)
    std_diff_modulation = np.std(diff_modulation, axis=0)

    return nominal_avg, nominal_modulation, std_diff_avg, std_diff_modulation

def plot_profiles(data, btag, ptbin, RLbin, title):
    nominal_avg, nominal_modulation, std_diff_avg, std_diff_modulation = compute_profile_errs(data, btag, ptbin, RLbin)

    nr, nphi = nominal_modulation.shape

    fig, (ax0, ax1) = plt.subplots(2,1)

    r_centers = (r_edges[1:] + r_edges[:-1])/2
    phi_centers = (phi_edges[1:] + phi_edges[:-1])/2

    r_widths = r_edges[1:] - r_edges[:-1]
    phi_widths = phi_edges[1:] - phi_edges[:-1]

    ax0.errorbar(r_centers, nominal_avg, 
                 yerr=std_diff_avg,
                 xerr=r_widths/2,
                 fmt='o')
    ax0.set_xlabel(r'$r$')
    ax0.set_ylabel("Radial profile")

    for r in [0, 2, 4]:
        ax1.errorbar(phi_centers[1:], nominal_modulation[r,1:], 
                     yerr=std_diff_modulation[r,1:], 
                     xerr=phi_widths[1:]/2, 
                     fmt='o',
                     label=r'$%0.2f < r < %0.2f$' % (r_edges[r], r_edges[r+1]))
    ax1.set_xlabel(r'$\phi$')
    ax1.set_ylabel("Angular profile")
    ax1.legend()

    plt.suptitle(title)
    plt.show()

def plot_res3(data, btag, ptbin, RLbin, modulation=False):
    val = data[0, 0, btag, ptbin, RLbin, 1:-1, 1:-1]

    nr, nphi = val.shape

    flux = val/area 

    if modulation:
        angular_avg = np.mean(flux, axis=1)[:,None]
        flux = flux/angular_avg

    flux = flux/np.mean(flux) 

    fig, ax = plt.subplots()
    norm = LogNorm()

    c = ax.pcolormesh(phi_edges, r_edges, flux, norm=norm)
    ax.set_xlabel(r'$\phi$')
    ax.set_ylabel(r'$r$')
    fig.colorbar(c, ax=ax)
    plt.show()
