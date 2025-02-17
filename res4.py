import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import numpy as np
import awkward as ak
import proj
import util

RLedges = np.array([0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80])
ptedges = np.array([0, 200, 400, 800, 1600, np.inf])

redges_triangle = np.array([0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
                            0.60, 0.70, 0.80, 0.90, 1.0, 1.1,
                            1.3, 1.5, 1.7, 2.0])
ctedges_triangle = np.array([
    0.0       , 0.19634954, 0.39269908, 0.58904862, 0.78539816,
    0.9817477 , 1.17809725, 1.37444679, 1.57079633, 1.76714587,
    1.96349541, 2.15984495, 2.35619449, 2.55254403, 2.74889357,
    2.94524311, 3.14159265, 3.33794219, 3.53429174, 3.73064128,
    3.92699082, 4.12334036, 4.3196899 , 4.51603944, 4.71238898,
    4.90873852, 5.10508806, 5.3014376 , 5.49778714, 5.69413668,
    5.89048623, 6.08683577, 6.28318531
])

redges_teedipole = np.array([
    0.00, 0.05, 0.10, 0.15,
    0.20, 0.25, 0.30, 0.35,
    0.40, 0.45, 0.50, 0.55,
    0.60, 0.65, 0.70, 0.75,
    0.80, 0.85, 0.90, 0.95,
    1.00
])
ctedges_teedipole = np.array([
    0.0       , 0.11219974, 0.22439948, 
    0.33659921, 0.44879895, 0.56099869, 
    0.67319843, 0.78539816, 0.8975979 , 
    1.00979764, 1.12199738, 1.23419711, 
    1.34639685, 1.45859659, 1.57079634
])

def compute_profiles(data, btag, ptbin, RLbin, iToy):
    val = data[0, iToy, btag, ptbin, RLbin, 1:-1, 1:-1]
    nr, ntheta = val.shape

    r_edges = redges_teedipole
    theta_edges = ctedges_teedipole

    area = 0.5*(np.square(r_edges[1:,None]) - np.square(r_edges[:-1,None]))*(theta_edges[None,1:] - theta_edges[None,:-1])

    flux = val/area
    flux = flux/np.mean(flux)

    angular_avg = np.sum(flux, axis=1)/ntheta

    angular_fluctuation = flux/angular_avg[:,None]

    return angular_avg, angular_fluctuation

def compute_profiles_errs(data, btag, ptbin, RLbin):
    ntoys = data.shape[1]

    nominal_avg, nominal_fluctuation = compute_profiles(data, btag, ptbin, RLbin, 0)

    diff_avg = []
    diff_fluctuation = []

    for itoy in range(1, ntoys):
        a, b = compute_profiles(data, btag, ptbin, RLbin, itoy)
        diff_avg.append(a-nominal_avg)
        diff_fluctuation.append(b-nominal_fluctuation)

    diff_avg = np.array(diff_avg)
    diff_fluctuation = np.array(diff_fluctuation)

    mean_diff_avg = np.mean(diff_avg, axis=0)
    mean_diff_fluctuation = np.mean(diff_fluctuation, axis=0)

    std_diff_avg = np.std(diff_avg, axis=0)
    std_diff_fluctuation = np.std(diff_fluctuation, axis=0)

    return nominal_avg, nominal_fluctuation, std_diff_avg, std_diff_fluctuation

def plot_profiles(data, btag, ptbin, RLbin,
                  title,
                  fit_powerlaw=False,
                  show=True):
    nominal_avg, nominal_fluctuation, std_diff_avg, std_diff_fluctuation = compute_profiles_errs(data, btag, ptbin, RLbin)

    nr, ntheta = nominal_fluctuation.shape

    redges = redges_teedipole
    thetaedges = ctedges_teedipole

    rwidths = redges[1:] - redges[:-1]
    thetawidths = thetaedges[1:] - thetaedges[:-1]

    rmid = (redges[1:] + redges[:-1])/2
    thetamid = (thetaedges[1:] + thetaedges[:-1])/2

    fig, (ax0, ax1) = plt.subplots(2,1)

    fig.suptitle("%s\n%0.1f < RL < %0.1f\n%0.0f < pT < %0.0f" % (title,RLedges[RLbin-1], RLedges[RLbin], ptedges[ptbin], ptedges[ptbin+1]))

    ax0.errorbar(rmid,
                 nominal_avg,
                 xerr = rwidths/2,
                 yerr=std_diff_avg,
                 fmt='o')
    
    if fit_powerlaw:
        import scipy.optimize as opt
        def powerlaw(x, A, B):
            return A*x**B
        popt, pcov = opt.curve_fit(powerlaw, rmid[4:], 
                                   nominal_avg[4:])
        print(popt)
        ax0.plot(rmid, powerlaw(rmid, *popt), color='red')
        ax0.set_xscale('log')
        ax0.text(0.8, 0.8, 'slope = %0.2f' % popt[1],
                 fontsize=16, 
                 bbox=dict(facecolor='white', alpha=0.5),
                 transform=ax0.transAxes)

    ax0.set_yscale('log')
    ax0.set_xlabel('r')
    ax0.set_ylabel('Radial profile')
    ax0.set_xlim(0, 1)

    for r in [1,2,3]:
        ax1.errorbar(thetamid,
                     nominal_fluctuation[r], 
                     yerr=std_diff_fluctuation[r], 
                     xerr=thetawidths/2,
                     fmt='o', label=f'%0.2f < r < %0.2f' % (redges[r], redges[r+1]))
    ax1.set_xlabel(r'$\theta$')
    ax1.set_ylabel('Angular profile')
    ax1.axhline(1, color='black', linestyle='--')
    #legend inside of opaque box
    ax1.legend(loc='upper right', frameon=True, fontsize=10)
    ax1.set_xlim(0, np.pi/2)

    plt.tight_layout()

    if show:
        plt.show()

def plot_total_heatmap(data, btag, ptbin, RLbin,
                       title, triangle=False,
                       show=True):
    val = data[:,0]

    if triangle:
        theval = val[0, btag, ptbin, RLbin, 1:-1, 1:-1]
    else:
        theval = val[0, btag, ptbin, RLbin, 1:-1, 1:-1]

    nr, ntheta = theval.shape

    if triangle:
        r_edges = redges_triangle
        theta_edges = ctedges_triangle
    else:
        r_edges = redges_teedipole
        theta_edges = ctedges_teedipole

    r_mid = (r_edges[1:] + r_edges[:-1])/2
    theta_mid = (theta_edges[1:] + theta_edges[:-1])/2


    area = 0.5*(r_edges[1:,None]**2 - r_edges[:-1,None]**2)*(theta_edges[None,1:] - theta_edges[None,:-1])

    print(theval.sum())

    theval = theval/area

    print(theval.sum())

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    norm = LogNorm(vmin=theval[theval>0].min(), vmax=theval.max())
    #norm = Normalize(vmin=theval[theval>0].min(), vmax=theval.max())

    if triangle:
        pc1 = ax.pcolormesh(theta_edges, r_edges,
                            theval,
                            cmap='inferno',
                            shading='auto',
                            norm=norm)
    else:
        cmap = 'inferno'
        pc1 = ax.pcolormesh(theta_edges, r_edges,
                            theval,
                            cmap=cmap,
                            shading='auto',
                            norm=norm)
        pc2 = ax.pcolormesh(np.pi-theta_edges, r_edges,
                            theval,
                            cmap=cmap,
                            shading='auto',
                            norm=norm)
        pc3 = ax.pcolormesh(np.pi+theta_edges, r_edges,
                            theval,
                            cmap=cmap,
                            shading='auto',
                            norm=norm)
        pc4 = ax.pcolormesh(2*np.pi-theta_edges, r_edges,
                            theval,
                            cmap=cmap,
                            shading='auto',
                            norm=norm)
    plt.colorbar(pc1, ax=ax)

    fig.suptitle("%s\n%0.1f < RL < %0.1f\n%0.0f < pT < %0.0f" % (title,RLedges[RLbin-1], RLedges[RLbin], ptedges[ptbin], ptedges[ptbin+1]))

    plt.tight_layout()
    if show:
        plt.show()

def plot_modulation_heatmap(data, btag, ptbin, RLbin,
                       title, show=True):
    val = data[:,0]

    theval = val[0, btag, ptbin, RLbin, 1:-1, 1:-1]

    nr, ntheta = theval.shape

    r_edges = redges_teedipole
    theta_edges = ctedges_teedipole

    r_mid = (r_edges[1:] + r_edges[:-1])/2
    theta_mid = (theta_edges[1:] + theta_edges[:-1])/2

    area = 0.5*(r_edges[1:,None]**2 - r_edges[:-1,None]**2)*(theta_edges[None,1:] - theta_edges[None,:-1])

    theval = theval/area

    angular_avg = np.sum(theval, axis=1, keepdims=True)/ntheta
    theval = theval/angular_avg

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    norm = Normalize(vmin=theval[theval>0].min(), vmax=theval.max())

    cmap = 'inferno'
    pc1 = ax.pcolormesh(theta_edges, r_edges,
                        theval,
                        cmap=cmap,
                        shading='auto',
                        norm=norm)
    pc2 = ax.pcolormesh(np.pi-theta_edges, r_edges,
                        theval,
                        cmap=cmap,
                        shading='auto',
                        norm=norm)
    pc3 = ax.pcolormesh(np.pi+theta_edges, r_edges,
                        theval,
                        cmap=cmap,
                        shading='auto',
                        norm=norm)
    pc4 = ax.pcolormesh(2*np.pi-theta_edges, r_edges,
                        theval,
                        cmap=cmap,
                        shading='auto',
                        norm=norm)
    plt.colorbar(pc1, ax=ax)
    fig.suptitle("%s\n%0.1f < RL < %0.1f\n%0.0f < pT < %0.0f" % (title,RLedges[RLbin-1], RLedges[RLbin], ptedges[ptbin], ptedges[ptbin+1]))
    plt.tight_layout()

    if show:
        plt.show()
