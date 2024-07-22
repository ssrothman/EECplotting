import matplotlib.pyplot as plt
import os.path
import numpy as np
import mplhep as hep
from matplotlib.colors import Normalize, LogNorm

from util import setup_ratiopad_canvas, setup_plain_canvas, get_ax_edges, has_overflow, has_underflow, should_logx, should_logy, setup_cbar_canvas, get_ax_label, savefig
from EECutil import binning_AND, binning_name, binning_string, project_projected, project_cov1x2_projected, bin_transfer_projected, bin_cov_projected, project_covprojected
from stats import getratio, maybe_density, maybe_density_cross

def plotProjectedEECratio(val1, cov1,
                          val2, cov2,
                          cov1x2,
                          ax=None, isData=False,
                          wrt='dR', color=None):
    if ax is None:
        fig, ax = setup_plain_canvas(isData)
        ax.set_xlabel(get_ax_label(wrt, 'EECproj'))

    ratio, covratio = getratio(val1, val2, cov1, cov2, cov1x2)

    xedges = get_ax_edges(wrt)
    xcenters = (xedges[1:] + xedges[:-1]) / 2

    xerrs = (xedges[1:] - xedges[:-1]) / 2
    yerrs = np.sqrt(np.diag(covratio))

    ax.errorbar(xcenters, ratio,
                xerr = xerrs, yerr = yerrs,
                fmt='o', color=color)

    ax.axhline(1, color='black', linestyle='--')

def plotProjectedEEC(x, covx,
                     ax=None, isData=False,
                     wrt='dR', binning={},
                     logwidth=True,
                     density=True,
                     color = None,
                     label = None):
    if ax is None:
        fig, ax = setup_plain_canvas(isData)
        ax.set_xlabel(get_ax_label(wrt, 'EECproj'))

    ax.set_ylabel("EEC")

    if wrt != 'dR':
        if logwidth:
            print("Warning: logwidth is only sensible for dR. Setting to False")
        logwidth=False

    vals = project_projected(x, binning, onto=wrt)
    covs = project_covprojected(covx, binning, onto=wrt)

    vals, covs, N = maybe_density(vals, covs, density, return_N=True)

    xedges = get_ax_edges(wrt)
    xcenters = (xedges[1:] + xedges[:-1]) / 2

    xerr = (xedges[1:] - xedges[:-1])/2

    if logwidth:
        xwidths = np.log(xedges[1:]) - np.log(xedges[:-1])
    else:
        xwidths = xedges[1:] - xedges[:-1]

    ys = vals[:]
    yerrs = np.sqrt(np.diag(covs))
    

    if has_overflow(wrt):
        ys = ys[:-1]
        yerrs = yerrs[:-1]
        vals = vals[:-1]
        covs = covs[:-1,:-1]

    if has_underflow(wrt):
        ys = ys[1:]
        yerrs = yerrs[1:]
        vals = vals[1:]
        covs = covs[1:,1:]

    ys/=xwidths
    yerrs/=xwidths

    if should_logx(wrt):
        ax.set_xscale('log')

    if should_logy():
        ax.set_yscale('log')

    ax.errorbar(xcenters, ys,
                xerr=xerr, yerr=yerrs,
                fmt='o', label=label,
                color=color)

    if label is not None:
        ax.legend()

    return vals, covs, N
 
def compareProjectedEEC(
               x_l, covx_l, 
               binning_l = [{}],
               density = True,
               label_l = None,
               color_l = None,
               isData=False,
               wrt='dR', logwidth=True,
               folder=None, fprefix=None):

    fig, (ax, rax) = setup_ratiopad_canvas(isData)

    vals = []
    covs = []
    Ns = []

    for x, covx, binning, label, color in zip(x_l, covx_l, binning_l, label_l, color_l):
        v, c, n = plotProjectedEEC(x, covx,
                                   ax=ax, isData=isData,
                                   wrt=wrt, binning=binning,
                                   logwidth=logwidth,
                                   density=density,
                                   label=label,
                                   color=color)
        vals.append(v)
        covs.append(c)
        Ns.append(n)

    for ix2 in range(1, len(x_l)):
        if x_l[ix2] is x_l[0]:
            cov1x2 = project_cov1x2_projected(covx_l[0],
                                              binning_l[ix2], 
                                              binning_l[0],
                                              wrt)
            if has_overflow(wrt):
                cov1x2 = cov1x2[:-1,:-1]
            if has_underflow(wrt):
                cov1x2 = cov1x2[1:,1:]

            cov1x2 = maybe_density_cross(vals[ix2], vals[0],
                                         density, density,
                                         Ns[ix2], Ns[0],
                                         cov1x2)
        else:
            cov1x2 = None

        plotProjectedEECratio(vals[ix2], covs[ix2],
                              vals[0], covs[0],
                              cov1x2,
                              ax=rax,
                              isData=isData,
                              wrt=wrt,
                              color = color_l[ix2])

    rax.set_xlabel(get_ax_label(wrt, 'EECproj'))
    rax.set_ylabel('Ratio', loc='center')

    labelstr = binning_string(binning_AND(binning_l))

    ax.text(0.95, 0.05, labelstr, 
            transform=ax.transAxes,
            verticalalignment='bottom',
            horizontalalignment='right',
            fontsize=18,
            weight='bold',
            bbox=dict(facecolor='white', 
                      boxstyle='square',
                      linewidth=3,
                      edgecolor='black'))

    if folder is not None:
        fname = ''
        if fprefix is not None:
            fname += fprefix+"_"
        fname += binning_name(binning_AND(binning_l))
        fname += ".png"

        outname = os.path.join(folder, fname)

        savefig(outname)
    else:
        plt.show()

def plotProjectedCorrelation(cov,
                             binning1={},
                             binning2=None,
                             isData=False,
                             wrt='dR'):
    fig, (ax, cax) = setup_cbar_canvas(isData)

    if binning2 is None:
        projcov = project_projected(None, cov, binning1, wrt)
    else:
        projcov = project_cov1x2_projected(cov, binning1, binning2, wrt)

    correl = np.zeros_like(projcov)
    for i in range(projcov.shape[0]):
        for j in range(projcov.shape[1]):
            correl[i,j] = projcov[i,j] / np.sqrt(projcov[i,i] * projcov[j,j])

    im = ax.pcolormesh(correl, cmap='coolwarm', 
                   vmin=-1, vmax=1)
    fig.colorbar(im, cax=cax, orientation='vertical')

    ax.set_xlabel(get_ax_label(wrt, 'EECproj')+" bin")
    ax.set_ylabel(get_ax_label(wrt, 'EECproj')+" bin")

    plt.show()

def plotProjectedTransfer(transfer,
                          binningGen={},
                          binningReco={}):
    fig, (ax, cax) = setup_cbar_canvas(False)

    projtransfer = bin_transfer_projected(transfer, 
                                              binningGen,
                                              binningReco)

    im = ax.pcolormesh(projtransfer, 
                       cmap='viridis',
                       norm=LogNorm())
    fig.colorbar(im, cax=cax, orientation='vertical')

    ax.set_xlabel("Gen bin")
    ax.set_ylabel("Reco bin")

    plt.show()
