import matplotlib.pyplot as plt
import numpy as np
import mplhep as hep
from matplotlib.colors import Normalize, LogNorm

from util import setup_ratiopad_canvas, setup_plain_canvas, get_ax_edges, project_projected, has_overflow, has_underflow, should_logx, should_logy, project_cov1x2_projected, setup_cbar_canvas, get_ax_label, bin_transfer_projected
from stats import getratio, maybe_density, maybe_density_cross

def plotProjectedEECratio(val1, cov1,
                          val2, cov2,
                          cov1x2,
                          ax=None, isData=False,
                          wrt='dR'):
    if ax is None:
        fig, ax = setup_ratiopad_canvas(isData)
        ax.set_xlabel(get_ax_label(wrt))

    ratio, covratio = getratio(val1, val2, cov1, cov2, cov1x2)

    xedges = get_ax_edges(wrt)
    xcenters = (xedges[1:] + xedges[:-1]) / 2

    xerrs = (xedges[1:] - xedges[:-1]) / 2
    yerrs = np.sqrt(np.diag(covratio))

    ax.errorbar(xcenters, ratio,
                xerr = xerrs, yerr = yerrs,
                fmt='o')

    ax.axhline(1, color='black', linestyle='--')

def plotProjectedEEC(x, covx,
                     ax=None, isData=False,
                     wrt='dR', binning={},
                     logwidth=True,
                     density=True,
                     label = None):
    if ax is None:
        fig, ax = setup_plain_canvas(isData)
        ax.set_xlabel(get_ax_label(wrt))

    ax.set_ylabel("EEC")

    if wrt != 'dR':
        if logwidth:
            print("Warning: logwidth is only sensible for dR. Setting to False")
        logwidth=False

    vals, covs = project_projected(x, covx, binning, onto=wrt)
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
                fmt='o', label=label)

    if label is not None:
        ax.legend()

    return vals, covs, N
 
def compareProjectedEEC(x1, covx1, 
               x2, covx2,
               binning1 = {},
               binning2 = {},
               density1 = True,
               density2 = True,
               label1 = None,
               label2 = None,
               isData=False,
               wrt='dR', logwidth=True):
    fig, (ax, rax) = setup_ratiopad_canvas(isData)

    vals1, covs1, N1 = plotProjectedEEC(x1, covx1,
                                    ax=ax, isData=isData,
                                    wrt=wrt, binning=binning1,
                                    logwidth=logwidth,
                                    density=density1,
                                    label=label1)
    vals2, covs2, N2 = plotProjectedEEC(x2, covx2,
                                    ax=ax, isData=isData,
                                    wrt=wrt, binning=binning2,
                                    logwidth=logwidth,
                                    density=density2,
                                    label=label2)

    if x1 is x2:
        cov1x2 = project_cov1x2_projected(covx1,
                                          binning1, binning2, 
                                          wrt)
        if has_overflow(wrt):
            cov1x2 = cov1x2[:-1,:-1]
        if has_underflow(wrt):
            cov1x2 = cov1x2[1:,1:]

        cov1x2 = maybe_density_cross(vals1, vals2, 
                                     density1, density2,
                                     N1, N2,
                                     cov1x2)
    else:
        cov1x2 = None

    plotProjectedEECratio(vals1, covs1,
                          vals2, covs2,
                          cov1x2,
                          ax=rax, isData=isData,
                          wrt=wrt)

    rax.set_xlabel(get_ax_label(wrt))
    rax.set_ylabel('Ratio', loc='center')

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

    ax.set_xlabel(get_ax_label(wrt)+" bin")
    ax.set_ylabel(get_ax_label(wrt)+" bin")

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
