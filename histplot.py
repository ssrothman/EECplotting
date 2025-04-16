import matplotlib.pyplot as plt
import hist
import numpy as np

def _call_errorbar(ax, x, y, xerr, yerr, **kwargs):
    return ax.errorbar(
        x, y, xerr = xerr, yerr = yerr,
        fmt = 'o', markersize=4, capsize=1, 
        **kwargs
    )


def simon_histplot(H, ax=None, density=False, **kwargs):
    if len(H.axes) > 1:
        raise ValueError("histplot only supports 1D histograms")

    if ax is None:
        ax = plt.gca()

    vals = H.values()
    errs = np.sqrt(H.variances())

    edges = H.axes[0].edges
    centers = H.axes[0].centers
    widths = H.axes[0].widths

    if type(H.axes[0]) is hist.axis.Integer:
        centers -= 0.5

    if density:
        N = np.sum(vals)
        vals /= N
        errs /= N

    plotvals = vals/widths
    ploterrs = errs/widths

    return _call_errorbar(ax, centers, plotvals, widths/2, ploterrs, **kwargs)

def simon_histplot_ratio(Hnum, Hdenom, ax=None, 
                         density=False, pulls=False, **kwargs):

    if len(Hnum.axes) > 1 or len(Hdenom.axes) > 1:
        raise ValueError("histplot only supports 1D histograms")

    if Hnum.axes[0] != Hdenom.axes[0]:
        raise ValueError("histograms must have the same axes")

    if ax is None:
        ax = plt.gca()

    vals_num = Hnum.values()
    errs_num = np.sqrt(Hnum.variances())

    vals_denom = Hdenom.values()
    errs_denom = np.sqrt(Hdenom.variances())

    edges = Hnum.axes[0].edges 
    centers = Hnum.axes[0].centers
    widths = Hnum.axes[0].widths

    if type(Hnum.axes[0]) is hist.axis.Integer:
        centers -= 0.5

    if density:
        Nnum = np.sum(vals_num)
        Ndenom = np.sum(vals_denom)

        vals_num /= Nnum
        vals_denom /= Ndenom
        
        errs_num /= Nnum
        errs_denom /= Ndenom

    vals_num /= widths
    errs_num /= widths

    vals_denom /= widths
    errs_denom /= widths

    ratio = vals_num / vals_denom
    ratio_err = np.sqrt(
        (errs_num/vals_denom)**2 + 
        (vals_num*errs_denom/vals_denom**2)**2
    )

    if pulls:
        ratio = ratio-1
        ratio = ratio/ratio_err
        ratio_err = np.ones_like(ratio)

    return _call_errorbar(ax, centers, ratio, widths/2, ratio_err, **kwargs)
