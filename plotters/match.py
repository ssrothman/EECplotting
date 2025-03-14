import matplotlib.pyplot as plt
import hist
from scipy.optimize import curve_fit
import numpy as np

from util import setup_plain_canvas

def plot_Jmatch(H, wrt, cutsdict):
    fig, ax = setup_plain_canvas(False)

    try:
        Hp = H['Jmatch'][cutsdict]
        Hp = Hp.project(wrt, "match")
        vals = Hp.values(flow=True)
        err2s = Hp.variances(flow=True)

        matched = vals[:, 2]
        unmatched = vals[:, 1]
        rate = matched/(matched + unmatched)

        matched_err2 = err2s[:, 2]
        unmatched_err2 = err2s[:, 1]
        rate_err = np.sqrt(matched_err2)/(matched + unmatched)

        axis = Hp.axes[wrt]
        centers = axis.centers
        widths = axis.widths

        plt.errorbar(centers, rate[1:-1], 
                     xerr=widths/2, 
                     yerr=rate_err[1:-1],
                     fmt='o', color='black')
        plt.xlabel(axis.label)
        plt.ylabel("Matched fraction")
        plt.ylim(0, 1.1)
        plt.axhline(1, color='red', linestyle='--')
        plt.grid()
        plt.show()
    finally:
        plt.close()

def plot_Pmatch(H, wrt, cutsdict, with_type=None, with_charge=None,
                title=None):
    fig, ax = setup_plain_canvas(False)

    try:
        Hp = H[cutsdict]
        total = Hp.project(wrt).values(flow=True)

        if ('matchType' in Hp.axes.name) and ('sameCharge' in Hp.axes.name):
            Hmatch = Hp.project(wrt, "match", "matchType", 'sameCharge').values(flow=True)
            if with_type is None:
                Hmatch = np.sum(Hmatch, axis=2)
            else:
                Hmatch = Hmatch[:, :, with_type]

            if with_charge is None:
                Hmatch = np.sum(Hmatch, axis=2)
            else:
                Hmatch = Hmatch[:, :, with_charge]

            matched = Hmatch[:, 3]
        else:
            Hmatch = Hp.project(wrt, "match").values(flow=True)

            matched = Hmatch[:, 2]

        rate = matched/total
        rate_err = np.sqrt(matched)/total

        axis = Hp.axes[wrt]
        centers = axis.centers
        widths = axis.widths

        if axis.traits.overflow:
            rate = rate[:-1]
            rate_err = rate_err[:-1]
        if axis.traits.underflow:
            rate = rate[1:]
            rate_err = rate_err[1:]

        plt.errorbar(centers, rate, 
                     xerr=widths/2, 
                     yerr=rate_err,
                     fmt='o', color='black')
        plt.xlabel(axis.label)
        plt.ylabel("Matched fraction")
        plt.ylim(0, 1.1)
        plt.axhline(1, color='red', linestyle='--')
        plt.grid()
        if axis.transform is not None:
            plt.xscale('log')
        plt.text(0.10, 0.90, title,
                 fontsize=24,
                 bbox = {
                     'facecolor': 'white',
                     'edgecolor': 'black',
                     'boxstyle': 'round'
                    })
        plt.show()
    finally:
        plt.close()

def percentile(x, w, q):
    sumwt = np.sum(w)
    cumsum = np.cumsum(w)
    cumsum /= sumwt
    return np.interp(q/100, cumsum, x)

def plot_res(H, wrt, cutsdict, fit, rebin=1):
    fig, ax = setup_plain_canvas(False)

    try:
        Hp = H[wrt][cutsdict]
        Hp = Hp.project("d%s"%wrt)[::hist.rebin(rebin)]
        vals = Hp.values(flow=True)
        errs = np.sqrt(Hp.variances(flow=True))

        axis = Hp.axes["d%s"%wrt]
        centers = axis.centers
        widths = axis.widths

        if axis.traits.overflow:
            vals = vals[:-1]
            errs = errs[:-1]
        if axis.traits.underflow:
            vals = vals[1:]
            errs = errs[1:]

        if fit=='no':
            pass
        elif fit in ['gaus', 'cruijff']:
            from scipy.optimize import curve_fit
            fitrange_start = np.argmax(vals > vals/1e6)
            fitrange_end = len(vals) - np.argmax(vals[::-1] > vals/1e6)
            fitrange = slice(fitrange_start, fitrange_end)
            fitvals = vals[fitrange]
            fitcenters = centers[fitrange]

            if fit=='gaus':
                def fitfunc(x, a, mu, sigma):
                    return a*np.exp(-(x-mu)**2/(2*sigma**2))



                Aguess = fitvals.max()
                muguess = fitcenters[np.argmax(fitvals)]
                sigmaguess = np.sum(fitcenters*fitcenters*fitvals)/np.sum(fitvals)
                p0 = [Aguess, muguess, sigmaguess]
            elif fit=='cruijff':
                def fitfunc(x, a, mu, sigmaL, sigmaR):
                    ans = np.zeros_like(x)
                    L = x < mu
                    R = x >= mu
                    ans[L] = a*np.exp(-(x[L]-mu)**2/(2*sigmaL**2))
                    ans[R] = a*np.exp(-(x[R]-mu)**2/(2*sigmaR**2))
                    return ans

                Aguess = fitvals.max()
                muguess = fitcenters[np.argmax(fitvals)]
                sigmaguess = np.sum(fitcenters*fitcenters*fitvals)/np.sum(fitvals)
                p0 = [Aguess, muguess, sigmaguess, sigmaguess]

            popt, pcov = curve_fit(fitfunc, fitcenters, fitvals, 
                                   p0=p0)
            fine_centers = np.linspace(centers[0], centers[-1], 500)
            plt.plot(fine_centers, 
                     fitfunc(fine_centers, *popt), 
                     'b')

            if fit=='gaus':
                mu = popt[1]
                sigma = popt[2]
            elif fit=='cruijff':
                mu = popt[1]
                sigmaL = popt[2]
                sigmaR = popt[3]
                sigma = (sigmaL + sigmaR)/2
            fittext = "$\\mu = %.2g$\n$\\sigma = %.2g$"%(mu, sigma)
        elif fit=='quantile':

            leftQ = (100-68.27)/2
            rightQ = 100 - leftQ
            q25 = percentile(centers, vals, leftQ)
            q50 = percentile(centers, vals, 50)
            q75 = percentile(centers, vals, rightQ)

            plt.axvline(q25, color='blue', linestyle='--')
            plt.axvline(q50, color='blue', linestyle='--')
            plt.axvline(q75, color='blue', linestyle='--')

            fittext = 'Median = %.2g\n$\\sigma_{68} = %.2g$'%(q50, (q75-q25)/2)

        if fit != 'no':
            plt.text(0.05, 0.95, fittext, 
                    transform=ax.transAxes, 
                    bbox = {
                        'facecolor': 'white',
                        'edgecolor': 'black',
                        'boxstyle': 'round'
                    },
                    verticalalignment='top')

        plt.errorbar(centers, vals, 
                     xerr=widths/2, 
                     yerr=errs,
                     fmt='o', color='black')
        plt.xlabel(axis.label)
        plt.ylabel("Counts / bin")
        plt.grid()
        plt.axvline(0, color='red', linestyle='--')

        plt.tight_layout()
        plt.show()
    finally:
        plt.close()

def res_line(H, wrt, cutsdict, 
             step=5, etastep=2,
             which='sigma', 
             fitmode='TrackerAngRes_QuadAdd'):

    fig, ax = setup_plain_canvas(False)
    try:
        Hp = H[wrt][cutsdict]
        Hp = Hp.project('pt', 'eta', "d%s"%wrt)

        ax_eta = Hp.axes['eta']
        eta_edges = ax_eta.edges
        eta_centers = ax_eta.centers

        ax_pt = Hp.axes['pt']
        pt_edges = ax_pt.edges
        pt_centers = ax_pt.centers

        ax_wrt = Hp.axes["d%s"%wrt]
        wrt_centers = ax_wrt.centers

        for iEta in range(len(eta_centers)//etastep):
            print(iEta)
            medians = []
            widths = []

            thecenters = []
            thewidths = []

            for i in range(len(pt_centers)//step):
                cut = {'pt': slice(step*i, step*(i+1), sum),
                       'eta': slice(etastep*iEta, etastep*(iEta+1), sum)}
                Hcut = Hp[cut]

                vals = Hcut.values(flow=False)
                median = percentile(wrt_centers, vals, 50)
                low = percentile(wrt_centers, vals, (100-68.27)/2)
                high = percentile(wrt_centers, vals, 100 - (100-68.27)/2)
                medians.append(median)
                widths.append((high-low)/2)

                high_edge = pt_edges[step*(i+1)]
                low_edge = pt_edges[step*i]

                thecenters.append((high_edge + low_edge)/2)
                thewidths.append((high_edge - low_edge)/2)


            if which == 'mu':
                plt.errorbar(thecenters, medians, fmt='o', color='black',
                             xerr=thewidths)
            else:
                if fitmode is not None:
                    if fitmode == 'TrackerAngRes_NoQuad':
                        import fitfuncs.TrackerAngRes_NoQuad as fitclass
                    elif fitmode == 'TrackerAngRes_QuadAdd':
                        import fitfuncs.TrackerAngRes_QuadAdd as fitclass
                    elif fitmode == 'TrackerAngRes_FloatPower':
                        import fitfuncs.TrackerAngRes_FloatPower as fitclass
                    else:
                        raise ValueError("Unknown fitmode %s"%fitmode)

                    thecenters = np.asarray(thecenters)
                    widths = np.asarray(widths)

                    good = np.isfinite(widths)
                    thecenters_forfit = thecenters[good][1:-1]
                    widths_forfit = widths[good][1:-1]

                    popt, pcov = curve_fit(
                            fitclass.func, 
                            thecenters_forfit, widths_forfit, 
                            p0=fitclass.p0())
                    print(popt)

                    fine_centers = np.linspace(thecenters[0],
                                               thecenters[-1], 
                                               500)
                    q = plt.plot(fine_centers, 
                             fitclass.func(fine_centers, *popt), 
                             linestyle='--',
                             label=fitclass.get_text(popt))

                plt.errorbar(thecenters, widths, fmt='o',
                             xerr=thewidths,
                             color = q[0].get_color() if fitmode is not None else None,
                             label = '$%0.2f < |\\eta| < %0.2f$'%(eta_edges[etastep*iEta],
                                                                  eta_edges[etastep*(iEta+1)]))

        plt.legend()
        plt.xlabel(ax_pt.label)
        plt.ylabel(ax_wrt.label)
        plt.grid()
        plt.xscale('log')
        plt.yscale('log')
        plt.tight_layout()
        plt.show()
    finally:
        plt.close()

def plot_Pres():
    fig, ax = setup_plain_canvas(False)

    try:
        pass
    finally:
        plt.close()
