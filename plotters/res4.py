import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import numpy as np
import awkward as ak
import util
import fitfuncs
from scipy.optimize import curve_fit
import contextlib

RLedges = np.linspace(0, 1.0, 11)

redges_teedipole = np.linspace(0, 1.0, 21)
ctedges_teedipole = np.linspace(0, np.pi/2, 21)

redges_triangle = np.linspace(0, 1.0, 21)
ctedges_triangle = np.linspace(0, 2*np.pi, 31)

area_teedipole = 0.5 * (redges_teedipole[1:]**2 - redges_teedipole[:-1]**2)[:, None] \
                     * (ctedges_teedipole[1:] - ctedges_teedipole[:-1])[None, :]

def compute_flux(H, RLbin, ptbin, iToy,
                 jacobian=True,
                 normalize=True):
    H2d = H[{'R' : RLbin, 'pt' : ptbin, 'bootstrap' : iToy}]

    vals = H2d.values(flow=True)

    if jacobian:
        vals = vals/area_teedipole

    if normalize:
        vals = vals/np.sum(vals)

    return vals

def compute_angular_average(H, RLbin, ptbin, iToy,
                            jacobian=True,
                            normalize=True):
    flux = compute_flux(H, RLbin, ptbin, iToy,
                        jacobian=jacobian,
                        normalize=normalize)

    angular_avg = np.mean(flux, axis=1)

    return angular_avg

def compute_angular_fluctuation(H, RLbin, ptbin, iToy,
                                jacobian=True,
                                normalize=True):
    flux = compute_flux(H, RLbin, ptbin, iToy,
                        jacobian=jacobian,
                        normalize=normalize)

    angular_avg = compute_angular_average(H, RLbin, ptbin, iToy,
                                          jacobian=jacobian,
                                          normalize=normalize)

    angular_fluctuation = flux/angular_avg[:,None]

    return angular_fluctuation

def compute_differnce(H1, H2, RLbin, ptbin, iToy,
                      jacobian=True,
                      normalize=True,
                      relative=False):
    flux1 = compute_flux(H1, RLbin, ptbin, iToy,
                         jacobian=jacobian,
                         normalize=normalize)

    flux2 = compute_flux(H2, RLbin, ptbin, iToy,
                         jacobian=jacobian,
                         normalize=normalize)

    if relative:
        return np.nan_to_num((flux2 - flux1)/flux1)
    else:
        return flux2 - flux1

def errs_from_toys(fluxfunc, pulls=False, **kwargs):

    nominal = fluxfunc(**kwargs, iToy=0)
    
    if 'H' in kwargs:
        ntoys = kwargs['H'].axes['bootstrap'].extent - 1
    elif 'H1' in kwargs:
        ntoys = kwargs['H1'].axes['bootstrap'].extent - 1
    else:
        raise ValueError("No H or H1 provided to determine number of toys")

    sumdiff = np.zeros_like(nominal)
    sumdiff2 = np.zeros_like(nominal)

    for iToy in range(1, ntoys+1):
        nextflux = fluxfunc(**kwargs, iToy=iToy)

        sumdiff += (nextflux - nominal)
        sumdiff2 += (nextflux - nominal)**2

    sumdiff /= ntoys
    sumdiff2 /= ntoys

    std_diff = np.sqrt(sumdiff2 - sumdiff**2)

    if pulls:
        return nominal/std_diff, np.ones_like(std_diff)
    else:
        return nominal, std_diff

def flux_vals_errs(H, RLbin, ptbin,
                   jacobian=True,
                   normalize=True,
                   pulls=False):
    return errs_from_toys(compute_flux,
                          H=H,
                          RLbin=RLbin,
                          ptbin=ptbin,
                          jacobian=jacobian,
                          normalize=normalize,
                          pulls=pulls)

def angular_avg_vals_errs(H, RLbin, ptbin,
                          jacobian=True,
                          normalize=True,
                          pulls=False):
    return errs_from_toys(compute_angular_average,
                          H=H,
                          RLbin=RLbin,
                          ptbin=ptbin,
                          jacobian=jacobian,
                          normalize=normalize,
                          pulls=pulls)

def angular_fluctuation_vals_errs(H, RLbin, ptbin,
                                  jacobian=True,
                                  normalize=True,
                                  pulls=False):
    return errs_from_toys(compute_angular_fluctuation,
                          H=H,
                          RLbin=RLbin,
                          ptbin=ptbin,
                          jacobian=jacobian,
                          normalize=normalize,
                          pulls=pulls)

def difference_vals_errs(H1, H2, RLbin, ptbin,
                         jacobian=True,
                         normalize=True,
                         pulls=False,
                         relative=False):
    return errs_from_toys(compute_differnce,
                          H1=H1,
                          H2=H2,
                          RLbin=RLbin,
                          ptbin=ptbin,
                          jacobian=jacobian,
                          normalize=normalize,
                          pulls=pulls,
                          relative=relative)

def plot_flux_2d(H, RLbin, ptbin,
                 jacobian=True,
                 normalize=True,
                 logz = True,
                 cmap='inferno'):
    flux = compute_flux(H, RLbin, ptbin, 0,
                        jacobian=jacobian,
                        normalize=normalize)

    ptedges = H.axes['pt'].edges

    fig = plt.figure(figsize=(12, 12))
    try:
        ax = fig.add_subplot(111, projection='polar')
        if logz:
            norm = LogNorm(vmin=flux[flux>0].min(), vmax=flux.max())
        else:
            norm = Normalize(vmin=flux.min(), vmax=flux.max())

        pc1 = ax.pcolormesh(ctedges_teedipole, redges_teedipole,
                            flux,
                            cmap=cmap,
                            shading='auto',
                            norm=norm)
        pc2 = ax.pcolormesh(np.pi-ctedges_teedipole, redges_teedipole,
                            flux,
                            cmap=cmap,
                            shading='auto',
                            norm=norm)
        pc3 = ax.pcolormesh(np.pi+ctedges_teedipole, redges_teedipole,
                            flux,
                            cmap=cmap,
                            shading='auto',
                            norm=norm)
        pc4 = ax.pcolormesh(2*np.pi-ctedges_teedipole, redges_teedipole,
                            flux,
                            cmap=cmap,
                            shading='auto',
                            norm=norm)

        plt.colorbar(pc1, ax=ax)

        fig.suptitle("%0.1f < RL < %0.1f\n%0.0f < pT < %0.0f" % (RLedges[RLbin-1], RLedges[RLbin], ptedges[ptbin], ptedges[ptbin+1]))

        plt.tight_layout()
        plt.show()
    finally:
        plt.close(fig)

def plot_angular_avg(H, RLbin, ptbin,
                     jacobian=True,
                     normalize=True,
                     logx=False,
                     logy=True):
    fig = plt.figure(figsize=(12, 12))
    try:
        val, err = angular_avg_vals_errs(H, RLbin, ptbin, 
                                         jacobian=jacobian,
                                         normalize=normalize)
        rmid = (redges_teedipole[1:] + redges_teedipole[:-1])/2
        rwidths = redges_teedipole[1:] - redges_teedipole[:-1]

        ax = fig.add_subplot(111)
        ax.errorbar(rmid, val,
                    xerr = rwidths/2,
                    yerr=err,
                    fmt='o')
        if logx:
            ax.set_xscale('log')
        if logy:
            ax.set_yscale('log')
        ax.set_xlabel('r')
        ax.set_ylabel('Angular average')

        plt.tight_layout()
        plt.show()
    finally:
        plt.close(fig)

def plot_angular_fluctuation_2d(H, RLbin, ptbin,
                                jacobian=True,
                                normalize=True,
                                cmap='coolwarm'):
    flux = compute_angular_fluctuation(H, RLbin, ptbin, 0,
                                        jacobian=jacobian,
                                        normalize=normalize)
    ptedges = H.axes['pt'].edges
    fig = plt.figure(figsize=(12, 12))
    try:
        ax = fig.add_subplot(111, projection='polar')

        d1 = flux-1
        maxvar = np.max(np.abs(d1))

        norm = Normalize(vmin=1-maxvar, vmax=1+maxvar)

        pc1 = ax.pcolormesh(ctedges_teedipole, redges_teedipole,
                            flux,
                            cmap=cmap,
                            shading='auto',
                            norm=norm)
        pc2 = ax.pcolormesh(np.pi-ctedges_teedipole, redges_teedipole,
                            flux,
                            cmap=cmap,
                            shading='auto',
                            norm=norm)
        pc3 = ax.pcolormesh(np.pi+ctedges_teedipole, redges_teedipole,
                            flux,
                            cmap=cmap,
                            shading='auto',
                            norm=norm)
        pc4 = ax.pcolormesh(2*np.pi-ctedges_teedipole, redges_teedipole,
                            flux,
                            cmap=cmap,
                            shading='auto',
                            norm=norm)
        plt.colorbar(pc1, ax=ax)

        fig.suptitle("%0.1f < RL < %0.1f\n%0.0f < pT < %0.0f" % (RLedges[RLbin-1], RLedges[RLbin], ptedges[ptbin], ptedges[ptbin+1]))
        plt.tight_layout()
        plt.show()
    finally:
        plt.close(fig)

def plot_angular_fluctuation_1d(H, RLbin, ptbin,
                                jacobian=True,
                                normalize=True,
                                rstep=5):
    fig = plt.figure(figsize=(12, 12))
    try:
        val, err = angular_fluctuation_vals_errs(H, RLbin, ptbin, 
                                                 jacobian=jacobian,
                                                 normalize=normalize)
        tmid = (ctedges_teedipole[1:] + ctedges_teedipole[:-1])/2
        twidths = ctedges_teedipole[1:] - ctedges_teedipole[:-1]

        ax = fig.add_subplot(111)
        for i in range(0, val.shape[0], rstep):
            q = ax.errorbar(tmid, val[i],
                        xerr = twidths/2,
                        yerr=err[i],
                        fmt='o', label=f'%0.2f < r < %0.2f' % (redges_teedipole[i], redges_teedipole[i+1]))
            ax.errorbar(np.pi-tmid, val[i],
                        xerr = twidths/2,
                        yerr=err[i],
                        fmt='o', color=q[0].get_color())
            ax.errorbar(np.pi+tmid, val[i],
                        xerr = twidths/2,
                        yerr=err[i],
                        fmt='o', color=q[0].get_color())
            ax.errorbar(2*np.pi-tmid, val[i],
                        xerr = twidths/2,
                        yerr=err[i],
                        fmt='o', color=q[0].get_color())
            
        ax.axhline(1, color='black', linestyle='--')
        ax.axvline(np.pi/2, color='gray', linestyle='--')
        ax.axvline(np.pi, color='gray', linestyle='--')
        ax.axvline(3*np.pi/2, color='gray', linestyle='--')

        ax.set_xlabel('$\\phi$')
        ax.set_ylabel('Angular fluctuation')
        ax.legend(loc='upper right', frameon=True)

        plt.tight_layout()
        plt.show()
    finally:
        plt.close(fig)

def plot_diff(H1, H2, RLbin, ptbin,
              jacobian=True, normalize=True,
              pulls=False, relative=False,
              plot2d=True, plotr1d=True, plotphi1d=True, plothist=True,
              savefig=None, show=True):
    ptedges = H1.axes['pt'].edges
    if plot2d:
        fig = plt.figure(figsize=(12, 12))
        try:
            flux = compute_differnce(H1, H2, RLbin, ptbin, 0,
                                      jacobian=jacobian,
                                      normalize=normalize,
                                      relative=relative)

            ax = fig.add_subplot(111, projection='polar')

            d1 = flux
            maxvar = np.max(np.abs(d1))

            norm = Normalize(vmin=-maxvar, vmax=maxvar)

            pc1 = ax.pcolormesh(ctedges_teedipole, redges_teedipole,
                                flux,
                                cmap='coolwarm',
                                shading='auto',
                                norm=norm)
            pc2 = ax.pcolormesh(np.pi-ctedges_teedipole, redges_teedipole,
                                flux,
                                cmap='coolwarm',
                                shading='auto',
                                norm=norm)
            pc3 = ax.pcolormesh(np.pi+ctedges_teedipole, redges_teedipole,
                                flux,
                                cmap='coolwarm',
                                shading='auto',
                                norm=norm)
            pc4 = ax.pcolormesh(2*np.pi-ctedges_teedipole, redges_teedipole,
                                flux,
                                cmap='coolwarm',
                                shading='auto',
                                norm=norm)
            cb = plt.colorbar(pc1, ax=ax)

            if relative:
                cb.set_label('Relative difference')
            else:
                cb.set_label('Difference')

            fig.suptitle("%0.1f < RL < %0.1f\n%0.0f < pT < %0.0f" % (RLedges[RLbin-1], RLedges[RLbin], ptedges[ptbin], ptedges[ptbin+1]))
            plt.tight_layout()

            if savefig is not None:
                plt.savefig(savefig + '_2d.png', dpi=300, format='png', bbox_inches='tight')
            if show:
                plt.show()
        finally:
            plt.close(fig)

    if plotr1d or plotphi1d:
        val, err = difference_vals_errs(H1, H2, RLbin, ptbin,
                                        jacobian=jacobian,
                                        normalize=normalize,
                                        pulls=pulls,
                                        relative=relative)

    if plotr1d:
        fig = plt.figure(figsize=(12, 6))
        try:
            rmid = (redges_teedipole[1:] + redges_teedipole[:-1])/2
            rwidths = redges_teedipole[1:] - redges_teedipole[:-1]

            rmid_b = np.broadcast_to(rmid[:, None], val.shape)
            rwidths_b = np.broadcast_to(rwidths[None, :], val.shape)

            ctmid = (ctedges_teedipole[1:] + ctedges_teedipole[:-1])/2
            cmap = plt.get_cmap('Reds')

            ax = fig.add_subplot(111)

            for i in range(val.shape[1]):
                jitter = ((-val.shape[1]/2 + i)/val.shape[1]) * rwidths/3
                ax.errorbar(rmid_b[:,i]+jitter, 
                            val[:,i],
                            xerr = rwidths_b[:,i]/2,
                            yerr=err[:,i],
                            color=cmap(ctmid[i]/ctmid.max()),
                            fmt='o')
            ax.set_xlabel('r')

            if pulls:
                ax.set_ylabel('Pulls')
            elif relative:
                ax.set_ylabel('Relative difference')
            else:
                ax.set_ylabel('Difference')

            ax.axhline(0, color='black', linestyle='--')

            plt.tight_layout()
            if savefig is not None:
                plt.savefig(savefig + '_r1d.png', dpi=300, format='png', bbox_inches='tight')
            if show:
                plt.show()
        finally:
            plt.close(fig)

    if plotphi1d:
        fig = plt.figure(figsize=(12, 6))
        try:
            tmid = (ctedges_teedipole[1:] + ctedges_teedipole[:-1])/2
            twidths = ctedges_teedipole[1:] - ctedges_teedipole[:-1]

            tmid_b = np.broadcast_to(tmid[None, :], val.shape)
            twidths_b = np.broadcast_to(twidths[:, None], val.shape)

            jitter = np.random.uniform(-twidths/5, twidths/5, size=tmid_b.shape)
            rmid_b = np.broadcast_to(rmid[:, None], val.shape)
            cmap = plt.get_cmap('Reds')
            colors = cmap(rmid_b.ravel())

            ax = fig.add_subplot(111)
            for i in range(val.shape[0]):
                jitter = ((-val.shape[0]/2 + i)/val.shape[0]) * twidths/3
                ax.errorbar(tmid_b[i]+jitter, val[i],
                            xerr = twidths_b[i]/2,
                            yerr=err[i],
                            color=cmap(rmid[i]/rmid.max()),
                            fmt='o')
            ax.set_xlabel('$\\phi$')
            if pulls:
                ax.set_ylabel('Pulls')
            elif relative:
                ax.set_ylabel('Relative difference')
            else:
                ax.set_ylabel('Difference')

            ax.axhline(0, color='black', linestyle='--')

            plt.tight_layout()
            if savefig is not None:
                plt.savefig(savefig + '_phi1d.png', dpi=300, format='png', bbox_inches='tight')
            if show:
                plt.show()
        finally:
            plt.close(fig)

    if plothist:
        fig = plt.figure(figsize=(12, 12))
        try:
            ax = fig.add_subplot(111)
            ax.hist(val.ravel(), bins=21, histtype='step')
            if pulls:
                ax.set_xlabel('Pulls')
            elif relative:
                ax.set_xlabel('Relative difference')
            else:
                ax.set_xlabel('Difference')
            ax.set_ylabel('Counts')
            ax.axvline(0, color='black', linestyle='--')

            plt.tight_layout()
            if savefig is not None:
                plt.savefig(savefig + '_hist.png', dpi=300, format='png', bbox_inches='tight')
            if show: 
                plt.show()
        finally:
            plt.close(fig)

def model_background(arr, totalname, backgroundname_l, label_l,
                     btag, ptbin, RLbin,
                     triangle=False,
                     whichfit='Linear'):

    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2,2, figsize=(16,12))

    total_nom = compute_shape(arr[totalname], 
                              btag, ptbin, RLbin, 0,
                              triangle=triangle,
                              normalize=False,
                              jacobian=True)
    bkgs_nom = {}
    for bkgname in backgroundname_l:
        bkgs_nom[bkgname] = compute_shape(arr[bkgname], 
                                          btag, ptbin, RLbin, 0,
                                          triangle=triangle,
                                          normalize=False,
                                          jacobian=True)

    total_var = []
    bkgs_var = {}
    for bkgname in backgroundname_l:
        bkgs_var[bkgname] = []

    ntoys = arr[totalname].shape[1] - 1
    for iToy in range(1, ntoys+1):
        nexttotal = compute_shape(arr[totalname],
                                  btag, ptbin, RLbin, iToy,
                                  triangle=triangle,
                                  normalize=False,
                                  jacobian=True)
        total_var.append(nexttotal * total_nom.sum()/nexttotal.sum())

        for bkgname in backgroundname_l:
            nextbkg = compute_shape(arr[bkgname],
                                    btag, ptbin, RLbin, iToy,
                                    triangle=triangle,
                                    normalize=False,
                                    jacobian=True)
            bkgs_var[bkgname].append(nextbkg * bkgs_nom[bkgname].sum()/nextbkg.sum())

    thediff = total_var - total_nom[None,:,:]

    total_var = np.asarray(total_var)
    dtotal = np.sqrt(np.sum(np.square(total_var - total_nom[None,:,:]), axis=0))/ntoys

    dbkgs = {}
    for bkgname in backgroundname_l:
        bkgs_var[bkgname] = np.asarray(bkgs_var[bkgname])
        dbkgs[bkgname] = np.sqrt(np.sum(np.square(bkgs_var[bkgname] - bkgs_nom[bkgname][None,:,:]), axis=0))/ntoys


    residuals = {}
    dresiduals = {}
    fitfunc = fitfuncs.get_func(whichfit)
    for bkgname, label in zip(backgroundname_l, label_l):
        popt, pcov = curve_fit(
            fitfunc.func,
            total_nom.ravel(),
            bkgs_nom[bkgname].ravel(),
            p0 = fitfunc.p0(),
            sigma = dbkgs[bkgname].ravel(),
            absolute_sigma = True
        )
        
        fine_x = np.linspace(total_nom.min(), 
                             total_nom.max(), 
                             1000)
        fine_y = fitfunc.func(fine_x, *popt)
        q = ax0.plot(fine_x, fine_y, '--',
                     label=fitfunc.get_text(popt))

        ax0.errorbar(total_nom.ravel(), 
                     bkgs_nom[bkgname].ravel(), 
                     xerr=dtotal.ravel(),
                     yerr=dbkgs[bkgname].ravel(),
                     fmt='o',
                     label=label,
                     color = q[0].get_color())

        residuals[bkgname] = (bkgs_nom[bkgname] - fitfunc.func(total_nom, *popt))
        dresiduals[bkgname] = np.sqrt(np.square(dbkgs[bkgname]))

    ax0.legend(fontsize=16)
    ax0.set_xlabel("Total")
    ax0.set_ylabel("Background")
    ax0.set_xscale('log')
    ax0.set_yscale('log')

    rcenters = (redges_teedipole[1:] + redges_teedipole[:-1])/2
    ctcenters = (ctedges_teedipole[1:] + ctedges_teedipole[:-1])/2

    rcenters_b = np.broadcast_to(rcenters[:, None], total_nom.shape)
    ctcenters_b = np.broadcast_to(ctcenters[None, :], total_nom.shape)

    for bkgname, label in zip(backgroundname_l, label_l):
        z = residuals[bkgname]/total_nom
        maxZ = np.max(np.abs(z))

        ax1.pcolormesh(rcenters_b, ctcenters_b, z, 
                       cmap='RdBu', vmin=-maxZ, vmax=maxZ)

        ax2.errorbar(rcenters_b.ravel(), 
                     z.ravel(),
                     fmt='o',
                     label=label)

        ax3.errorbar(ctcenters_b.ravel(),
                     z.ravel(),
                     fmt='o',
                     label=label)

    ax1.set_xlabel('r')
    ax1.set_ylabel(r'$\phi$')

    ax2.set_xlabel('r')
    ax2.set_ylabel('Residuals/Total')
    ax2.axhline(0, color='red', linestyle='--')

    ax3.set_xlabel(r'$\phi$')
    ax3.set_ylabel('Residuals/Total')
    ax3.axhline(0, color='red', linestyle='--')

    plt.tight_layout()
    plt.show()
