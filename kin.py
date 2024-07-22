import matplotlib.pyplot as plt
import os.path
import numpy as np
import mplhep as hep
from matplotlib.colors import Normalize, LogNorm

from util import setup_ratiopad_canvas, setup_plain_canvas, get_ax_edges, has_overflow, has_underflow, should_logx, should_logy, setup_cbar_canvas, get_ax_label, get_hist, get_color, get_label, get_lumi, get_xsec, order_by_jet_yield, savefig

def plotKin(Hdict, Hname, AXname,
            ax=None, isData=False,
            color=None, label=None, 
            density=True):
    if ax is None:
        fig, ax = setup_plain_canvas(isData)
        ax.set_xlabel(get_ax_label(AXname, Hname))
        if density:
            ax.set_ylabel("Events [a.u.]")
        else:
            ax.set_ylabel("Events [millions]")

    H = Hdict[Hname].project(AXname)

    vals = H.values(flow=True)
    if H.variances() is None:
        errs = np.sqrt(vals)
    else:
        errs = np.sqrt(H.variances(flow=True))

    if density:
        N = vals.sum()
    else:
        N = 1e6

    vals = vals / N
    errs = errs / N

    xedges = H.axes[0].edges
    xcenters = (xedges[1:] + xedges[:-1]) / 2

    xerr = (xedges[1:] - xedges[:-1])/2

    if has_overflow(AXname):
        errs = errs[:-1]
        vals = vals[:-1]
    if has_underflow(AXname):
        errs = errs[1:]
        vals = vals[1:]

    ax.errorbar(xcenters, vals/(xerr*2),
                xerr = xerr, yerr = errs,
                fmt='o', color=color, label=label)

    ax.set_yscale('log')

    if should_logx(AXname):
        ax.set_xscale('log')

    if label is not None:
        ax.legend()

    return xcenters, xerr, vals, errs

def plotKinRatio(x, xerr,
                 val1, err1,
                 val2, err2,
                 ax=None, isData=False,
                 color=None):
    if ax is None:
        fig, ax = setup_plain_canvas(isData)

    ratio = val1 / val2
    ratioerr = ratio * np.sqrt(np.square(err1/val1) + np.square(err2/val2))

    ax.errorbar(x, ratio, 
                xerr=xerr,  yerr=ratioerr,
                fmt='o', color=color)

    ax.axhline(1, color='black', linestyle='--')

    return ratio, ratioerr

def stack(Hdict_l, 
          Hname, AXname,
          color_l, label_l,
          xsec_l, target_lumi,
          isData=False,
          density=False,
          plotsteps=True,
          normPerBin=False,
          points=False,
          ax=None):
    if ax is None:
        fig, ax = setup_plain_canvas(isData)
        ax.set_xlabel(get_ax_label(AXname, Hname))

        if normPerBin:
            ax.set_ylabel("Relative rate")
        else:
            if density:
                ax.set_ylabel("Events [a.u.]")
            else:
                ax.set_ylabel("Events [millions]")

    stackedvals = []
    stackedvars = []

    xedges = Hdict_l[0][Hname].axes[AXname].edges
    xcenters = (xedges[1:] + xedges[:-1]) / 2
    xerr = (xedges[1:] - xedges[:-1])/2

    for Hdict, label, color, xsec in zip(Hdict_l, label_l, color_l, xsec_l):
        H = Hdict[Hname].project(AXname)

        vals = H.values(flow=True)
        if H.variances() is None:
            variances = np.sqrt(vals)
        else:
            variances = np.sqrt(H.variances(flow=True))

        sumwt = Hdict['sumwt'] 
        expected_evts = xsec * target_lumi * 1000 # xsec in pb, lumi in fb^-1
        weight = expected_evts / sumwt

        vals = vals * weight
        variances = variances * np.square(weight) 

        if len(stackedvals) == 0:
            stackedvals.append(vals)
            stackedvars.append(variances)
        else:
            stackedvals.append(stackedvals[-1] + vals)
            stackedvars.append(stackedvars[-1] + variances)

    if density:
        N = stackedvals[-1].sum()
    else:
        N = 1e6

    for i in range(len(stackedvals)):
        stackedvals[i] = stackedvals[i] / N
        stackedvars[i] = stackedvars[i] / (N*N)

    for i in range(len(stackedvals)):
        if has_overflow(AXname):
            stackedvals[i] = stackedvals[i][:-1]
            stackedvars[i] = stackedvars[i][:-1]
        if has_underflow(AXname):
            stackedvals[i] = stackedvals[i][1:]
            stackedvars[i] = stackedvars[i][1:]

    if plotsteps:
        if points:
            raise ValueError("points and plotsteps makes no sense")

        for i in range(len(stackedvals)):
            thisval = stackedvals[i]
            if i==0:
                prevval = np.zeros_like(thisval)
            else:
                prevval = stackedvals[i-1]
            
            if normPerBin:
                ax.stairs(thisval/stackedvals[-1], xedges, 
                          baseline=prevval/stackedvals[-1],
                      fill=True, color=color_l[i], label=label_l[i])
            else:
                ax.stairs(thisval/(2*xerr), xedges, 
                          baseline=prevval/(2*xerr),
                          fill=True, color=color_l[i], label=label_l[i])
    else:
        if normPerBin:
            raise ValueError("normPerBin and not plotsteps makes no sense")
        else:
            if points:
                ax.errorbar(xcenters, stackedvals[-1]/(2*xerr),
                            xerr=xerr, yerr=np.sqrt(stackedvars[-1]),
                            fmt='o', color=color_l[-1], label=label_l[-1])
            else:
                ax.stairs(stackedvals[-1]/(2*xerr), xedges, 
                          baseline=0,
                          fill=False, color=color_l[-1], 
                          label=label_l[-1],
                          linewidth=3)


    if should_logx(AXname):
        ax.set_xscale('log')

    ax.set_yscale('log')

    #legend = ax.legend(
    #    bbox_to_anchor=(0.9, 0.9),
    #    loc='upper left',
    #    reverse=True,
    #    edgecolor='black',
    #    facecolor='white',
    #    frameon=True
    #)
    #
    #Sx, Sy = plt.gcf().get_size_inches()
    #plt.gcf().set_size_inches(Sx*1.5, Sy)
    #ax.set_box_aspect(0.9)
    #plt.tight_layout()

    return xcenters, xerr, stackedvals[-1], np.sqrt(stackedvars[-1])

def compareDataMC(Hdict_data,
                  Hdict_MC_l,
                  Hname, AXname,
                  data_color, data_label,
                  MC_color_l, MC_label_l,
                  data_lumi, MC_xsec_l,
                  isData=True, density=False,
                  plotsteps=True):


    fig, (ax, rax) = setup_ratiopad_canvas(isData)

    x, xerr, datavals, dataerrs = plotKin(Hdict_data,
                                          Hname, AXname,
                                          ax=ax, 
                                          color=data_color, 
                                          label=data_label,
                                          density=density)

    if type(Hdict_MC_l[0]) is dict:
        x, xerr, MCvals, MCerrs = stack(Hdict_MC_l, 
                                        Hname, AXname,
                                        MC_color_l, MC_label_l,
                                        MC_xsec_l, data_lumi,
                                        density=density,
                                        plotsteps=plotsteps,
                                        ax=ax)

        plotKinRatio(x, xerr,
                     datavals, dataerrs,
                     MCvals, MCerrs,
                     ax=rax, color=data_color)
    else:
        for i in range(len(Hdict_MC_l)):
            Nsample = len(Hdict_MC_l[i])

            x, xerr, MCvals, MCerrs = stack(Hdict_MC_l[i],
                                            Hname, AXname,
                                            [MC_color_l[i]]*Nsample,
                                            [MC_label_l[i]]*Nsample,
                                            MC_xsec_l, data_lumi,
                                            density=density,
                                            plotsteps=plotsteps,
                                            ax=ax)
            plotKinRatio(x, xerr,
                         datavals, dataerrs,
                         MCvals, MCerrs,
                         ax=rax, isData=isData,
                         color=MC_color_l[i])

    legend = ax.legend(
        bbox_to_anchor=(0.9, 0.95),
        loc='upper left',
        reverse=True,
        edgecolor='black',
        facecolor='white',
        frameon=True
    )

    Sx, Sy = plt.gcf().get_size_inches()
    plt.gcf().set_size_inches(Sx*1.5, Sy)

    ax.set_box_aspect(0.9 * 3/4)
    rax.set_box_aspect(0.9 * 1/4)

    #ax.set_ylabel("Events")
    rax.set_xlabel(get_ax_label(AXname, Hname))
    rax.set_ylabel("Data/MC")
    rax.set_ylim(0.5, 1.5)

def compareKin(Hdict_l_l,
               xsec_l_l,
               Hname, AXname,
               color_l, label_l,
               isData=False,
               folder=None,fprefix=None,
               density=True):
    fig, (ax, rax) = setup_ratiopad_canvas(isData)

    vals_l = []
    errs_l = []

    for Hdict_l, xsec_l, color, label in zip(Hdict_l_l, xsec_l_l, color_l, label_l):
        Nsample = len(Hdict_l)
        x, xerr, vals, errs = stack(
            Hdict_l,
            Hname, AXname,
            [color]*Nsample, 
            [label]*Nsample,
            xsec_l, get_lumi(),
            isData=isData,
            density=density,
            plotsteps=False,
            normPerBin=False,
            points=True,
            ax = ax
        )

        vals_l.append(vals)
        errs_l.append(errs)

    for i2 in range(1, len(vals_l)):
        plotKinRatio(x, xerr,
                     vals_l[i2], errs_l[i2],
                     vals_l[0], errs_l[0],
                     ax=rax, isData=isData,
                     color=color_l[i2])

    rax.set_xlabel(get_ax_label(AXname, Hname))
    rax.set_ylabel("Ratio")

    legend = ax.legend(
        bbox_to_anchor=(0.9, 0.95),
        loc='upper left',
        reverse=True,
        edgecolor='black',
        facecolor='white',
        frameon=True
    )

    Sx, Sy = plt.gcf().get_size_inches()
    plt.gcf().set_size_inches(Sx*1.5, Sy)

    ax.set_box_aspect(0.9 * 3/4)
    rax.set_box_aspect(0.9 * 1/4)

    #ax.set_ylabel("Events")
    rax.set_xlabel(get_ax_label(AXname, Hname))
    rax.set_ylabel("Data/MC")
    rax.set_ylim(0.5, 1.5)



    if folder is not None:
        outname = os.path.join(folder, fprefix)
        outname += ".png"
        savefig(outname)
    else:
        plt.show()

def dataMCdriver_varySyst(MCsamples,
                          MCsyst_l,
                          histtag,
                          colors_l,
                          Hname, AXname,
                          density=False,
                          folder=None):
    MCsamples = order_by_jet_yield(MCsamples, histtag)

    MCdicts_l = []
    for MCsyst in MCsyst_l:
        MCdicts_l.append([get_hist(MC, "Kinematics", histtag, MCsyst) for MC in MCsamples])
    
    MC_xsecs = [get_xsec(MC) for MC in MCsamples]
    MC_labels = MCsyst_l

    data_lumi = get_lumi()
    data_color = get_color('DATA')
    data_label = get_label('DATA')

    datadict = get_hist('DATA', "Kinematics", histtag)

    compareDataMC(datadict, MCdicts_l,
                  Hname, AXname,
                  data_color, data_label,
                  colors_l, MC_labels,
                  data_lumi, MC_xsecs,
                  density=density,
                  plotsteps=False,
                  isData=True)

    if folder is not None:
        fname = "%s_%s_%s" % (histtag, Hname, AXname)
        for syst in MCsyst_l:
            fname += "_%s" % syst
        if density:
            fname += "_density"
        fname += ".png"
        outname = os.path.join(folder, fname)
        savefig(outname)
    else:
        plt.show()

def dataMCdriver(MCsamples, 
                 MCsyst,
                 histtag,
                 Hname, AXname,
                 density=False,
                 plotsteps=True,
                 folder=None):
    
    MCsamples = order_by_jet_yield(MCsamples, histtag)

    MCdicts = [get_hist(MC, "Kinematics", histtag, MCsyst) for MC in MCsamples]
    datadict = get_hist('DATA', "Kinematics", histtag)

    data_color = get_color('DATA')
    data_label = get_label('DATA')

    MC_colors = [get_color(MC) for MC in MCsamples]
    if plotsteps:
        MC_labels = [get_label(MC) for MC in MCsamples]
    else:
        MC_labels = ['MC' for MC in MCsamples]

    data_lumi = get_lumi()
    MC_xsecs = [get_xsec(MC) for MC in MCsamples]

    compareDataMC(datadict, MCdicts,
                  Hname, AXname,
                  data_color, data_label,
                  MC_colors, MC_labels,
                  data_lumi, MC_xsecs,
                  density=density,
                  plotsteps=plotsteps,
                  isData=True)

    if folder is not None:
        fname = "%s_%s_%s" % (histtag, Hname, AXname)
        if density:
            fname += "_density"
        if plotsteps:
            fname += "_steps"
        fname += ".png"
        outname = os.path.join(folder, fname)
        savefig(outname)
    else:
        plt.show()

def compareKindriver(samples, systs, histtag,
                     Hname, AXname, 
                     colors,
                     density=False):
    samples = order_by_jet_yield(samples, histtag)

    Hdict_l_l = []
    xsec_l_l = []
    for syst in systs:
        Hdict_l_l.append([get_hist(sample, "Kinematics", histtag, syst) for sample in samples])
        xsec_l_l.append([get_xsec(sample) for sample in samples])

    compareKin(Hdict_l_l,
               xsec_l_l,
               Hname, AXname,
               colors,
               systs, 
               density=density,
               isData=False)

def MCstackdriver(samples, systs, histtag,
                Hname, AXname,
                density=False,
                plotsteps=True,
                normPerBin=False,
                points=False,
                folder=None):

    samples, systs = order_by_jet_yield(zip(samples, systs), histtag)

    Hdicts = [get_hist(sample, "Kinematics", histtag, syst) for sample, syst in zip(samples, systs)]
    colors = [get_color(sample) for sample in samples]
    labels = [get_label(sample) for sample in samples]
    xsecs = [get_xsec(sample) for sample in samples]
    lumi = get_lumi()

    stack(Hdicts,
          Hname, AXname,
          colors, labels,
          xsecs, lumi,
          isData=False,
          density=density,
          plotsteps=plotsteps,
          normPerBin=normPerBin,
          points=points)

    if folder is not None:
        fname = "%s_%s_%s" % (histtag, Hname, AXname)
        if density:
            fname += "_density"
        if normPerBin:
            fname += "_normPerBin"
        if points:
            fname += "_points"
        fname += ".png"
        outname = os.path.join(folder, fname)
        savefig(outname)
    else:
        plt.show()

