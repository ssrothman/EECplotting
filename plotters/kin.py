import matplotlib.pyplot as plt
import os.path
import numpy as np
import mplhep as hep
from matplotlib.colors import Normalize, LogNorm
import hist
import pyarrow as pa

from histplot import simon_histplot, simon_histplot_ratio

xlabels = {
    'pt' : '$p_{T} [GeV]$',
    'eta' : '$\\eta$',
    'phi'  : '$\\phi$',
    'nConstituents' : 'Number of jet constituents',
    'nPassingParts' : 'Number of jet constitunts passing selections',
    'passLooseB' : 'B-tagged [loose working point]',
    'passMediumB' : 'B-tagged [medium working point]',
    'passTightB' : 'B-tagged [tight working point]',
    'nTrueInt' : 'MC-Truth number of interactions',
    'rho' : 'Transverse energy density [GeV]',
    'MET' : 'Missing transverse energy [GeV]',
    'numLooseB' : 'Number of loose B-jets',
    'numMediumB' : 'Number of medium B-jets',
    'numTightB' : 'Number of tight B-jets',
    'numJets' : 'Number of jets',
    'Zpt' : 'Dimuon $p_{T}$ [GeV]',
    'Zy' : 'Dimuon rapidity',
    'Zmass' : 'Dimuon mass [GeV]',
    'leadMuPt' : 'Leading muon $p_{T}$ [GeV]',
    'leadMuEta' : 'Leading muon $\\eta$',
    'leadMuCharge' : 'Leading muon charge',
    'subMuPt' : 'Subleading muon $p_{T}$ [GeV]',
    'subMuEta' : 'Subleading muon $\\eta$',
    'subMuCharge' : 'Subleading muon charge',
    'pdgid' : "Particle ID",
    'charge' : "Charge",
    'dxy' : "Transverse impact parameter [cm]",
    'dz' : "Longitudinal impact parameter [cm]",
    'puppiWeight' : "PUPPI weight",
    'fromPV' : 'From PV flag',
    'jetPt' : 'Jet $p_{T}$ [GeV]',
    'jetEta' : 'Jet $\\eta$',
    'jetPhi' : 'Jet $\\phi$',
    'pt_over_jetPt' : '$p_{T} / p_{T}^{jet}$',
}

logxs = {
    'pt' : True,
    'eta' : False,
    'phi'  : False,
    'nConstituents' : False,
    'nPassingParts' : False,
    'passLooseB' : False,
    'passMediumB' : False,
    'passTightB' : False,
    'nTrueInt' : False,
    'rho' : False,
    'MET' : True,
    'numLooseB' : False,
    'numMediumB' : False,
    'numTightB' : False,
    'numJets' : False,
    'Zpt' : True,
    'Zy' : False,
    'Zmass' : False,
    'leadMuPt' : True,
    'leadMuEta' : False,
    'leadMuCharge' : False,
    'subMuPt' : True,
    'subMuEta' : False,
    'subMuCharge' : False,
    'pdgid' : False,
    'charge' : False,
    'dxy' : False,
    'dz' : False,
    'puppiWeight' : False,
    'fromPV' : False,
    'jetPt' : True,
    'jetEta' : False,
    'jetPhi' : False,
    'pt_over_jetPt' : True,
}

logys = {
    'pt' : True,
    'eta' : False,
    'phi'  : False,
    'nConstituents' : True,
    'nPassingParts' : True,
    'passLooseB' : True,
    'passMediumB' : True,
    'passTightB' : True,
    'nTrueInt' : True,
    'rho' : True,
    'MET' : True,
    'numLooseB' : True,
    'numMediumB' : True,
    'numTightB' : True,
    'numJets' : True,
    'Zpt' : True,
    'Zy' : True,
    'Zmass' : True,
    'leadMuPt' : True,
    'leadMuEta' : True,
    'leadMuCharge' : False,
    'subMuPt' : True,
    'subMuEta' : True,
    'subMuCharge' : False,
    'pdgid' : True,
    'charge' : True,
    'dxy' : True,
    'dz' : True,
    'puppiWeight' : True,
    'fromPV' : True,
    'jetPt' : True,
    'jetEta' : True,
    'jetPhi' : True,
    'pt_over_jetPt' : True,
}

class KinPlotManager:
    def __init__(self):
        self.dfs_MC = []
        self.labels_MC = []
        self.dfs_data = None

    def add_MC(self, df, label):
        self.dfs_MC.append(df)
        self.labels_MC.append(label)

    def add_data(self, df):
        if self.dfs_data is None:
            self.dfs_data = [df]
        else:
            self.dfs_data.append(df)

    def plot_variable(self, toplot,
                      weightsyst='nominal', 
                      pulls=False, density=True):
        if type(toplot) is str:
            logx = logxs[toplot]
            logy = logys[toplot]
            xlabel = xlabels[toplot]
        elif type(toplot) is Variable:
            logx = logxs[toplot.name]
            logy = logys[toplot.name]
            xlabel = xlabels[toplot.name]
        elif type(toplot) is Ratio:
            thename = "%s_over_%s"%(toplot.num, toplot.denom)
            logx = logxs[thename]
            logy = logys[thename]
            xlabel = xlabels[thename]

        plot_variable(self.dfs_data, self.dfs_MC, self.labels_MC,
                      toplot, logx, logy, xlabel,
                      weightsyst=weightsyst,
                      pulls=pulls, density=density)


class Variable:
    def __init__(self, name):
        self.name = name

    @property
    def columns(self):
        return [self.name]

    def evaluate(self, x):
        return x
    

class Ratio:
    def __init__(self, num, denom):
        self.num = num
        self.denom = denom

    @property
    def columns(self):
        return [self.num, self.denom]

    def evaluate(self, x, y):
        return x.to_numpy()/y.to_numpy()

def plot_variable(dfs_data, dfs_MC, labels_MC, 
                  toplot, logx, logy, xlabel,
                  weightsyst='nominal',
                  pulls=False, density=True):

    if type(toplot) is str:
        toplot = Variable(toplot)

    if dfs_data is None:
        DO_DATA = False
    else:
        DO_DATA = True
        if type(dfs_data) not in [list,tuple]:
            dfs_data = [dfs_data]

    if type(dfs_MC) not in [list,tuple]:
        dfs_MC = [dfs_MC]
        labels_MC = [labels_MC]

    tables_MC = []
    for df in dfs_MC:
        tables_MC.append(
            df.to_table(columns=[*toplot.columns, 'evtwt_%s'%weightsyst])
        )

    if DO_DATA:
        tables_data = []
        for df in dfs_data:
            tables_data.append(
                df.to_table(columns=[*toplot.columns, 'evtwt_%s'%weightsyst])
            )


    vals_MC = []
    for table in tables_MC:
        toeval = [table[column] for column in toplot.columns]
        vals_MC.append(toplot.evaluate(*toeval))

    minvals_MC = [np.min(val) for val in vals_MC]
    maxvals_MC = [np.max(val) for val in vals_MC]

    minval = np.min(minvals_MC)
    maxval = np.max(maxvals_MC)

    if DO_DATA:
        vals_data = []
        for table in tables_data:
            toeval = [table[column] for column in toplot.columns]
            vals_data.append(toplot.evaluate(*toeval))

        minvals_data = [np.min(val) for val in vals_data]
        maxvals_data = [np.max(val) for val in vals_data]

        minval = np.min([minval, *minvals_data])
        maxval = np.max([maxval, *maxvals_data])

    if type(toplot) != Variable:
        theax = hist.axis.Regular(
            100, minval, maxval, 
            transform=hist.axis.transform.log if logx else None
        )

    else:
        dtype = tables_MC[0][toplot.columns[0]].type
        if dtype in [pa.float16(), pa.float32(), pa.float64()]:
            theax = hist.axis.Regular(
                100, minval, maxval, 
                transform=hist.axis.transform.log if logx else None
            )
        elif dtype in [pa.bool_(), pa.uint8(), pa.uint16(), pa.uint32(), pa.uint64(), pa.int8(), pa.int16(), pa.int32(), pa.int64()]:
            theax = hist.axis.Integer(minval, maxval+1)
            if logx:
                raise ValueError("logx is not supported for integer axes")
        else:
            raise ValueError("Unsupported column type: %s" % dtype)

    H = hist.Hist(
        theax,
        hist.axis.StrCategory([], name='label', growth=True),
        storage=hist.storage.Weight(),
    )

    for val, table, label in zip(vals_MC, tables_MC, labels_MC):
        H.fill(
            val,
            weight=table['evtwt_%s'%weightsyst],
            label=label,
        )

    if DO_DATA:
        for val, table in zip(vals_data, tables_data):
            H.fill(
                val,
                weight=table['evtwt_%s'%weightsyst],
                label='DATA'
            )

    fig = plt.figure(figsize=(12, 12))
    try:
        (ax_main, ax_ratio) = fig.subplots(2, 1, sharex=True, height_ratios=(1, 0.5))

        mainlines = {}
        for label in labels_MC:
            mainlines[label] = simon_histplot(
                                    H[{'label' : label}], 
                                    density=density, 
                                    label=label, ax=ax_main
                                ) 

        if DO_DATA:
            mainlines['DATA'] = simon_histplot(
                                    H[{'label' : 'DATA'}], 
                                    density=density, 
                                    label='DATA', c='k', ax=ax_main
                                ) 

        if logx:
            ax_main.set_xscale('log')
        if logy:
            ax_main.set_yscale('log')

        if density:
            ax_main.set_ylabel('Counts [denstiy]')
        else:
            ax_main.set_ylabel('Counts [a.u.]')

        ax_main.legend()

        if DO_DATA:
            for label in labels_MC:
                simon_histplot_ratio(
                    H[{'label' : 'DATA'}],
                    H[{'label' : label}],
                    density=density, ax=ax_ratio,
                    label=label,
                    color=mainlines[label][0].get_color(),
                    pulls=pulls,
                )
            if pulls:
                ax_ratio.set_ylabel("DATA/MC [pulls]")
            else:
                ax_ratio.set_ylabel("DATA/MC")

        else:
            nom = labels_MC[0]
            for label in labels_MC[1:]:
                simon_histplot_ratio(
                    H[{'label' : nom}],
                    H[{'label' : label}],
                    density=density, ax=ax_ratio,
                    label=label
                )
            if pulls:
                ax_ratio.set_ylabel("%s/MC [pulls]"%nom)
            else:
                ax_ratio.set_ylabel("%s/MC"%nom)

        ax_ratio.set_xlabel(xlabel)

        ax_ratio.axhline(1, color='black', linestyle='--')

        #the automatic axis ranges bug out for some reason
        #so we set them manually
        if not logx:
            if maxval.dtype == np.bool_:
                axrange = 1
            else:
                axrange = maxval - minval
            padded_range = axrange * 1.05
            axcenter = (minval + maxval) / 2
            newmin = axcenter - padded_range / 2
            newmax = axcenter + padded_range / 2
            ax_main.set_xlim(newmin, newmax)
        else:
            ax_main.set_xlim(0.95*minval, 1.05*maxval)

        plt.tight_layout()
        plt.subplots_adjust(wspace=0, hspace=0)

        plt.show()
    finally:
        plt.close(fig)



























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

