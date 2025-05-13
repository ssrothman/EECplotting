import matplotlib.pyplot as plt
import os
import os.path
import numpy as np
import mplhep as hep
from matplotlib.colors import Normalize, LogNorm
import hist
import pyarrow as pa
import json
from datasets import get_dataset, get_counts

from histplot import simon_histplot, simon_histplot_ratio

plt.style.use(hep.style.CMS)

with open("config/config.json", 'r') as f:
    config = json.load(f)

with open("config/datasets.json", 'r') as f:
    datasets = json.load(f)

#setup datasets
def setup_datasets_GENONLY(skimmer, which):
    GENONLYdsets = {}

    for key in datasets['DatasetsGENONLY'].keys():
        print("Building GENONLY dset %s"%key)
        entry = datasets['DatasetsGENONLY'][key]

        GENONLYdsets[entry['tag']] = PlottableDataset()
        GENONLYdsets[entry['tag']].setup_MC(
            df = get_dataset('May_04_2025', entry['tag'], skimmer, 'nominal', which),
            label = entry['label'],
            xsec = 1.0,
            numevts = 1.0,
            color = entry['color']
        )
    return GENONLYdsets

def setup_datasets_MC(skimmer, which):
    MCdsets = {}

    for key in datasets['DatasetsMC'].keys():
        print("Building MC dset %s"%key)
        entry = datasets['DatasetsMC'][key]

        MCdsets[entry['tag']] = PlottableDataset()
        MCdsets[entry['tag']].setup_MC(
            df = get_dataset('Apr_23_2025', entry['tag'], skimmer, 'nominal', which),
            label = entry['label'],
            xsec = entry['xsec'],
            numevts = get_counts('Apr_23_2025', entry['tag']),
            color = entry['color']
        )

    stackDoneMC = np.asarray([False] * len(datasets['StacksMC'].keys()))
    while(np.any(~stackDoneMC)):
        for i, key in enumerate(datasets['StacksMC'].keys()):
            if stackDoneMC[i]:
                continue

            entry = datasets['StacksMC'][key]

            canDo = True
            for stackTag in entry['stacks']:
                if stackTag not in MCdsets.keys():
                    print("Can't build MC stack %s because %s is not available"%(key, stackTag))
                    canDo = False
                    break

            if not canDo:
                continue
            
            print("Building MC stack %s"%key)
            MCdsets[entry['tag']] = PlottableDatasetStack()
            MCdsets[entry['tag']].setup_MC(
                datasets = [MCdsets[tag] for tag in entry['dsets'] + entry['stacks']],
                global_label = entry['global_label'],
                global_color = entry['global_color'],
                plot_resolved = False
            )
            stackDoneMC[i] = True

    return MCdsets

def setup_datasets_DATA(skimmer, which):
    DATAdsets = {}

    for key in datasets['DatasetsDATA'].keys():
        print("Building DATA dset %s"%key)
        entry = datasets['DatasetsDATA'][key]

        DATAdsets[entry['tag']] = PlottableDataset()
        DATAdsets[entry['tag']].setup_data(
            df = get_dataset('Apr_23_2025', entry['tag'], skimmer, 'nominal', which),
            label = entry['label'],
            lumi = entry['lumi'],
            color = entry['color'],
        )

    stackDoneData = np.asarray([False] * len(datasets['StacksDATA'].keys()))
    while(np.any(~stackDoneData)):
        for i, key in enumerate(datasets['StacksDATA'].keys()):
            if stackDoneData[i]:
                continue

            entry = datasets['StacksDATA'][key]

            canDo = True
            for stackTag in entry['stacks']:
                if stackTag not in DATAdsets.keys():
                    print("Can't build DATA stack %s because %s is not available"%(key, stackTag))
                    canDo = False
                    break

            if not canDo:
                continue

            print("Building DATA stack %s"%key)
            DATAdsets[entry['tag']] = PlottableDatasetStack()
            DATAdsets[entry['tag']].setup_data(
                datasets = [DATAdsets[tag] for tag in entry['dsets'] + entry['stacks']],
                global_label = entry['global_label'],
                global_color = entry['global_color'],
                plot_resolved = False
            )
            stackDoneData[i] = True

    return DATAdsets

def variable_from_string(name):
    if 'over' in name:
        num, denom = name.split('_over_')
        return Ratio(num, denom)
    elif 'times' in name:
        var1, var2 = name.split('_times_')
        return Product(var1, var2)
    else:
        return Variable(name)

class Variable:
    def __init__(self, name):
        self.name = name

    @property
    def columns(self):
        return [self.name]

    def evaluate(self, table):
        return table[self.name].to_numpy()

    @property
    def key(self):
        return self.name
    
class Ratio:
    def __init__(self, num, denom):
        self.num = num
        self.denom = denom

    @property
    def columns(self):
        return [self.num, self.denom]

    def evaluate(self, table):
        return table[self.num].to_numpy()/table[self.denom].to_numpy()

    @property
    def key(self):
        return "%s_over_%s"%(self.num, self.denom)

class Product:
    def __init__(self, var1, var2):
        self.var1 = var1
        self.var2 = var2

    @property
    def columns(self):
        return [self.var1, self.var2]

    def evaluate(self, table):
        return table[self.var1].to_numpy() * table[self.var2].to_numpy()

    @property
    def key(self):
        return "%s_times_%s"%(self.var1, self.var2)

class NoCut:
    def __init__(self):
        pass

    @property
    def columns(self):
        return []

    def evaluate(self, table):
        return np.ones(table.num_rows, dtype=bool)

    @property
    def key(self):
        return "none"

class TwoSidedCut:
    def __init__(self, variable, low, high):
        self.low = low
        self.high = high
        if type(variable) is str:
            self.variable = variable_from_string(variable)
        else:
            self.variable = variable

    @property
    def columns(self):
        return self.variable.columns

    def evaluate(self, table):
        ev = self.variable.evaluate(table)
        return np.logical_and(
            ev >= self.low,
            ev < self.high
        )

    @property
    def key(self):
        return "%sLT%gGT%g"%(self.variable.name, self.high, self.low)

class GreaterThanCut:
    def __init__(self, variable, value):
        self.value = value
        if type(variable) is str:
            self.variable = variable_from_string(variable)
        else:
            self.variable = variable

    @property
    def columns(self):
        return self.variable.columns

    def evaluate(self, table):
        ev = self.variable.evaluate(table)
        return ev >= self.value

    @property
    def key(self):
        return "%sGT%g"%(self.variable.name, self.value)

class LessThanCut:
    def __init__(self, variable, value):
        self.value = value
        if type(variable) is str:
            self.variable = variable_from_string(variable)
        else:
            self.variable = variable

    @property
    def columns(self):
        return self.variable.columns

    def evaluate(self, table):
        ev = self.variable.evaluate(table)
        return ev < self.value

    @property
    def key(self):
        return "%sLT%g"%(self.variable.name, self.value)

class PlottableDataset:
    def _init__():
        pass

    def setup_MC(self, df, label, xsec, numevts, color):
        self.df = df
        self.label = label
        self.xsec = xsec
        self.numevts = numevts
        self.color = color
        self.isMC = True

    def setup_data(self, df, label, lumi, color):
        self.df = df
        self.label = label
        self.lumi = lumi
        self.color = color
        self.isdata = True
        self.samplewt = 1

    def evaluate_table(self, toplot, cut, weighting):
        needed_columns = list(set(toplot.columns +
                                  cut.columns +
                                  weighting.columns))

        self.table = self.df.to_table(columns=needed_columns)

        self.mask = cut.evaluate(self.table)
        self.vals = toplot.evaluate(self.table)[self.mask]
        self.wts = weighting.evaluate(self.table)[self.mask]

        self.minval = np.min(self.vals)
        self.maxval = np.max(self.vals)

    def set_samplewt_MC(self, total_lumi):
        self.samplewt = total_lumi * self.xsec * 1000
        self.samplewt /= self.numevts

    def fill_hist(self, global_min, global_max, nbins, logx):
        if self.vals.dtype in [np.bool_, np.int16, np.int32, np.int64,
                               np.uint16, np.uint32, np.uint64]:

            theax = hist.axis.Integer(global_min, global_max+1)
            if logx:
                raise ValueError("logx is not supported for integer axes")
        else:
            theax = hist.axis.Regular(
                    nbins, global_min, global_max,
                    transform=hist.axis.transform.log if logx else None
            )
        self.H = hist.Hist(
            theax,
            storage=hist.storage.Weight(),
        )
        self.H.fill(
            self.vals,
            weight=self.wts * self.samplewt,
        )

    def plot(self, density, ax):
        return simon_histplot(self.H, density=density, 
                              label=self.label, 
                              ax=ax, color=self.color)[0]

    def plot_ratio(self, other, density, pulls, ax):
        return simon_histplot_ratio(self.H, other.H,
                                    density = density, 
                                    ax = ax, 
                                    label = other.label,
                                    color = other.color,
                                    pulls = pulls)

    def estimate_yield(self):
        return np.sum(self.df.to_table(columns = ["evtwt_nominal"])) * self.samplewt

class PlottableDatasetStack:
    def __init__(self):
        pass

    def sort_by_yield(self):
        for dset in self.datasets:
            dset.set_samplewt_MC(1)

        self.datasets = sorted(self.datasets, key=lambda dset : dset.estimate_yield())
        for dset in self.datasets:
            dset.samplewt = None

    def setup_MC(self, datasets, global_label, global_color, plot_resolved):
        self.datasets = datasets

        self.label = global_label
        self.color = global_color
        self.plot_resolved = plot_resolved
        self.xsec = np.max([dset.xsec for dset in self.datasets])

        if plot_resolved:
            self.sort_by_yield()

    #def setup_MC(self, dfs, labels, xsecs, numevts, colors,
    #             global_label, global_color, plot_resolved):
    #    self.datasets = []
    #    for df, label, xsec, numevt, color in zip(dfs, labels, xsecs, numevts, colors):
    #        if type(df) in [list, tuple]:
    #            newdf = PlottableDatasetStack()
    #            newdf.setup_MC(df, label, xsec, numevt, color,
    #                           label[-1], color[-1], False)
    #        else:
    #            newdf = PlottableDataset()
    #            newdf.setup_MC(df, label, xsec, numevt, color)
    #        self.datasets.append(newdf)

    #    self.label = global_label
    #    self.color = global_color
    #    self.plot_resolved = plot_resolved

    #    self.xsec = np.sum([dset.xsec for dset in self.datasets])

    #    if plot_resolved:
    #        self.sort_by_yeild()

    def setup_data(self, datasets, global_label, global_color, plot_resolved):
        self.datasets = datasets

        self.label = global_label
        self.color = global_color
        self.plot_resolved = plot_resolved

        self.lumi = np.sum([dset.lumi for dset in self.datasets])

    #def setup_data(self, dfs, labels, lumis, colors,
    #               global_label, global_color, plot_resolved):
    #    self.datasets = []
    #    self.lumi = 0
    #    for df, label, lumi, color in zip(dfs, labels, lumis, colors):
    #        self.lumi += lumi
    #        newdf = PlottableDataset()
    #        newdf.setup_data(df, label, lumi, color)
    #        self.datasets.append(newdf)

    #    self.label = global_label 
    #    self.color = global_color
    #    self.plot_resolved = plot_resolved

    def evaluate_table(self, toplot, cut, weighting):
        minvals = []
        maxvals = []
        for dset in self.datasets:
            dset.evaluate_table(toplot, cut, weighting)
            minvals.append(dset.minval)
            maxvals.append(dset.maxval)

        self.minval = np.min(minvals)
        self.maxval = np.max(maxvals)

    def set_samplewt_MC(self, total_lumi):
        for dset in self.datasets:
            dset.set_samplewt_MC(total_lumi)

    def fill_hist(self, global_min, global_max, nbins, logx):
        for dset in self.datasets:
            dset.fill_hist(global_min, global_max, nbins, logx)
    
        self.H = self.datasets[0].H.copy()
        for dset in self.datasets[1:]:
            self.H += dset.H

    def plot(self, density, ax): 
        if self.plot_resolved:
            artists = []
            fillbetween = 0
            for dset in self.datasets:
                artist, nextfill = simon_histplot(
                    dset.H, density=False,
                    label = dset.label,
                    ax = ax, color=dset.color,
                    fillbetween = fillbetween
                )
                fillbetween = nextfill
                artists.append(artist)
            return artists
        else:
            return simon_histplot(self.H, density=density,
                                  label = self.label,
                                  ax = ax, color=self.color)[0]

    def plot_ratio(self, other, density, pulls, ax):
        print("SELF H SUM")
        print(self.H.sum())
        print()
        return simon_histplot_ratio(self.H, other.H,
                                    density=density,
                                    ax =ax,
                                    label = other.label,
                                    color = other.color,
                                    pulls = pulls)

    def estimate_yield(self):
        return np.sum([dset.estimate_yield() for dset in self.datasets])

class KinPlotManager:
    def __init__(self):
        self.dfs_MC = []

        self.df_data = None

        self.folder = None
        self.show=True
        self.isCMS = True

    def set_CMS(self, value):
        self.isCMS = value

    def clear(self):
        self.dfs_MC = []
        self.df_data = None

    def add_MC(self, dset):
        self.dfs_MC.append(dset)

    def add_data(self, dset):
        self.df_data = dset

    def setup_savefig(self, folder):
        self.folder = folder
        os.makedirs(folder, exist_ok=True)

    def toggle_show(self, show):
        self.show = show

    def plot_variable(self, toplot, cut=NoCut(),
                      weighting='evtwt_nominal', 
                      pulls=False, density=False,
                      noResolved=False):
    
        numMCstacks = 0
        for dset in self.dfs_MC:
            if type(dset) == PlottableDatasetStack:
                numMCstacks += 1

        if (numMCstacks == 1) and (not noResolved):
            for dset in self.dfs_MC:
                if type(dset) == PlottableDatasetStack:
                    dset.plot_resolved = True
        else:
            for dset in self.dfs_MC:
                if type(dset) == PlottableDatasetStack:
                    dset.plot_resolved = False

        if type(toplot) is str:
            toplot = variable_from_string(toplot)

        if type(weighting) is str:
            weighting = variable_from_string(weighting)

        if type(toplot) is Variable:
            thename = toplot.name
        elif type(toplot) is Ratio:
            thename = "%s_over_%s"%(toplot.num, toplot.denom)
        else:
            raise ValueError("toplot must be a string or a Variable or Ratio object")

        logx = config['KinAxes'][toplot.name]['logx']
        logy = config['KinAxes'][toplot.name]['logy']
        nbins = config['KinAxes'][toplot.name]['nbins']
        xlabel = config['KinAxes'][toplot.name]['label']

        if self.folder is not None:
            thename = 'VAR-%s_WT-%s_CUT-%s_DENSITY-%d_PULL-%d.png' % (
                toplot.key, weighting.key, cut.key,
                density, pulls
            )
            savefig = os.path.join(self.folder, thename)
        else:
            savefig = None

        plot_variable(self.df_data, self.dfs_MC,
                      toplot, logx, logy, nbins, xlabel,
                      weighting=weighting,
                      cut=cut,
                      pulls=pulls, density=density,
                      savefig=savefig, show=self.show,
                      isCMS = self.isCMS)

def plot_variable(dataset_data,
                  datasets_MC,
                  toplot, logx, logy, nbins, xlabel,
                  weighting=Variable('evtwt_nominal'),
                  cut = NoCut(),
                  pulls=False, density=True,
                  savefig=None, show=True,
                  isCMS=True):
    print("Top of plot_variable()")

    if dataset_data is None:
        DO_DATA = False
    else:
        DO_DATA = True

    if type(datasets_MC) not in [list,tuple]:
        datasets_MC = [datasets_MC]

    for dMC in datasets_MC:
        dMC.evaluate_table(toplot, cut, weighting)

    if DO_DATA:
        dataset_data.evaluate_table(toplot, cut, weighting)

    global_min = np.min([dMC.minval for dMC in datasets_MC])
    global_max = np.max([dMC.maxval for dMC in datasets_MC])

    if DO_DATA:
        global_min = np.min([global_min, dataset_data.minval])
        global_max = np.max([global_max, dataset_data.maxval])

    print("evaluated tables")

    for dMC in datasets_MC:
        if DO_DATA:
            dMC.set_samplewt_MC(dataset_data.lumi)
        else:
            dMC.set_samplewt_MC(1.0)

        dMC.fill_hist(global_min, global_max, nbins, logx)

    if DO_DATA:
        dataset_data.fill_hist(global_min, global_max, nbins, logx)

    print("Filled hists")

    fig = plt.figure(figsize=config['Figure_Size'])
    try:
        (ax_main, ax_ratio) = fig.subplots(
                2, 1, sharex=True, 
                height_ratios=(1, config['Ratiopad_Height'])
        )

        if isCMS:
            if DO_DATA:
                hep.cms.label(ax=ax_main, data=True, label=config['Approval_Text'],
                              year=config['Year'], lumi=dataset_data.lumi)
            else:
                hep.cms.label(ax=ax_main, data=False, label=config['Approval_Text'])

        mainlines = {}
        for dMC in datasets_MC:
            dMC.plot(density, ax=ax_main)

        if DO_DATA:
            dataset_data.plot(density, ax=ax_main)

        if logx:
            ax_main.set_xscale('log')
        if logy:
            ax_main.set_yscale('log')

        if density:
            ax_main.set_ylabel('Counts [denstiy]')
        else:
            ax_main.set_ylabel('Counts [a.u.]')

        ax_main.legend(loc='best')

        if DO_DATA:
            for dMC in datasets_MC:
                dataset_data.plot_ratio(dMC, density, pulls, ax_ratio)

            if pulls:
                ax_ratio.set_ylabel("DATA/MC [pulls]")
            else:
                ax_ratio.set_ylabel("DATA/MC")

        else:
            nom = datasets_MC[0]
            for dMC in datasets_MC[1:]:
                nom.plot_ratio(dMC, density, pulls, ax_ratio)

            if pulls:
                ax_ratio.set_ylabel("%s/MC [pulls]"%nom.label)
            else:
                ax_ratio.set_ylabel("%s/MC"%nom.label)

        ax_ratio.set_xlabel(xlabel)

        ax_ratio.axhline(1, color='black', linestyle='--')

        #ax_ratio.set_ylim(0.0, 2.0)

        #the automatic axis ranges bug out for some reason
        #so we set them manually
        if not logx:
            if global_max.dtype == np.bool_:
                axrange = 1
            else:
                axrange = global_max - global_min
            padded_range = axrange * 1.05
            axcenter = (global_min + global_max) / 2
            newmin = axcenter - padded_range / 2
            newmax = axcenter + padded_range / 2
            ax_main.set_xlim(newmin, newmax)
        else:
            ax_main.set_xlim(0.95*global_min, 1.05*global_max)

        plt.tight_layout()
        plt.subplots_adjust(wspace=0, hspace=0)

        if savefig is not None:
            plt.savefig(savefig, dpi=300, bbox_inches='tight', format='png')

        if show:
            plt.show()
    finally:
        plt.close(fig)
