import matplotlib.pyplot as plt
from util import is_integral, make_ax
import os
import os.path
import numpy as np
import mplhep as hep
from matplotlib.colors import Normalize, LogNorm
import hist
import pyarrow as pa
import json
from datasets import get_dataset, get_procpkl

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
            numevts = get_procpkl('Apr_23_2025', entry['tag'], 'Count'),
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
    
class CorrectionlibVariable:
    def __init__(self, var_l, path, key):
        self.var_l = []
        for var in var_l:
            if type(var) is str:
                self.var_l.append(variable_from_string(var))
            else:
                self.var_l.append(var)

        from correctionlib import CorrectionSet
        cset = CorrectionSet.from_file(path)
        if key not in list(cset.keys()):
            print("Error: Correctionlib key '%s' not found in %s"%(key, path))
            print("Available keys: %s"%list(cset.keys()))
            raise ValueError("Correctionlib key not found")
        self.eval = cset[key].evaluate
        self.csetkey = key

    @property
    def columns(self):
        cols = []
        for var in self.var_l:
            cols += var.columns
        return list(set(cols))

    def evaluate(self, table):
        args = []
        for var in self.var_l:
            args.append(var.evaluate(table))

        return self.eval(*args)

    @property
    def key(self):
        return "CORRECTIONLIB(%s)"%(self.csetkey)

class UFuncVariable:
    def __init__(self, name, ufunc):
        self.name = name
        self.ufunc = ufunc

    @property
    def columns(self):
        return [self.name]

    def evaluate(self, table):
        return self.ufunc(table[self.name].to_numpy())

    @property
    def key(self):
        return 'UFUNC%s(%s)'%(self.ufunc.__name__, self.name)

class RateVariable:
    def __init__(self, binaryfield, wrt):
        if type(binaryfield) is str:
            self.binaryfield = Variable(binaryfield)
        else:
            self.binaryfield = binaryfield

        if type(wrt) is str:
            self.wrt = Variable(wrt)
        else:
            self.wrt = wrt

    @property
    def columns(self):
        return self.binaryfield.columns + self.wrt.columns

    def evaluate(self, table):
        return [self.binaryfield.evaluate(table),
                self.wrt.evaluate(table)]

    @property
    def key(self):
        return "%s_rate_wrt_%s"%(self.binaryfield.key, self.wrt.key)

class ResolutionVariable:
    def __init__(self, gen, reco):
        self.gen = gen
        self.reco = reco

    @property
    def columns(self):
        return [self.gen, self.reco]

    def evaluate(self, table):
        return table[self.reco].to_numpy() - table[self.gen].to_numpy() 

    @property
    def key(self):
        return "%s_minus_%s"%(self.reco, self.gen)

class RelativeResolutionVariable:
    def __init__(self, gen, reco):
        self.gen = gen
        self.reco = reco

    @property
    def columns(self):
        return [self.gen, self.reco]

    def evaluate(self, table):
        gen = table[self.gen].to_numpy()
        reco = table[self.reco].to_numpy()
        return (reco - gen) / gen

    @property
    def key(self):
        return "%s_minus_%s_over_%s"%(self.reco, self.gen, self.gen)

class Ratio:
    def __init__(self, num, denom):
        if type(num) is str:
            self.num = Variable(num)
        else:
            self.num = num

        if type(denom) is str:
            self.denom = Variable(denom)
        else:
            self.denom = denom

    @property
    def columns(self):
        return list(set(self.num.columns + self.denom.columns))

    def evaluate(self, table):
        return self.num.evaluate(table) / self.denom.evaluate(table)

    @property
    def key(self):
        return "%s_over_%s"%(self.num.key, self.denom.key)

class Product:
    def __init__(self, var1, var2):
        if type(var1) is str:
            self.var1 = Variable(var1)
        else:
            self.var1 = var1

        if type(var2) is str:
            self.var2 = Variable(var2)
        else:
            self.var2 = var2

    @property
    def columns(self):
        return self.var1.columns + self.var2.columns

    def evaluate(self, table):
        return self.var1.evaluate(table) * self.var2.evaluate(table)

    @property
    def key(self):
        return "%s_times_%s"%(self.var1.key, self.var2.key)

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

    @property
    def plottext(self):
        return "Inclusive"

class EqualsCut:
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
        return ev == self.value

    @property
    def key(self):
        return "%sEQ%g"%(self.variable.key, self.value)

    @property
    def plottext(self):
        return "%s$ = %g$"%(config['KinAxes'][self.variable.key]['label'], 
                            self.value)

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
        return "%sLT%gGT%g"%(self.variable.key, self.high, self.low)

    @property
    def plottext(self):
        return "$%g \\leq $%s$ < %g$"%(
                self.low,
                config['KinAxes'][self.variable.key]['label'], 
                self.high)
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
        return "%sGT%g"%(self.variable.key, self.value)

    @property
    def plottext(self):
        return "%s$ \\geq %g$"%(config['KinAxes'][self.variable.key]['label'], 
                            self.value)
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
        return "%sLT%g"%(self.variable.key, self.value)

    @property
    def plottext(self):
        return "%s$ < %g$"%(config['KinAxes'][self.variable.key]['label'], 
                            self.value)

class AndCuts:
    def __init__(self, *cuts):
        self.cuts = cuts

    @property
    def columns(self):
        cols = []
        for cut in self.cuts:
            cols += cut.columns
        return list(set(cols))

    def evaluate(self, table):
        mask = self.cuts[0].evaluate(table)
        for cut in self.cuts[1:]:
            mask = np.logical_and(mask, cut.evaluate(table))
        return mask

    @property
    def key(self):
        result = self.cuts[0].key
        for cut in self.cuts[1:]:
            result += "_AND_" + cut.key
        return result

    @property
    def plottext(self):
        result = self.cuts[0].plottext
        for cut in self.cuts[1:]:
            result += '\n' + cut.plottext
        return result

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

    def evaluate_table_Nd(self, toplot_l, cut, weighting):
        needed_columns = set(cut.columns + weighting.columns)
        for toplot in toplot_l:
            needed_columns.update(toplot.columns)
        needed_columns = list(needed_columns)

        self.table = self.df.to_table(columns=needed_columns)

        self.mask = cut.evaluate(self.table)
        self.vals = []
        for toplot in toplot_l:
            vals = toplot.evaluate(self.table)
            vals = vals[self.mask]
            self.vals.append(vals)
        self.wts = weighting.evaluate(self.table)[self.mask]

        if self.mask.sum() == 0:
            self.minval = -np.inf
            self.maxval = np.inf
        else:
            self.minval = np.asarray([np.min(vals) for vals in self.vals])
            self.maxval = np.asarray([np.max(vals) for vals in self.vals])

    def evaluate_table_rate(self, toplot, cut, weighting):
        needed_columns = list(set(toplot.columns +
                                  cut.columns + 
                                  weighting.columns))

        self.table = self.df.to_table(columns=needed_columns)

        self.mask = cut.evaluate(self.table)
        self.vals = toplot.evaluate(self.table)
        self.vals[0] = self.vals[0][self.mask]
        self.vals[1] = self.vals[1][self.mask]
        self.wts = weighting.evaluate(self.table)[self.mask]

        if self.mask.sum() == 0:
            self.minval = -np.inf
            self.maxval = np.inf
        else:
            self.minval = np.min(self.vals[1])
            self.maxval = np.max(self.vals[1])

    def evaluate_table(self, toplot, cut, weighting):
        needed_columns = list(set(toplot.columns +
                                  cut.columns +
                                  weighting.columns))

        self.table = self.df.to_table(columns=needed_columns)

        self.mask = cut.evaluate(self.table)
        self.vals = toplot.evaluate(self.table)[self.mask]
        self.wts = weighting.evaluate(self.table)[self.mask]

        if self.mask.sum() == 0:
            self.minval = -np.inf
            self.maxval = np.inf
        else:
            self.minval = np.min(self.vals)
            self.maxval = np.max(self.vals)

    def set_samplewt_MC(self, total_lumi):
        self.samplewt = total_lumi * self.xsec * 1000
        self.samplewt /= self.numevts

    def fill_hist_Nd(self, global_min, global_max, nbins, logx):
        theaxes = []
        for val, gmin, gmax, nbins, lx in zip(self.vals, global_min, 
                                                global_max, nbins, logx):
            theaxes.append(make_ax(val.dtype, gmin, gmax, nbins, lx))

        self.H = hist.Hist(
            *theaxes,
            storage=hist.storage.Weight()
        )

        self.H.fill(
            *self.vals,
            weight = self.wts * self.samplewt
        )

    def fill_hist_rate(self, global_min, global_max, nbins, logx):
        theax1 = make_ax(self.vals[0].dtype, 
                         0, 1,
                         None, False)

        theax2 = make_ax(self.vals[1].dtype, 
                         global_min, global_max, 
                         nbins, logx)

        self.H = hist.Hist(
            theax1,
            theax2,
            storage=hist.storage.Weight()
        )

        self.H.fill(
            self.vals[0],
            self.vals[1],
            weight = self.wts * self.samplewt
        )

    def fill_hist(self, global_min, global_max, nbins, logx):
        theax = make_ax(self.vals.dtype, global_min, global_max, nbins, logx)
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

    def plot_rate(self, ax):
        return simon_histplot_rate(self.H, label=self.label,
                                   ax=ax, color=self.color)

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

    def setup_data(self, datasets, global_label, global_color, plot_resolved):
        self.datasets = datasets

        self.label = global_label
        self.color = global_color
        self.plot_resolved = plot_resolved

        self.lumi = np.sum([dset.lumi for dset in self.datasets])

    def evaluate_table_Nd(self, toplot_l, cut, weighting):
        minvals = []
        maxvals = []
        for dset in self.datasets:
            dset.evaluate_table_Nd(toplot_l, cut, weighting)
            minvals.append(dset.minval)
            maxvals.append(dset.maxval)

        self.minval = np.min(minvals, axis=0)
        self.maxval = np.max(maxvals, axis=0)

    def evaluate_table_rate(self, toplot, cut, weighting):
        minvals = []
        maxvals = []
        for dset in self.datasets:
            dset.evaluate_table_rate(toplot, cut, weighting)
            minvals.append(dset.minval)
            maxvals.append(dset.maxval)

        self.minval = np.min(minvals)
        self.maxval = np.max(maxvals)

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

    def fill_hist_Nd(self, global_min, global_max, nbins, logx):
        for dset in self.datasets:
            dset.fill_hist_Nd(global_min, global_max, nbins, logx)

        self.H = self.datasets[0].H.copy()
        for dset in self.datasets[1:]:
            self.H += dset.H

    def fill_hist_rate(self, global_min, global_max, nbins, logx):
        for dset in self.datasets:
            dset.fill_hist_rate(global_min, global_max, nbins, logx)

        self.H = self.datasets[0].H.copy()
        for dset in self.datasets[1:]:
            self.H += dset.H

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
        return simon_histplot_ratio(self.H, other.H,
                                    density=density,
                                    ax =ax,
                                    label = other.label,
                                    color = other.color,
                                    pulls = pulls)

    def plot_rate(self, ax):
        if self.plot_resolved:
            raise ValueError("Rate variables not supported in resolved stacks")
        return simon_histplot_rate(self.H, label=self.label,
                                   ax = ax, color=self.color)

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
                      weighting_MC='evtwt_nominal', 
                      weighting_data='evtwt_nominal',
                      pulls=False, density=False,
                      clamp_ratiopad=None,
                      force_xlim=None,
                      cut_text = False,
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

        if type(weighting_MC) is str:
            weighting_MC = variable_from_string(weighting_MC)

        if type(weighting_data) is str:
            weighting_data = variable_from_string(weighting_data)

        if type(toplot) is RateVariable:
            logx = config['KinAxes'][toplot.wrt.key]['logx']
            logy = config['KinAxes'][toplot.wrt.key]['logy']
            nbins = config['KinAxes'][toplot.wrt.key]['nbins']
            xlabel = config['KinAxes'][toplot.wrt.key]['label']
        else:
            logx = config['KinAxes'][toplot.key]['logx']
            logy = config['KinAxes'][toplot.key]['logy']
            nbins = config['KinAxes'][toplot.key]['nbins']
            xlabel = config['KinAxes'][toplot.key]['label']

        if self.folder is not None:
            thename = 'VAR-%s_WT-%s_CUT-%s_DENSITY-%d_PULL-%d.png' % (
                toplot.key, weighting_MC.key, cut.key,
                density, pulls
            )
            savefig = os.path.join(self.folder, thename)
        else:
            savefig = None

        no_ratiopad = (len(self.dfs_MC) == 0 and self.df_data is None) or \
                      (len(self.dfs_MC) == 1 and self.df_data is None)

        vlines = []
        hlines = []
        if type(toplot) in [ResolutionVariable, RelativeResolutionVariable]:
            vlines += [0]
        elif type(toplot) is RateVariable:
            hlines += [0, 1]
            logy = False

        plot_variable(self.df_data, self.dfs_MC,
                      toplot, logx, logy, nbins, xlabel,
                      weighting_MC=weighting_MC,
                      weighting_data=weighting_data,
                      cut=cut,
                      pulls=pulls, density=density,
                      savefig=savefig, show=self.show,
                      clamp_ratiopad = clamp_ratiopad,
                      no_ratiopad = no_ratiopad,
                      force_xlim = force_xlim,
                      vlines=vlines,
                      hlines=hlines,
                      cut_text = cut_text,
                      isCMS = self.isCMS)

def plot_variable(dataset_data,
                  datasets_MC,
                  toplot, logx, logy, nbins, xlabel,
                  weighting_MC=Variable('evtwt_nominal'),
                  weighting_data=Variable('evtwt_nominal'),
                  cut = NoCut(),
                  pulls=False, density=True,
                  savefig=None, show=True,
                  clamp_ratiopad=None,
                  no_ratiopad = False,
                  force_xlim = None,
                  vlines = [],
                  hlines = [],
                  cut_text = False,
                  isCMS=True):
    print("Top of plot_variable()")

    if dataset_data is None:
        DO_DATA = False
    else:
        DO_DATA = True
    
    if type(toplot) is RateVariable:
        IS_RATE = True
    else:
        IS_RATE = False

    if type(datasets_MC) not in [list,tuple]:
        datasets_MC = [datasets_MC]

    for dMC in datasets_MC:
        if IS_RATE:
            dMC.evaluate_table_rate(toplot, cut, weighting_MC)
        else:
            dMC.evaluate_table(toplot, cut, weighting_MC)

    if DO_DATA:
        if IS_RATE:
            dataset_data.evaluate_table_rate(toplot, cut, weighting_data)
        else:
            dataset_data.evaluate_table(toplot, cut, weighting_data)

    global_min = np.min([dMC.minval for dMC in datasets_MC])
    global_max = np.max([dMC.maxval for dMC in datasets_MC])

    if DO_DATA:
        global_min = np.min([global_min, dataset_data.minval])
        global_max = np.max([global_max, dataset_data.maxval])

    if force_xlim is not None:
        global_min = np.asarray(force_xlim[0])
        global_max = np.asarray(force_xlim[1])

    print("evaluated tables")

    for dMC in datasets_MC:
        if DO_DATA:
            dMC.set_samplewt_MC(dataset_data.lumi)
        else:
            dMC.set_samplewt_MC(1.0)

        if IS_RATE:
            dMC.fill_hist_rate(global_min, global_max, nbins, logx)
        else:
            dMC.fill_hist(global_min, global_max, nbins, logx)

    if DO_DATA:
        if IS_RATE:
            dataset_data.fill_hist_rate(global_min, global_max, nbins, logx)
        else:
            dataset_data.fill_hist(global_min, global_max, nbins, logx)

    print("Filled hists")

    fig = plt.figure(figsize=config['Figure_Size'])
    try:
        if no_ratiopad:
            ax_main = fig.subplots(1, 1)
        else:
            (ax_main, ax_ratio) = fig.subplots(
                    2, 1, sharex=True, 
                    height_ratios=(1, config['Ratiopad_Height'])
            )

        if isCMS:
            if DO_DATA:
                hep.cms.label(ax=ax_main, data=True, label=config['Approval_Text'],
                              year=config['Year'], lumi='%0.2f'%dataset_data.lumi)
            else:
                hep.cms.label(ax=ax_main, data=False, label=config['Approval_Text'])

        for dMC in datasets_MC:
            if IS_RATE:
                dMC.plot_rate(ax=ax_main)
            else:
                dMC.plot(density, ax=ax_main)

        if DO_DATA:
            if IS_RATE:
                dataset_data.plot_rate(ax=ax_main)
            else:
                dataset_data.plot(density, ax=ax_main)

        if logx:
            ax_main.set_xscale('log')
        if logy:
            ax_main.set_yscale('log')

        if density:
            ax_main.set_ylabel('Counts [denstiy]')
        else:
            ax_main.set_ylabel('Counts [a.u.]')

        for vline in vlines:
            ax_main.axvline(vline, color='black', linestyle='--')
        for hline in hlines:
            ax_main.axhline(hline, color='black', linestyle='--')

        if cut_text:
            ax_main.text(
                0.05, 0.95, cut.plottext,
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax_main.transAxes,
                fontsize=24, 
                bbox={
                    'boxstyle': 'round,pad=0.3',
                    'facecolor': 'white',
                    'edgecolor': 'black',
                    'alpha': 0.8
                }
            )

        if not no_ratiopad:
            ax_main.legend(loc='best')

            if DO_DATA:
                for dMC in datasets_MC:
                    if IS_RATE:
                        raise NotImplementedError("Rate variables not supported in ratio plots")
                    else:
                        dataset_data.plot_ratio(dMC, density, pulls, ax_ratio)

                if pulls:
                    ax_ratio.set_ylabel("DATA/MC [pulls]")
                else:
                    ax_ratio.set_ylabel("DATA/MC")

            else:
                nom = datasets_MC[0]
                for dMC in datasets_MC[1:]:
                    if IS_RATE:
                        raise NotImplementedError("Rate variables not supported in ratio plots")
                    else:
                        nom.plot_ratio(dMC, density, pulls, ax_ratio)

                if pulls:
                    ax_ratio.set_ylabel("%s/MC [pulls]"%nom.label)
                else:
                    ax_ratio.set_ylabel("%s/MC"%nom.label)

            ax_ratio.set_xlabel(xlabel)

            ax_ratio.axhline(1, color='black', linestyle='--')

            if clamp_ratiopad is not None:
                ax_ratio.set_ylim(clamp_ratiopad[0], clamp_ratiopad[1])
        else:
            ax_main.set_xlabel(xlabel)

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
