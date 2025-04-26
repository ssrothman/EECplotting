import matplotlib.pyplot as plt
import os
import os.path
import numpy as np
import mplhep as hep
from matplotlib.colors import Normalize, LogNorm
import hist
import pyarrow as pa
import json

from histplot import simon_histplot, simon_histplot_ratio

plt.style.use(hep.style.CMS)

with open("config.json", 'rb') as f:
    config = json.load(f)

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
    def __init__(df, label, xsec, numevts, color):
        self.df = df
        self.label = label
        self.xsec = xsec
        self.numevts = numevts
        self.color = color

    def evaluate_table(self, toplot, cut, weighting):
        needed_columns = list(set(toplot.columns +
                                  cut.columns +
                                  weighting.columns))

        self.table = self.df.to_table(columns=needed_columns)

        self.mask = cut.evaluate(self.table)
        self.vals = toplot.evaluate(self.table)[self.mask]
        self.wts = weighting.evaluate(self.table)[self.mask]

        self.minval = ak.min(self.vals)
        self.maxval = ak.max(self.vals)

    def set_samplewt_MC(self, total_lumi):
        self.samplewt = total_lumi * self.xsec / 1000
        self.samplewt /= self.numevts

    def set_samplewt_data(self):
        self.samplewt = 1.0

    def plot(self, global_min, global_max, nbins, logx, density, ax):
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

        return simon_histplot(self.H, density, label, 
                              ax=ax_main, color=self.color)

class PlottableDatasetStack:
    def __init__(datasets):
        self.dataset = datasets

class KinPlotManager:
    def __init__(self):
        self.dfs_MC = []
        self.labels_MC = []
        self.xsecs_MC = []
        self.numevts_MC = []

        self.dfs_data = None
        self.lumis = None

        self.folder = None
        self.show=True

    def add_MC_stack(self, dfs, labels, 
                     xsecs, numevts):
        self.dfs_MC.append(dfs)
        self.labels_MC.append(labels)
        self.xsecs_MC.append(xsecs)
        self.numevts_MC.append(numevts)

    def add_MC(self, df, label, 
               xsec, numevts):
        self.dfs_MC.append(df)
        self.labels_MC.append(label)
        self.xsecs_MC.append(xsec)
        self.numevts_MC.append(numevts)

    def add_data(self, df, lumi):
        if self.dfs_data is None:
            self.dfs_data = [df]
            self.lumis = [lumi]
        else:
            self.dfs_data.append(df)
            self.lumis.append(lumi)

    def setup_savefig(self, folder):
        self.folder = folder
        os.makedirs(folder, exist_ok=True)

    def toggle_show(self, show):
        self.show = show

    def plot_variable(self, toplot, cut=NoCut(),
                      weighting='evtwt_nominal', 
                      pulls=False, density=False):

        if type(toplot) is str:
            toplot = variable_from_string(toplot)

        if type(weighting) is str:
            weighting = variable_from_string(weighting)

        if type(toplot) is Variable:
            logx = config['KinAxes'][toplot.name]['logx']
            logy = config['KinAxes'][toplot.name]['logy']
            xlabel = config['KinAxes'][toplot.name]['label']
        elif type(toplot) is Ratio:
            thename = "%s_over_%s"%(toplot.num, toplot.denom)
            logx = config['KinAxes'][thename]['logx']
            logy = config['KinAxes'][thename]['logy']
            xlabel = config['KinAxes'][thename]['label']
        else:
            raise ValueError("toplot must be a string or a Variable or Ratio object")

        if self.folder is not None:
            thename = 'VAR-%s_WT-%s_CUT-%s_DENSITY-%d_PULL-%d.png' % (
                toplot.key, weighting.key, cut.key,
                density, pulls
            )
            savefig = os.path.join(self.folder, thename)
        else:
            savefig = None

        plot_variable(self.dfs_data, self.lumis,
                      self.dfs_MC, self.labels_MC, 
                      self.xsecs_MC, self.numevts_MC,
                      toplot, logx, logy, xlabel,
                      weighting=weighting,
                      cut=cut,
                      pulls=pulls, density=density,
                      savefig=savefig, show=self.show)

def plot_variable(dataset_data,
                  datasets_MC,
                  toplot, logx, logy, xlabel,
                  weighting=Variable('evtwt_nominal'),
                  cut = NoCut(),
                  pulls=False, density=True,
                  savefig=None, show=True):

    if datasets_data is None:
        DO_DATA = False
    else:
        DO_DATA = True

    if type(datasets_MC) not in [list,tuple]:
        datasets_MC = [datasets_MC]

    for dMC in datasets_MC:
        dMC.evaluate_table(toplot, cut, weighting)

    if DO_DATA:
        datasets_data.evaluate_table(toplot, cut, weighting)

    global_min = np.min([dMC.minval for dMC in datasets_MC])
    global_max = np.max([dMC.maxval for dMC in datasets_MC])

    if DO_DATA:
        global_min = np.min([global_min, datasets_data.minval])
        global_max = np.max([global_max, datasets_data.maxval])

    fig = plt.figure(figsize=config['Figure_Size'])
    try:
        (ax_main, ax_ratio) = fig.subplots(
                2, 1, sharex=True, 
                height_ratios=(1, config['Ratiopad_Height'])
        )

        if DO_DATA:
            hep.cms.label(ax=ax_main, data=True, label=config['Approval_Text'],
                          year=config['Year'], lumi=datasets_data.lumi)
        else:
            hep.cms.label(ax=ax_main, data=False, label=config['Approval_Text'])

        mainlines = {}
        for dMC in datasets_MC:
            mainlines[label] = dMC.plot(global_min, global_max, 50,
                                        logx, density, ax=ax_main)

        if DO_DATA:
            mainlines['DATA'] = datasets_data.plot(global_min, global_max, 50,
                                                   logx, density, ax=ax_main)

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
                    label=label, 
                    color=mainlines[label][0].get_color(),
                )
            if pulls:
                ax_ratio.set_ylabel("%s/MC [pulls]"%nom)
            else:
                ax_ratio.set_ylabel("%s/MC"%nom)

        ax_ratio.set_xlabel(xlabel)

        ax_ratio.axhline(1, color='black', linestyle='--')

        ax_ratio.set_ylim(0.0, 2.0)

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

        if savefig is not None:
            plt.savefig(savefig, dpi=300, bbox_inches='tight', format='png')

        if show:
            plt.show()
    finally:
        plt.close(fig)
