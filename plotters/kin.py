import matplotlib.pyplot as plt
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
    
class Ratio:
    def __init__(self, num, denom):
        self.num = num
        self.denom = denom

    @property
    def columns(self):
        return [self.num, self.denom]

    def evaluate(self, table):
        return table[self.num].to_numpy()/table[self.denom].to_numpy()

class Product:
    def __init__(self, var1, var2):
        self.var1 = var1
        self.var2 = var2

    @property
    def columns(self):
        return [self.var1, self.var2]

    def evaluate(self, table):
        return table[self.var1].to_numpy() * table[self.var2].to_numpy()

class NoCut:
    def __init__(self):
        pass

    @property
    def columns(self):
        return []

    def evaluate(self, table):
        return np.ones(table.num_rows, dtype=bool)

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

    def plot_variable(self, toplot, cut=NoCut(),
                      weighting='evtwt_nominal', 
                      pulls=False, density=True):

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

        plot_variable(self.dfs_data, self.dfs_MC, self.labels_MC,
                      toplot, logx, logy, xlabel,
                      weighting=weighting,
                      cut=cut,
                      pulls=pulls, density=density)

def plot_variable(dfs_data, dfs_MC, labels_MC, 
                  toplot, logx, logy, xlabel,
                  weighting=Variable('evtwt_nominal'),
                  cut = NoCut(),
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


    needed_columns = list(set(toplot.columns + 
                              cut.columns +
                              weighting.columns))

    tables_MC = []
    for df in dfs_MC:
        tables_MC.append(
            df.to_table(columns=needed_columns)
        )

    if DO_DATA:
        tables_data = []
        for df in dfs_data:
            tables_data.append(
                df.to_table(columns=needed_columns)
            )

    vals_MC = []
    wts_MC = []
    for table in tables_MC:
        mask = cut.evaluate(table)
        vals_MC.append(toplot.evaluate(table)[mask])
        wts_MC.append(weighting.evaluate(table)[mask])

    minvals_MC = [np.min(val) for val in vals_MC]
    maxvals_MC = [np.max(val) for val in vals_MC]

    minval = np.min(minvals_MC)
    maxval = np.max(maxvals_MC)

    if DO_DATA:
        vals_data = []
        wts_data = []
        for table in tables_data:
            mask = cut.evaluate(table)
            vals_data.append(toplot.evaluate(table)[mask])
            wts_data.append(weighting.evaluate(table)[mask])

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

    for val, wt, label in zip(vals_MC, wts_MC, labels_MC):
        H.fill(
            val,
            weight=wt,
            label=label,
        )

    if DO_DATA:
        for val, wt in zip(vals_data, wts_data):
            H.fill(
                val,
                weight=wt,
                label='DATA'
            )

    fig = plt.figure(figsize=config['Figure_Size'])
    try:
        (ax_main, ax_ratio) = fig.subplots(
                2, 1, sharex=True, 
                height_ratios=(1, config['Ratiopad_Height'])
        )

        if DO_DATA:
            hep.cms.label(ax=ax_main, data=True, label=config['Approval_Text'],
                          year=config['Year'], lumi=config['Lumi'])
        else:
            hep.cms.label(ax=ax_main, data=False, label=config['Approval_Text'])

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
