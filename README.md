# Plotting

This is the plotting repo. I strongly recommend having a relatively new version of matplotlib because in newer matplotlib the legend placement is much smarter.

Kinematics plotting is handled in plotters/kin.py. This runs straight off the .parquet datasets 

EEC res4 plotting is handled in plotters/res4.py. This requires binned histograms.

I don't have a projected EEC plotter at the moment, but it should be easy to put together.

# How to actually run the plotting

An example usage of the kinematics plotter is demo'd in testkin.py

An example usage of the res4 plotter in demo'd in testres4.py

Both of them rely on:

 - datasets.py needs to be pointed to where your datasets are. I've hard-coded the path I use, so that will need to be changed
 - config/config.json configures the plotting
 - config/datasets.json configures the datasets lookup. The format of this is:
```json
{
    "DatasetsGENONLY" : {
        "dataset1" : {
            "tag" : "unique identified string",
            "label" : "label to write on plot key",
            "color" : "color to uyse in plot" 
        },
        ...
    },
    "DatasetsDATA" : {
        "dataset1" : {
            "tag" : "unique identified string",
            "label" : "label to write on plot key",
            "color" : "color to uyse in plot",
            "lumi" : "integrated luminosity"
        },
        ...
    },
    "DatasetsMC" : {
        "dataset1" : {
            "tag" : "unique identified string",
            "label" : "label to write on plot key",
            "color" : "color to uyse in plot",
            "xsec" : cross-section
        },
        ...
    },
    "StacksMC" : {
        "stack1" : {
            "tag" : "unique identifier string",
            "global_label" : "label to write on plot key",
            "global_color" : "color to use in plot",
            "dsets" : ["dataset1", "dataset2", ...]
            "stacks" : ["stackA", "stackB", ...]
        },
        ...
    },
    "StacksDATA" : {
        "stack1" : {
            "tag" : "unique identifier string",
            "global_label" : "label to write on plot key",
            "global_color" : "color to use in plot",
            "dsets" : ["dataset1", "dataset2", ...]
            "stacks" : ["stackA", "stackB", ...]
        },
    }
}
```

Note there there are both "datasets" = the actual datasets and "stacks" = collections of datasets. The stacks are made up of a list of datasets and other stacks. The stack building code in plotters/kin.py tries to be inteligent about solving the resulting dependency chain. Stacks can be plotted as a total histogram or as resolved stacked histograms. When ploitted as a total histogram the global\_label and global\_color attributes are used. Nested stacks are always plotted as total histograms.


# Kinematics 

The kinematics plotting is somewhat sophisticated, and does not rely on any binned histograms. configs/config.py has the configuration for the actually binning, which is run JIT on the parquet datasets. The main driver is the KinPlotManager class, as demonstrated in the testkin.py script. Hopefully most of the methods here are reasonably obvious in function. Once the configuration and datasets are set as desired the plotting function is plot\_variable(). I'll spell out the arguments to this function in detail:
```python

def plot_variable(self, toplot, cut, weighting, pulls, density, noResolved):
    @param toplot: the variable to plot. This can be the string name of a column in the parquet dataset, or an instance of one of the Variable, Ratio, or Product classes, which describe functions of columns. 
    @param cut: the selection to apply when plotting. The default is an instance of NoCut() which does nothing. You can pass an instance of the NoCut, TwoSidedCut, GreaterThanCut, or LessThanCut classes, which take variables as in the toplot argument, as well as the cut values, in the constructor. 
    @param weighting: the weighting for each entry. This is of the same format as the toplot argument.
    @param pulls: whether to plot the ratiopad in units of pulls
    @param density: whether to normalize the histograms
    @param noResolved: force any stacks to be plotted as a total histogram. If false, there is some heuristic about what you probably wanted to do that dictates whether stacks should be resolved. 
```

# Res4

This is less sophisticated, and doesn't have a fancy driver class. Again we rely on the config files and datasets.py to actually get the data off disk from the right place and weight it correctly. 
