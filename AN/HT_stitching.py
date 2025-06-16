from datasets import get_dataset, get_counts
import plotters.kin

from importlib import reload
import numpy as np
import json

skimmer = 'Kinematics'
which = 'event'

plotter = plotters.kin.KinPlotManager()

plotter.toggle_show(True)

datasets_MC = plotters.kin.setup_datasets_MC(skimmer, which)
datasets_data = plotters.kin.setup_datasets_DATA(skimmer, which)

plotter.add_MC(datasets_MC['Pythia_inclusive'])
plotter.add_MC(datasets_MC['HT'])
plotter.setup_savefig("AN/HT_stitching/")

plotter.toggle_show(False)
plotter.plot_variable("genHT", clamp_ratiopad = [0.5, 1.5])

which = 'jet'

plotter = plotters.kin.KinPlotManager()

plotter.toggle_show(True)

datasets_MC = plotters.kin.setup_datasets_MC(skimmer, which)
datasets_data = plotters.kin.setup_datasets_DATA(skimmer, which)

plotter.add_MC(datasets_MC['Pythia_inclusive'])
plotter.add_MC(datasets_MC['HT'])
plotter.setup_savefig("AN/HT_stitching/")

plotter.toggle_show(False)
plotter.plot_variable("pt", clamp_ratiopad = [0.5, 1.5])
