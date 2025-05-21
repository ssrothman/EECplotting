from datasets import get_dataset, get_counts
import plotters.kin

from importlib import reload
import numpy as np
import json

skimmer = 'Kinematics'
which = 'jet'

plotter = plotters.kin.KinPlotManager()

plotter.toggle_show(True)

datasets_MC = plotters.kin.setup_datasets_MC(skimmer, which)
datasets_data = plotters.kin.setup_datasets_DATA(skimmer, which)

plotter.add_MC(datasets_MC['Pythia_inclusive'])
plotter.add_MC(datasets_MC['Herwig_inclusive'])
plotter.add_data(datasets_data['allDATA'])

plotter.plot_variable("pt")
