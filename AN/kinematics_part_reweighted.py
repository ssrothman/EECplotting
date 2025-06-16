import plotters.kin

from importlib import reload
import numpy as np
import json


import dill as pickle
with open("kinSF/pythia_jetpt_SF.pkl", 'rb') as f:
    func = pickle.load(f)
with open("kinSF/fit_pythia.pkl", 'rb') as f:
    data = pickle.load(f)

weighting = plotters.kin.Product(
    plotters.kin.Variable("evtwt_nominal"),
    plotters.kin.UFuncVariable("pt", lambda pt:func.evaluate(pt, data))
    #plotters.kin.Variable("pt")
)

skimmer = 'Kinematics'
which = 'part'

plotter = plotters.kin.KinPlotManager()

plotter.toggle_show(True)

datasets_MC = plotters.kin.setup_datasets_MC(skimmer, which)
datasets_data = plotters.kin.setup_datasets_DATA(skimmer, which)

plotter.add_MC(datasets_MC['allMC'])
plotter.add_data(datasets_data['allDATA'])

plotter.toggle_show(False)
plotter.setup_savefig("AN/kinematics_part_reweighted")

for variable in ['pt', 'pt_over_jetPt']:
    if 'evtwt' in variable:
        continue

    print(variable)
    plotter.plot_variable(variable, clamp_ratiopad = [0.0, 2.0],
                          weighting_MC = weighting)

