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

plotter.add_MC(datasets_MC['allMCHerwig'])
plotter.add_data(datasets_data['allDATA'])

plotter.toggle_show(False)
plotter.setup_savefig("AN/kinematics_event_herwig_kinreweight")

kinreweight = plotters.kin.CorrectionlibVariable(
    [
        'Zpt',
        plotters.kin.UFuncVariable('Zy', np.abs)
    ],
    'kinSF/Zkin.json', 
    'Herwig_Zkinweight'
)

weighting = plotters.kin.Product(
    'evtwt_nominal', 
    kinreweight,
)

for variable in datasets_data['DATA_2018A'].df.schema.names:
    if 'evtwt' in variable:
        continue

    print(variable)
    plotter.plot_variable(variable, clamp_ratiopad = [0.0, 2.0],
                          weighting_MC=weighting)

