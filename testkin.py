from datasets import get_dataset, get_counts
import plotters.kin

from importlib import reload
import numpy as np
import json

skimmer = 'Kinematics'
which = 'jet'

#df = get_dataset('Apr_23_2025', tag, skimmer, 'nominal', which)
#count = get_counts('Apr_23_2025', tag)

plotter = plotters.kin.KinPlotManager()

plotter.toggle_show(True)
#plotter.setup_savefig('test_kin/parts')

#from tqdm import tqdm
#
#for key in df_dataD.schema:
#    toplot = key.name
#    if toplot.startswith('evtwt'):
#        continue
#    if toplot in ['CHSpt', 'CHSeta', 'CHSphi']:
#        continue
#
#    print("plotting", toplot)
#    plotter.plot_variable(toplot)
#
#if which == 'part':
#    plotter.plot_variable('pt_over_jetPt')
