from datasets import get_dataset, get_counts
import plotters.kin

from importlib import reload
import numpy as np
import json

skimmer = 'Kinematics'
which = 'event'

with open("config.json", 'r') as f:
    config = json.load(f)

datasets = {}
for dataset in config['DatasetsMC'].keys():
    tag = config['DatasetsMC'][dataset]['tag']
    label = config['DatasetsMC'][dataset]['label']
    xsec = config['DatasetsMC'][dataset]['xsec']

    df = get_dataset('Apr_23_2025', tag, skimmer, 'nominal', which)
    count = get_counts('Apr_23_2025', tag)
    datasets[dataset] = {
        'df' : df,
        'label' : label,
        'xsec' : xsec,
        'numevts' : count
    }

plotter = plotters.kin.KinPlotManager()

for dataset in datasets.keys():
    if 'Pythia' in dataset:
        plotter.add_MC(**datasets[dataset])

#plotter.add_MC(df_pythia, 'Pythia', 1, count_pythia)
#plotter.add_MC(df_herwig, "Herwig", 1, count_herwig)
#plotter.add_MC(df_HT0to70, 'Pythia HT0to70', 1, count_HT0to70)
#plotter.add_MC(df_HT70to70, 'Pythia HT70to100', 1, count_HT70to100)
#plotter.add_MC(df_HT100to200, 'Pythia HT100to200', 1, count_HT100to200)
#plotter.add_MC(df_HT200to400, 'Pythia HT200to400', 1, count_HT200to400)
#plotter.add_MC(df_HT400to600, 'Pythia HT400to600', 1, count_HT400to600)
#plotter.add_MC(df_HT600to800, 'Pythia HT600to800', 1, count_HT600to800)
#plotter.add_MC(df_HT800to1200, 'Pythia HT800to1200', 1, count_HT800to1200)
#plotter.add_MC(df_HT1200to2500, 'Pythia HT1200to2500', 1, count_HT1200to2500)
#plotter.add_MC(df_HT2500toInf, 'Pythia HT2500toInf', 1, count_HT2500toInf)
#plotter.add_MC(df_herwig, 'Herwig/2', 0.5, count_herwig)
#plotter.add_MC(df_herwig, 'Herwig*2', 2.0, count_herwig)
#plotter.add_data(df_dataA)
#plotter.add_data(df_dataB)
#plotter.add_data(df_dataC)
#plotter.add_data(df_dataD)

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
