import plotters.kin

from importlib import reload
import numpy as np
import json

skimmer = 'Kinematics'

EVT_datasets_MC = plotters.kin.setup_datasets_MC(skimmer, 'event')
EVT_datasets_data = plotters.kin.setup_datasets_DATA(skimmer, 'event')

EVT_allData = EVT_datasets_data['allDATA']
EVT_allMC = EVT_datasets_MC['allMC']

EVT_allMC.set_samplewt_MC(EVT_allData.lumi)



JET_datasets_MC = plotters.kin.setup_datasets_MC(skimmer, 'jet')
JET_datasets_data = plotters.kin.setup_datasets_DATA(skimmer, 'jet')

JET_allData = JET_datasets_data['allDATA']
JET_allMC = JET_datasets_MC['allMC']

JET_allMC.set_samplewt_MC(JET_allData.lumi)





DATA_labels = []
EVT_DATA_yields = []
for sub in EVT_allData.datasets:
    DATA_labels.append(sub.label)
    EVT_DATA_yields.append(sub.estimate_yield())
EVT_total_DATA = np.sum(EVT_DATA_yields)

MC_labels = []
EVT_MC_yields = []
for sub in EVT_allMC.datasets:
    MC_labels.append(sub.label)
    EVT_MC_yields.append(sub.estimate_yield())
EVT_total_MC = np.sum(EVT_MC_yields)



JET_DATA_yields = []
for sub in JET_allData.datasets:
    JET_DATA_yields.append(sub.estimate_yield())
JET_total_DATA = np.sum(JET_DATA_yields)

JET_MC_yields = []
for sub in JET_allMC.datasets:
    JET_MC_yields.append(sub.estimate_yield())
JET_total_MC = np.sum(JET_MC_yields)



with open("AN/yields.tex", 'w') as f:
    f.write("\\begin{tabular}{l|r|r|r|r|}\n")
    f.write("\tSample & Event yield &  [\\%] & Jet yield &  [\\%]\\\\\n")
    f.write("\\hline\n")
    for i in range(len(DATA_labels)):
        evtyield = int(EVT_DATA_yields[i]+0.5)
        jetyield = int(JET_DATA_yields[i]+0.5)
        f.write("%s & %s & %0.2f & %s & %0.2f\\\\\n"%(
            DATA_labels[i],
            f'{evtyield:,}',
            EVT_DATA_yields[i]/EVT_total_DATA*100,
            f'{jetyield:,}',
            JET_DATA_yields[i]/JET_total_DATA*100,
        ))
    f.write('\\hline\n')
    MCorder = np.flip(np.argsort(EVT_MC_yields))
    for i in MCorder:
        evtyield = int(EVT_MC_yields[i]+0.5)
        jetyield = int(JET_MC_yields[i]+0.5)
        f.write("%s & %s & %0.2f & %s & %0.2f\\\\\n"%(
            MC_labels[i],
            f'{evtyield:,}',
            EVT_MC_yields[i]/EVT_total_MC*100,
            f'{jetyield:,}',
            JET_MC_yields[i]/JET_total_MC*100,
        ))
    f.write('\\hline\n')
    f.write('\\hline\n')
    evtyield = int(EVT_total_DATA+0.5)
    jetyield = int(JET_total_DATA+0.5)
    f.write("%s & %s & %0.2f & %s & %0.2f\\\\\n"%(
        "Total Data",
        f'{evtyield:,}',
        100,
        f'{jetyield:,}',
        100
    ))
    f.write('\\hline\n')
    evtyield = int(EVT_total_MC+0.5)
    jetyield = int(JET_total_MC+0.5)
    f.write("%s & %s & %0.2f & %s & %0.2f\\\\\n"%(
        "Total MC",
        f'{evtyield:,}',
        100,
        f'{jetyield:,}',
        100
    ))

    f.write("\\end{tabular}\n")
