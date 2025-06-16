from datasets import get_procpkl
import json

with open("config/datasets.json", 'r') as f:
    datasets = json.load(f)

cutflows_counts = {}
for sample in datasets['DatasetsMC'].keys():
    thetag = datasets['DatasetsMC'][sample]['tag']
    cutflows_counts[thetag] = (
            get_procpkl('Apr_23_2025', thetag, 'Cutflow')['nominal'],
            get_procpkl('Apr_23_2025', thetag, 'Count')
    )

evt_denom = 'noiseFilters'
jet_denom = 'vetomap'
num = 'nbtag'

name = []
evt_eff = []
jet_eff = []
xsec = []
count = []
label = []

for sample in cutflows_counts.keys():
    name.append(sample)
    evt_eff.append(
        cutflows_counts[sample][0]['evt'][num] / cutflows_counts[sample][0]['evt'][evt_denom]
    )
    jet_eff.append(
        cutflows_counts[sample][0]['jet'][num] / cutflows_counts[sample][0]['jet'][jet_denom]
    )
    xsec.append(
        datasets['DatasetsMC'][sample]['xsec']
    )
    label.append(
        datasets['DatasetsMC'][sample]['label']
    )
    count.append(
        cutflows_counts[sample][1]
    )

import numpy as np
order = np.flip(np.argsort(xsec))
with open("AN/bkgveto_efficiency.tex", 'w') as f:
#    f.write("\\begin{table}[h]\n")
#    f.write("\\centering\n")
    f.write("\\begin{tabular}{l|l|l}\n")
    f.write("\tSample & Event Efficiency [\\%] & Jet Efficiency [\\%]\\\\\n")
    f.write("\t\\hline\n")
    for i in order:
        if "<" in label[i]:
            continue
        thelabel = label[i].replace("<", "$<$")
        f.write("\t%s & %0.2f & %0.2f\\\\\n"%(thelabel, 100*evt_eff[i], 100*jet_eff[i]))
    f.write("\\end{tabular}\n")
#    f.write("\\caption{Background veto efficiency for all MC samples}\n")
#    f.write("\\label{tab:vetoeff}\n")
#    f.write("\\end{table}\n")
