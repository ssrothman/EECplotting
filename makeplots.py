import argparse
import matplotlib.pyplot as plt
import json
import pickle
from proj import *
from binningloop import binningloop
from loaddata import loaddata
import matplotlib

matplotlib.use('Agg')

argparser = argparse.ArgumentParser(description='Make plots for EEC')
argparser.add_argument('json', type=str, help='Path to plot command json')

args = argparser.parse_args()

with open(args.json, 'r') as f:
    command = json.load(f)

vals = []
covs = []
isdata = []

for dd in command['data']:
    v, c, d = loaddata(dd)
    vals.append(v)
    covs.append(c)
    isdata.append(d)

for command in command['commands']:
    if command['command'] == 'compare':
        which = command['which']
        whichvals = [vals[i] for i in which]
        whichcovs = [covs[i] for i in which]
        whichisdata = [isdata[i] for i in which]

        for binning_l in binningloop(command):
            compareProjectedEEC(whichvals, whichcovs,
                                binning_l,
                                density=True,
                                label_l = command['labels'],
                                color_l = command['colors'],
                                isData = any(whichisdata),
                                wrt = command['wrt'],
                                folder= command['folder'],
                                fprefix=command['fprefix'])

    elif command['command'] == 'correlation':
        for binning1, binning2 in binningloop(command):
            pass

    break
