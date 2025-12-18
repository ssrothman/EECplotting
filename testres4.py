import argparse

parser = argparse.ArgumentParser(description='Run the minimizer for EEC reconstruction')
parser.add_argument('RecoTag', type=str)
parser.add_argument('RecoSample', type=str)
parser.add_argument('--reco_nboot', type=int, default=-1)
parser.add_argument('--reco_statN', type=int, default=-1)
parser.add_argument('--reco_statK', type=int, default=-1)
parser.add_argument('--reco_firstN', type=int, default=-1)
parser.add_argument('--reco_wtsyst', type=str, default='nominal')
parser.add_argument('--reco_objsyst', type=str, default='nominal')

parser.add_argument('GenTag', type=str)
parser.add_argument('GenSample', type=str)
parser.add_argument('--gen_nboot', type=int, default=-1)
parser.add_argument('--gen_statN', type=int, default=-1)
parser.add_argument('--gen_statK', type=int, default=-1)
parser.add_argument('--gen_firstN', type=int, default=-1)

parser.add_argument("--run", type=str, default=None)

parser.add_argument('--systlist', type=str, nargs='*',
                    default=['scale', 'isosf', 'idsf', 'triggersf',
                             'PU', 'PDF', 'aS', 'PDFaS',
                             'ISR', 'FSR',
                             'CH', 'JES', 'JER', 'UNCLUSTERED',
                             'TRK_EFF'])

parser.add_argument('--help_condition', type=float, default=0.001)
parser.add_argument('--testcut', action='store_true')

args = parser.parse_args()

import filenames
import os

reco_folder = filenames.reco_folder(
    args.RecoTag, args.RecoSample, args.reco_nboot,
    args.reco_statN, args.reco_statK, args.reco_firstN,
    args.reco_objsyst, args.reco_wtsyst, args.testcut
)
loss_folder = filenames.loss_folder(
    args.GenTag, args.GenSample, args.gen_nboot,
    args.gen_statN, args.gen_statK, args.gen_firstN,
    args.systlist, args.testcut
)
loss_name = os.path.basename(loss_folder)

if args.run is None:
    run_options = os.listdir(os.path.join(reco_folder, loss_name))
    run_options = list(filter(lambda x: x.startswith('RUN'), run_options))
    print("options:")
    for run in run_options:
        if run.startswith('RUN'):
            print(run)

base_folder = os.path.join(reco_folder, loss_name, args.run)

import plotters.res4
from importlib import reload
import numpy as np
import matplotlib.pyplot as plt
import datasets
import ioutil

x = ioutil.wrapped_read_np(os.path.join(base_folder, 'minimization_result', 'x.npy'))
reco = ioutil.wrapped_read_np(os.path.join(base_folder, 'minimization_result', 'RECO.npy'))
recoerr = ioutil.wrapped_read_np(os.path.join(base_folder, 'minimization_result', 'RECOERR.npy'))

Hinv = ioutil.wrapped_read_np(os.path.join(base_folder, 'minimization_result', 'INVHESS.npy'))

Hunf = ioutil.wrapped_read_pickle(os.path.join(base_folder, 'minimization_result', 'Hunf_boot5000.pkl'))
Hfwd = ioutil.wrapped_read_pickle(os.path.join(base_folder, 'minimization_result', 'Hfwd_boot5000.pkl'))

gen = datasets.get_pickled_histogram(args.RecoTag, args.RecoSample, 'EECres4tee', 
                                     args.reco_objsyst, args.reco_wtsyst, 'gen',
                                     args.reco_statN, args.reco_statK, 
                                     -1, args.reco_firstN,
                                     None, -1)


reco = datasets.get_pickled_histogram(args.RecoTag, args.RecoSample, 'EECres4tee', 
                                     args.reco_objsyst, args.reco_wtsyst, 'reco',
                                     args.reco_statN, args.reco_statK, 
                                     -1, args.reco_firstN,
                                     None, -1)

print("Reco chi2:")
print("\t1D:", plotters.res4.chi2_1d(reco, Hfwd))
print("\t2D:", plotters.res4.chi2_2d(reco, Hfwd))
print("Gen chi2:")
print("\t1D:", plotters.res4.chi2_1d(gen, Hfwd))
print("\t2D:", plotters.res4.chi2_2d(gen, Hfwd))
