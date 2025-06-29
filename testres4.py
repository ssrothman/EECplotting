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

parser.add_argument("Run", type=str)

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

base_folder = os.path.join(reco_folder, loss_name, run)

import plotters.res4
from importlib import reload
import numpy as np
import matplotlib.pyplot as plt
import datasets
import ioutil
