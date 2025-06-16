from datasets import get_pickled_histogram, get_unfolded_histogram, get_pickled_histogram_sum
import plotters.res4
from importlib import reload
import numpy as np
import matplotlib.pyplot as plt

#import json
#with open("config/datasets.json", 'r') as f:
#    datasets = json.load(f)
#
#tags = [datasets['DatasetsMC'][key]['tag'] for key in datasets['DatasetsMC'] if 'HT-' in key]
#xsecs = [datasets['DatasetsMC'][key]['xsec'] for key in datasets['DatasetsMC'] if 'HT-' in key]
#
#Hsum = get_pickled_histogram_sum(tags, xsecs, 'Apr_23_2025', 'EECres4tee', 'nominal', 'nominal', 'transfer')
#
#Hpythia = get_pickled_histogram("Apr_23_2025", 'Pythia_inclusive', 'EECres4tee', 'nominal', 'nominal', 'transfer')
#
#plotters.res4.plot_purity_stability(Hsum, 'R', {})

import pickle

with open("/home/submit/srothman/cmsdata/EEC/Apr_23_2025/Herwig_inclusive/EECres4tee/hists_file0to124_MC/nominal/gen_nominal_nominal_boot100_rng0_2stat1.pkl", 'rb') as f:
    gen = pickle.load(f)

with open("/home/submit/srothman/cmsdata/EEC/Apr_23_2025/Herwig_inclusive/EECres4tee/hists_file0to124_MC/nominal/reco_nominal_nominal_boot100_rng0_2stat1.pkl", 'rb') as f:
    reco = pickle.load(f)

with open("/home/submit/srothman/work/EEC/EECunfold/PYTHON_UNFOLD/results/H1d.pkl", 'rb') as f:
    unf1d = pickle.load(f)

with open("/home/submit/srothman/work/EEC/EECunfold/PYTHON_UNFOLD/results/H1d_stat.pkl", 'rb') as f:
    unf1d_stat = pickle.load(f)

with open("/home/submit/srothman/work/EEC/EECunfold/PYTHON_UNFOLD/results/H2d.pkl", 'rb') as f:
    unf2d = pickle.load(f)

with open("/home/submit/srothman/work/EEC/EECunfold/PYTHON_UNFOLD/results/H2d_stat.pkl", 'rb') as f:
    unf2d_stat = pickle.load(f)
