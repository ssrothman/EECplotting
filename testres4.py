from datasets import get_pickled_histogram, get_unfolded_histogram, get_pickled_histogram_sum
import plotters.res4
from importlib import reload
import numpy as np
import matplotlib.pyplot as plt

import json
with open("config/datasets.json", 'r') as f:
    datasets = json.load(f)

tags = [datasets['DatasetsMC'][key]['tag'] for key in datasets['DatasetsMC'] if 'HT-' in key]
xsecs = [datasets['DatasetsMC'][key]['xsec'] for key in datasets['DatasetsMC'] if 'HT-' in key]

Hsum = get_pickled_histogram_sum(tags, xsecs, 'Apr_23_2025', 'EECres4tee', 'nominal', 'nominal', 'transfer')

Hpythia = get_pickled_histogram("Apr_23_2025", 'Pythia_inclusive', 'EECres4tee', 'nominal', 'nominal', 'transfer')

plotters.res4.plot_purity_stability(Hsum, 'R', {})
