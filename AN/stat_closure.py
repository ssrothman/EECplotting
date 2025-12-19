import filenames
import plotters.res4
import datasets
import os

systlist = ['scale', 'isosf', 'idsf', 'triggersf',
            'PU', 'PDF', 'aS', 'PDFaS',
            'ISR', 'FSR',
            'CH', 'JES', 'JER', 'UNCLUSTERED',
            'TRK_EFF']

RUN_1d_folder = filenames.run_folder(
    'Apr_23_2025', 'Pythia_HTsum', -1,
    2, 1, -1,
    'nominal', 'nominal',
    'Apr_23_2025', 'Pythia_HTsum', -1,
    2, 0, -1,
    systlist, False, False,
    'RUN_2025_07_01-11_31_33'
)

RUN_2d_folder = filenames.run_folder(
    'Apr_23_2025', 'Pythia_HTsum', -1,
    2, 1, -1,
    'nominal', 'nominal',
    'Apr_23_2025', 'Pythia_HTsum', -1,
    2, 0, -1,
    systlist, False, False,
    'RUN_2025_07_01-11_21_06'
)


RUN_1d_smooth_folder = filenames.run_folder(
    'Apr_23_2025', 'Pythia_HTsum', -1,
    2, 1, -1,
    'nominal', 'nominal',
    'Apr_23_2025', 'Pythia_HTsum', -1,
    2, 0, -1,
    systlist, False, True,
    'RUN_2025_07_01-11_28_20'
)

RUN_2d_smooth_folder = filenames.run_folder(
    'Apr_23_2025', 'Pythia_HTsum', -1,
    2, 1, -1,
    'nominal', 'nominal',
    'Apr_23_2025', 'Pythia_HTsum', -1,
    2, 0, -1,
    systlist, False, True,
    'RUN_2025_07_01-11_23_03'
)

Hgen = datasets.get_pickled_histogram(
    'Apr_23_2025', 'Pythia_HTsum', 'EECres4tee',
    'nominal', 'nominal', 'gen', 
    2, 0, -1, -1, None, -1
)

Hreco = datasets.get_pickled_histogram(
    'Apr_23_2025', 'Pythia_HTsum', 'EECres4tee',
    'nominal', 'nominal', 'gen', 
    2, 1, -1, -1, None, -1
)

import closureplots
from importlib import reload
